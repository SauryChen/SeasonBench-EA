import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv => BN => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        # Encoder
        for feat in features:
            self.downs.append(DoubleConv(in_channels, feat))
            in_channels = feat
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        # Decoder
        rev_features = features[::-1]
        for feat in rev_features:
            self.ups.append(nn.ConvTranspose2d(feat*2, feat, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feat*2, feat))

        # Final conv
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)  # upsample
            skip = skip_connections[i // 2]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])  # in case of mismatch due to odd dims
            x = torch.cat((skip, x), dim=1)
            x = self.ups[i+1](x)

        return self.final_conv(x) # [batch_size, out_channels, H, W]


if __name__ == "__main__":
    import os
    import torch
    import pynvml
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from omegaconf import OmegaConf
    cfg = OmegaConf.load('../correction/unet_config.yaml')
    model_args = cfg.model_args
    data_args = cfg.data_args

    def print_gpu_memory():
        allocated = torch.cuda.memory_allocated(device) / 1024**2  # MB
        reserved = torch.cuda.memory_reserved(device) / 1024**2    # MB
        max_allocated = torch.cuda.max_memory_allocated(device) / 1024**2
        print(f"[Memory] Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB | Peak: {max_allocated:.2f} MB")
    
    # input_size = len(model_args['input_vars'] + model_args['input_cons'])
    input_size = len(model_args['input_vars']['pressure_levels'] + model_args['input_vars']['single_level'] + model_args['input_cons'])
    output_size = len(model_args['output_vars'])

    model = UNet(
                in_channels=input_size,
                out_channels=output_size,
                features= model_args['features'],
            ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    print("After model init: ")
    print_gpu_memory()

    torch.cuda.empty_cache()
    rand_input = torch.rand((1, input_size, data_args['crop_size'][0], data_args['crop_size'][1])).to(device)
    output = model(rand_input)
    print("After model forward: ")
    print_gpu_memory()
    print("Output shape: ", output.shape)