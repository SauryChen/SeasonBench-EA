from transformers import ViTConfig, ViTModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTConfig, ViTModel

class ViT_Prediction(nn.Module):
    def __init__(self, img_size=(200,400), patch_size=16, in_channels=3, out_channels=3, hidden_size=256, num_layers=6, num_heads=8, mlp_dim=512):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        config = ViTConfig(
            image_size=max(img_size), # we patch manually, so the parameter is not used
            patch_size=patch_size,
            num_channels=in_channels,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=mlp_dim,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
        )
        self.vit = ViTModel(config)
        del self.vit.embeddings
        del self.vit.pooler
        del self.vit.layernorm

        self.patch_embed = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size))
        self.output_proj = nn.Sequential(
            # number of nnConvTranspose2d layers should be changed with the patch size
            nn.ConvTranspose2d(hidden_size, hidden_size//2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_size//2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(hidden_size//2, hidden_size//4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_size//4),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_size//4, out_channels, kernel_size=3, padding=1),
        )

        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.pos_embed, std=0.02)


    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        h, w = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed 

        x = self.vit.encoder(x).last_hidden_state 
        x = x.transpose(1,2).view(B, -1, h, w)
        x = self.output_proj(x)
        return x

if __name__ == "__main__":
    import os
    import torch
    import pynvml
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from omegaconf import OmegaConf
    cfg = OmegaConf.load('../correction/vit_config.yaml')
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

    model = ViT_Prediction(
                img_size = data_args['crop_size'],
                patch_size = model_args['patch_size'],
                in_channels = input_size,
                out_channels = output_size,
                hidden_size = model_args['hidden_size'],
                num_layers = model_args['num_layers'],
                num_heads = model_args['num_heads'],
                mlp_dim = model_args['mlp_dim'],
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