import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, latent_dim=128, input_shape=(200, 400)):
        super(VAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), 
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1), 
            nn.ReLU(),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.ReLU(),
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, *input_shape)
            enc_output = self.encoder(dummy_input)
            self.flatten_dim = enc_output.view(1, -1).shape[1]
            self._enc_output_shape = enc_output.shape[1:]
            
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, self.flatten_dim)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, self._enc_output_shape),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.GroupNorm(8, 256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, 1, 1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, out_channels, 4, 2, 1),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        original_shape = x.shape[-2:] # (H, W)

        enc = self.encoder(x)
        flat = torch.flatten(enc, start_dim=1)
        mu = self.fc_mu(flat)
        logvar = self.fc_logvar(flat)
        z = self.reparameterize(mu, logvar)
        dec_input = self.decoder_input(z)
        recon = self.decoder(dec_input)

        if recon.shape[-2:] != original_shape:
            recon = F.interpolate(recon, size=original_shape, mode='bilinear', align_corners=False)

        return recon, mu, logvar

if __name__ == '__main__':
    import os
    import pynvml
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from omegaconf import OmegaConf
    cfg = OmegaConf.load('../correction/vae_config.yaml')
    model_args, data_args = cfg.model_args, cfg.data_args

    def print_gpu_memory():
        allocated = torch.cuda.memory_allocated(device) / 1024**2  # MB
        reserved = torch.cuda.memory_reserved(device) / 1024**2    # MB
        max_allocated = torch.cuda.max_memory_allocated(device) / 1024**2
        print(f"[Memory] Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB | Peak: {max_allocated:.2f} MB")
    
    # input_size = len(model_args['input_vars'] + model_args['input_cons'])
    input_size = len(model_args['input_vars']['pressure_levels'] + model_args['input_vars']['single_level'] + model_args['input_cons'])
    output_size = len(model_args['output_vars'])

    model = VAE(in_channels=input_size, latent_dim=model_args['latent_dim'], input_shape=data_args['crop_size']).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    print("After model init: ")
    print_gpu_memory()
    
    torch.cuda.empty_cache()
    rand_input = torch.rand((1, input_size, data_args['crop_size'][0], data_args['crop_size'][1])).to(device)
    recon, mu, logvar = model(rand_input)
    print("After model forward: ")
    print_gpu_memory()
    print("Output shape: ", recon.shape)