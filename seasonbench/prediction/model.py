import os
import torch
torch.autograd.set_detect_anomaly(True)
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import lightning.pytorch as pl
import yaml

from ..criterion import RMSE_train, VAELoss
from .dataset import ERA5_Dataset

class SeasonalPred(pl.LightningModule):
    def __init__(
        self,
        model_args,
        data_args,
    ):
        super(SeasonalPred, self).__init__()
        self.save_hyperparameters()
        self.model_args = model_args
        self.data_args = data_args

        # Initialize model
        input_size = len(self.model_args['input_vars'] + self.model_args['input_cons'])
        output_size = len(self.model_args['output_vars'])

        if 'fno' in self.model_args['model_name']:
            from ..models.fno import FNO2d
            self.model = FNO2d(
                input_size=input_size,
                output_size=output_size,
                modes1=self.model_args['modes1'],
                modes2=self.model_args['modes2'],
                width=self.model_args['width'],
                initial_step=self.model_args['initial_step'],
            )
            self.loss = self.init_loss_fn()
        elif 'vit' in self.model_args['model_name']:
            from ..models.vit import ViT_Prediction
            self.model = ViT_Prediction(
                img_size = self.data_args['crop_size'],
                patch_size = self.model_args['patch_size'],
                in_channels = input_size,
                out_channels = output_size,
                hidden_size = self.model_args['hidden_size'],
                num_layers = self.model_args['num_layers'],
                num_heads = self.model_args['num_heads'],
                mlp_dim = self.model_args['mlp_dim'],
            )
            self.loss = self.init_loss_fn()
        elif 'vae' in self.model_args['model_name']:
            from ..models.vae import VAE
            self.model = VAE(
                in_channels=input_size,
                out_channels=output_size,
                latent_dim=self.model_args['latent_dim'],
                input_shape=self.data_args['crop_size'],
            )
            self.loss = self.init_loss_vae()
            self.loss_val = self.init_loss_fn()
        elif 'unet' in self.model_args['model_name']:
            from ..models.unet import UNet
            self.model = UNet(
                in_channels=input_size,
                out_channels=output_size,
                features=self.model_args['features'],
            )
            self.loss = self.init_loss_fn()
        else:
            raise ValueError(f"Model {self.model_args['model_name']} not recognized.")
        

    def init_loss_fn(self):
        loss = RMSE_train(is_weight = False, lat = None)
        return loss
    
    def init_loss_vae(self):
        loss = VAELoss(recon_weight=self.model_args['recon_weight'], 
                        kld_weight=self.model_args['kld_weight'],
                        is_weight = False, lat = None)
        return loss
    

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, cons, y, timestamp = batch
        y = y.permute(0, 2, 1, 3, 4)

        # iterative loss
        n_steps = y.size(1)
        loss = 0
        for step_idx in range(n_steps):
            x = torch.cat([x, cons], dim=1)

            if 'vae' in self.model_args['model_name']:
                preds, mu, logvar = self.model(x)
                loss += self.loss(preds, y[:, step_idx], mu, logvar)
            else:
                preds = self(x)
                loss += self.loss(preds, y[:, step_idx])
            x = preds
        
        loss = loss / n_steps
    
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, cons, y, timestamp = batch
        y = y.permute(0, 2, 1, 3, 4)

        # iterative loss
        n_steps = y.size(1)
        loss = 0
        for step_idx in range(n_steps):
            x = torch.cat([x, cons], dim=1)
            if 'vae' in self.model_args['model_name']:
                preds, mu, logvar = self(x)
                loss += self.loss_val(preds, y[:, step_idx])
            else:
                preds = self(x)
                loss += self.loss(preds, y[:, step_idx])
            x = preds
        
        loss = loss / n_steps
    
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=float(self.model_args['learning_rate']))
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': CosineAnnealingLR(optimizer, T_max=self.model_args['t_max'], eta_min=float(self.model_args['learning_rate']) / 10),
                'interval': 'epoch',
            }
        }
    
    def setup(self, stage=None):
        self.train_dataset = ERA5_Dataset(
            data_dir=self.data_args['data_dir'],
            input_vars=self.model_args['input_vars'],
            input_cons=self.model_args['input_cons'],
            output_vars=self.model_args['output_vars'],
            status='train',
            lead_step=self.data_args['lead_step'],
            pred_step=self.data_args['pred_step'],
            crop_size=self.data_args['crop_size'],
        )
        self.val_dataset = ERA5_Dataset(
            data_dir=self.data_args['data_dir'],
            input_vars=self.model_args['input_vars'],
            input_cons=self.model_args['input_cons'],
            output_vars=self.model_args['output_vars'],
            status='val',
            lead_step=self.data_args['lead_step'],
            pred_step=self.data_args['pred_step'],
            crop_size=self.data_args['crop_size'],
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, num_workers = self.model_args['num_workers'], batch_size = self.data_args['batch_size'], shuffle = True
        )
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, num_workers = self.model_args['num_workers'], batch_size = self.data_args['batch_size']
        )