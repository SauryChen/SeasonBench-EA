import os
import copy
import torch
import numpy as np
torch.autograd.set_detect_anomaly(True)
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import lightning.pytorch as pl
import yaml
from pathlib import Path

from ..criterion import RMSE_train, VAELoss
from .dataset import NWP_Dataset

class NWPCorrection(pl.LightningModule):
    def __init__(
        self,
        model_args,
        data_args,
    ):
        super(NWPCorrection, self).__init__()
        self.save_hyperparameters()
        self.model_args = model_args
        self.data_args = data_args

        input_size = len(self.model_args['input_vars']['pressure_levels'] + self.model_args['input_vars']['single_level'] + self.model_args['input_cons'])
        output_size = len(self.model_args['output_vars'])
        input_size = input_size * 6 # 6 months
        output_size = output_size * 6 # 6 months

        self.lat = np.load(os.path.join(self.data_args['data_dir'], 'ERA5_monthly_mean/global_1deg/processed_data', 'lat_lon.npz'), allow_pickle=True)['lat']
        top = (len(self.lat) - self.data_args['crop_size'][0]) // 2
        self.lat = self.lat[top:top + self.data_args['crop_size'][0]]

        self.used_var_indices = self.find_used_vars()

        if self.model_args['model_name'] == 'fno':
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

        elif self.model_args['model_name'] == 'sfno':
            from ..models.sfno import SphericalFourierNeuralOperatorNet
            self.model = SphericalFourierNeuralOperatorNet(
                img_size = self.data_args['crop_size'],
                scale_factor = self.model_args['scale_factor'],
                in_chans = input_size,
                out_chans = output_size,
                embed_dim = self.model_args['embed_dim'],
                num_layers = self.model_args['num_layers'],
                use_mlp = self.model_args['use_mlp'],
            )
            self.loss = self.init_loss_fn()
        
        elif self.model_args['model_name'] == 'unet':
            from ..models.unet import UNet
            self.model = UNet(
                in_channels=input_size,
                out_channels=output_size,
                features=self.model_args['features'],
            )
            self.loss = self.init_loss_fn()
        
        elif self.model_args['model_name'] == 'vit':
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
        
        elif self.model_args['model_name'] == 'graphcast':
            from ..models.graph_cast_net import GraphCastNet
            self.model = GraphCastNet(
                input_res = self.data_args['crop_size'],
                input_dim_grid_nodes = input_size,
                output_dim_grid_nodes = output_size,
                processor_layers = self.model_args['processor_layers'],
                multimesh = self.model_args['multimesh'],
                mesh_level = self.model_args['mesh_level'],
                hidden_layers = self.model_args['hidden_layers'],
                hidden_dim = self.model_args['hidden_dim'],
            )
            self.loss = self.init_loss_fn()

        elif self.model_args['model_name'] == 'vae':
            from ..models.vae import VAE
            self.model = VAE(
                in_channels=input_size,
                latent_dim=self.model_args['latent_dim'],
                input_shape=self.data_args['crop_size'],
            )
            self.loss = self.init_loss_vae()
            self.loss_val = self.init_loss_fn()

        else:
            raise NotImplementedError(f"Model {self.model_args['model_name']} not implemented.")
    
    def init_loss_fn(self):
        loss = RMSE_train(is_weight=True, lat=self.lat)
        return loss
    
    def init_loss_vae(self):
        loss = VAELoss(recon_weight=self.model_args['recon_weight'], kld_weight=self.model_args['kld_weight'], is_weight=True, lat=self.lat)
        return loss

    def find_used_vars(self):
        """
            Find the index of input variables if it is used in the output variables for all the steps
            e.g. ['t2m','t_850','z_500','q_700','tp']
        """
        input_var_flat = (
            self.data_args['input_vars']['pressure_levels'] + 
            self.data_args['input_vars']['single_level'] + 
            self.data_args['input_cons'])
        input_var_flat = ['tp' if var == 'tprate' else var for var in input_var_flat]
        used_var_indices = []
        for var in self.data_args['output_vars']:
            if var in input_var_flat:
                idx = input_var_flat.index(var)
                used_var_indices.append(idx)
        return used_var_indices
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        """
            x_nwp: [B, C, T=6, H, W] -> [B, C*T, H, W]
            x_cons: [B, C, T=1, H, W] broadcast to [B, C*T, H, W]
            y: [B, C, T=6, H, W] -> [B, C*T, H, W]
        """
        x_nwp, x_cons, y, time = batch
        x_nwp_copy = copy.deepcopy(x_nwp) # used in learning residual # [B,C,T,H,W]

        B, C, T, H, W = x_nwp.shape
        C_ = x_cons.shape[1]
        x_nwp = x_nwp.reshape(x_nwp.shape[0], -1, x_nwp.shape[-2], x_nwp.shape[-1])
        x_cons = x_cons.expand(B, C_, T, H, W).reshape(x_cons.shape[0], -1, x_cons.shape[-2], x_cons.shape[-1])
        y = y.reshape(y.shape[0], -1, y.shape[-2], y.shape[-1])

        loss = 0
        x = torch.cat([x_nwp, x_cons], dim=1)
        x_nwp_ori = x_nwp_copy[:, self.used_var_indices, ...].reshape(x_nwp_copy.shape[0], -1, x_nwp_copy.shape[-2], x_nwp_copy.shape[-1])

        if 'vae' in self.model_args['model_name']:
            preds, mu, logvar = self.model(x)
            preds = preds + x_nwp_ori # learning residual
            loss += self.loss(preds, y, mu, logvar)
        else:
            preds = self.model(x)
            preds = preds + x_nwp_ori # learning residual
            loss += self.loss(preds, y)

        self.log('train_loss', loss, prog_bar=True, on_step=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x_nwp, x_cons, y, time = batch
        x_nwp_copy = copy.deepcopy(x_nwp) # used in learning residual # [B,C,T,H,W]

        B, C, T, H, W = x_nwp.shape
        C_ = x_cons.shape[1]
        x_nwp = x_nwp.reshape(x_nwp.shape[0], -1, x_nwp.shape[-2], x_nwp.shape[-1])
        x_cons = x_cons.expand(B, C_, T, H, W).reshape(x_cons.shape[0], -1, x_cons.shape[-2], x_cons.shape[-1])
        y = y.reshape(y.shape[0], -1, y.shape[-2], y.shape[-1])

        loss = 0
        x = torch.cat([x_nwp, x_cons], dim=1)
        x_nwp_ori = x_nwp_copy[:, self.used_var_indices, ...].reshape(x_nwp_copy.shape[0], -1, x_nwp_copy.shape[-2], x_nwp_copy.shape[-1])
        
        if 'vae' in self.model_args['model_name']:
            preds, mu, logvar = self.model(x)
            preds = preds + x_nwp_ori # learning residual
            loss += self.loss_val(preds, y)
        else:
            preds = self.model(x)
            preds = preds + x_nwp_ori # learning residual
            loss += self.loss(preds, y)

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
    