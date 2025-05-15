import torch
import torch.nn as nn
import numpy as np
import torch.special as special
import xarray as xr
from pathlib import Path
import torch.nn.functional as F
from xskillscore import crps_ensemble, rank_histogram

def get_lat_weights(lat):
    lat = np.deg2rad(lat)
    weights = np.cos(lat)
    weights /= np.mean(weights)
    return weights


class RMSE_train(nn.Module):
    def __init__(self, is_weight=False, lat=None):
        super(RMSE_train, self).__init__()
        self.is_weight = is_weight
        self.lat = lat

    def forward(self, predictions, targets):

        squared_diff = (predictions - targets) ** 2
        if self.is_weight and self.lat is not None:
            weights = get_lat_weights(self.lat)
            squared_diff = squared_diff * torch.from_numpy(weights).view(1, 1, -1, 1).to(squared_diff.device)
        mean_squared = squared_diff.mean(dim=(2,3)) # (B, C, H, W) -> (B, C)
        mean_squared_ = mean_squared.mean(dim=1) # (B, C) -> (B,)
        rmse_per_sample = torch.sqrt(mean_squared_) # (B,)
        rmse = torch.mean(rmse_per_sample)  # (1,)
        return rmse


class VAELoss(nn.Module):
    def __init__(self, recon_weight=1.0, kld_weight=1.0, is_weight = False, lat=None):
        super(VAELoss, self).__init__()
        self.recon_weight = recon_weight
        self.kld_weight = kld_weight
        self.is_weight = is_weight
        self.lat = lat

    def forward(self, recon_x, x, mu, logvar):
        """
        Args:
            recon_x (Tensor): Reconstructed input, shape (B, C, H, W)
            x (Tensor): Original input, shape (B, C, H, W)
            mu (Tensor): Mean of the latent distribution, shape (B, latent_dim)
            logvar (Tensor): Log variance of the latent distribution, shape (B, latent_dim)
        Returns:
            loss (Tensor): Total loss = reconstruction loss + KL divergence
            recon_loss (Tensor): Reconstruction loss
            kld (Tensor): KL divergence
        """
        if self.is_weight and self.lat is not None:
            weights = get_lat_weights(self.lat)
            mse = (recon_x - x) ** 2
            recon_loss = mse * torch.from_numpy(weights).view(1, 1, -1, 1).to(mse.device)
            recon_loss = recon_loss.mean(dim=(2,3)) # (B, C, H, W) -> (B, C)
            recon_loss = torch.mean(recon_loss.mean(dim=1))
        else:
            recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = self.recon_weight*recon_loss + self.kld_weight*kld
        return loss

######################### Below used in evaluation #########################

def ensure_batch_dim(x):
    if x.ndim == 2:
        x = x.unsqueeze(0)
    return x


class RMSE(nn.Module):
    def __init__(self, is_weight=False, lat=None):
        super(RMSE, self).__init__()
        self.is_weight = is_weight

    def forward(self, predictions, targets):
        predictions, targets = ensure_batch_dim(predictions), ensure_batch_dim(targets)
        squared_diff = (predictions - targets) ** 2
        if self.is_weight:
            print("TODO: Implement weighted RMSE")
        else:
            rmse_per_sample = torch.sqrt(squared_diff.mean(dim=(-1, -2)))  # (B,)
            rmse = torch.mean(rmse_per_sample)  # (1,)
        return rmse

class Bias(nn.Module):
    def __init__(self, is_weight=False, lat=None):
        super(Bias, self).__init__()
        self.is_weight = is_weight

    def forward(self, predictions, targets):
        predictions, targets = ensure_batch_dim(predictions), ensure_batch_dim(targets)
        if self.is_weight:
            print("TODO: Implement weighted Bias")
        else:
            diff = predictions - targets
            bias_per_sample = torch.mean(diff, dim=(-1, -2))  # (B,)
            bias = torch.mean(bias_per_sample)  # (1,)
        return bias

class Willmott_Index(nn.Module):
    def __init__(self):
        super(Willmott_Index, self).__init__()
        self.eps = 1e-8
    def forward(self, predictions, targets, target_mean):
        predictions, targets = ensure_batch_dim(predictions), ensure_batch_dim(targets)
        numerator = torch.sum(torch.mean((predictions - targets) ** 2, dim=(-1, -2)))
        denominator = torch.sum(torch.mean((torch.abs(predictions - target_mean) + torch.abs(targets - target_mean)) ** 2, dim=(-1, -2)))
        wi_per_sample = 1 - numerator / (denominator + self.eps)
        wi = torch.mean(wi_per_sample)  # (1,)
        return wi


class ACC(nn.Module):
    def __init__(self, climatology: np.ndarray, is_weight: bool = False, lat: np.ndarray = None, eps: float = 1e-8, crop_size=None, lat_idx=None, lon_idx=None):
        super(ACC, self).__init__()
        self.is_weight = is_weight
        self.lat = lat
        self.eps = eps
        self.crop_size = crop_size
        self.climatology = {}
        for var, data in climatology.items():
            tensor_data = torch.tensor(data, dtype=torch.float32)
            if crop_size is not None:
                tensor_data = self.center_crop(tensor_data)
            self.climatology[var] = tensor_data
            if lat_idx is not None and lon_idx is not None:
                self.climatology[var] = tensor_data[:, lat_idx][:, :, lon_idx]
        
    def center_crop(self, arr: torch.Tensor) -> torch.Tensor:
        h, w = arr.shape[-2], arr.shape[-1]
        crop_h, crop_w = self.crop_size
        top = (h - crop_h) // 2
        left = (w - crop_w) // 2
        return arr[..., top:top + crop_h, left:left + crop_w]

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, timestamps: list[str], var_name) -> torch.Tensor:
        predictions, targets = ensure_batch_dim(predictions), ensure_batch_dim(targets)
        if self.is_weight:
            print("TODO: Implement weighted ACC")
        else:
            assert predictions.shape == targets.shape, "Predictions and targets must have the same shape"
            B, H, W = predictions.shape
            assert len(timestamps) == B, "Length of timestamps must match the batch size"

            clim = torch.stack([
                self.climatology[var_name][int(ts.split('-')[1]) - 1].to(predictions.device) for ts in timestamps
            ])
            pred_anom = predictions - clim
            obs_anom = targets - clim
            
            numerator = torch.sum(pred_anom * obs_anom, dim=(-1, -2))
            denominator = torch.sqrt(torch.sum(pred_anom ** 2, dim=(-1, -2)) * torch.sum(obs_anom ** 2, dim=(-1, -2)))
            acc_per_batch = numerator / (denominator + self.eps)
            acc = torch.mean(acc_per_batch)
            return acc # (1,)

class CSI(nn.Module):
    # for precipitation
    def __init__(self, thresholds, crop_size=None, lat_idx=None, lon_idx=None):
        """ 
        Args:
            thresholds: list or tensor, like [q50, q75, q90, q95, q99]， shape: (num_thresholds, H, W)
        """
        super(CSI, self).__init__()
        self.crop_size = crop_size

        thresholds = torch.tensor(thresholds, dtype=torch.float32)
        if crop_size is not None:
            thresholds = self.center_crop(thresholds)
        if lat_idx is not None and lon_idx is not None:
            thresholds = thresholds[:, lat_idx][:, :, lon_idx]
        
        self.register_buffer('thresholds', thresholds.clone().detach())  # (num_thresholds, H, W)


    def center_crop(self, arr: torch.Tensor) -> torch.Tensor:
        h, w = arr.shape[-2], arr.shape[-1]
        crop_h, crop_w = self.crop_size
        top = (h - crop_h) // 2
        left = (w - crop_w) // 2
        return arr[..., top:top + crop_h, left:left + crop_w]

    def forward(self, preds, targets):
        preds, targets = ensure_batch_dim(preds), ensure_batch_dim(targets)
        """
        Args:
            preds: (B, H, W)
            targets: (B, H, W)
        """
        preds = preds.float()
        targets = targets.float()
        csi_list = []

        for i, threshold in enumerate(self.thresholds.to(preds.device)):
            pred_extreme = preds > threshold.unsqueeze(0)  # (B, H, W)
            true_extreme = targets > threshold.unsqueeze(0)
            TP = (pred_extreme & true_extreme).sum(dim=(-1, -2)) # (B,)
            FN = ((~pred_extreme) & true_extreme).sum(dim=(-1, -2)) # (B,)
            FP = (pred_extreme & (~true_extreme)).sum(dim=(-1, -2)) # (B,)
            denominator = TP + FN + FP
            csi_per_sample = torch.where(
                denominator == 0,
                torch.full_like(denominator, float('nan'), dtype=torch.float32),
                TP.float() / denominator.float()
            )
            csi_mean = torch.nanmean(csi_per_sample)  # (1,)
            csi_list.append(csi_mean)
        return torch.stack(csi_list)  # (num_thresholds)

class Energy_Spectral(nn.Module):
    # zonal is more suitable for monthly mean precipitation, especially in Monsoon regions
    def __init__(self, mean_latitude=True):
        super(Energy_Spectral, self).__init__()
        self.mean_latitude = mean_latitude
    def forward(self, data):
        data = ensure_batch_dim(data)
        """
        data: torch.Tensor (B, H, W)
        if mean_latitude is True, then data should be (B, H, W//2+1)
        if mean_latitude is False, then data should be (B, W//2+1)
        """
        fft_data = torch.fft.rfft(data, dim = -1)
        power_spectrum = (fft_data.real ** 2 + fft_data.imag ** 2)
        if self.mean_latitude:
            power_spectrum = torch.mean(power_spectrum, dim = 1)
        return power_spectrum


########## Probabilistic metrics ##########

class Rank_Histogram(nn.Module):
    def __init__(self):
        super(Rank_Histogram, self).__init__()

    def forward(self, predictions, targets):
        """
        Args:
            predictions: (N, H, W)
            targets: (H, W)
            Return: histogram of ranks
        """
        rh = None
        N, H, W = predictions.shape
        coords_pred = {"member": range(N), "lat": range(H), "lon": range(W)}
        coords_target = {"lat": range(H), "lon": range(W)}

        pred_xr = xr.DataArray(predictions.detach().cpu().numpy(),
                                dims=["member", "lat", "lon"],
                                coords=coords_pred)
        target_xr = xr.DataArray(targets.detach().cpu().numpy(),
                                    dims=["lat", "lon"],
                                    coords=coords_target)

        rh = rank_histogram(target_xr, pred_xr).values
        return rh # (N_bins)

        
class CRPS(nn.Module):
    def __init__(self):
        super(CRPS, self).__init__()

    def forward(self, predictions, targets):
        """
        Args:
            predictions: (N, H, W)
            targets: (H, W)
        """

        N, H, W = predictions.shape
        coords_pred = {"member": range(N), "lat": range(H), "lon": range(W)}
        coords_target = {"lat": range(H), "lon": range(W)}

        pred_xr = xr.DataArray(predictions.detach().cpu().numpy(), 
                                dims=["member", "lat", "lon"], 
                                coords=coords_pred)
        target_xr = xr.DataArray(targets.detach().cpu().numpy(), 
                                    dims=["lat", "lon"], 
                                    coords=coords_target)

        crps = crps_ensemble(target_xr, pred_xr).item()
        return crps


class Spread_Skill_Ratio(nn.Module):
    def __init__(self):
        super(Spread_Skill_Ratio, self).__init__()
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: (N, H, W)
            targets: (H, W)
            A well-calibrated ensemble forecast should have a spread-skill ratio of 1.
        """
        ensemble_mean = predictions.mean(dim=0)
        mse = ((ensemble_mean - targets) ** 2)
        rmse = torch.sqrt(torch.mean(mse, dim = (-1, -2))) # (1,)
        spread = torch.std(predictions, dim=0, unbiased=True) # (H, W)
        spread_mean = torch.mean(spread) # (1,)
        ssr = spread_mean / rmse
        return ssr