import os
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from torch.utils.data import DataLoader

import sys
sys.path.append('..')
import yaml
import argparse
from tqdm import tqdm
from pathlib import Path
from seasonbench.correction import dataset, model


def plot_prediction(args):
    """
    Plot correction ensemble results (ensemble mean) and ground truth for a given model and given variables on the test dataset
    Example usage:
        python plot_correction.py --device 7 --center cmcc --model_name fno --version 0
    """

    H_, W_ = 181, 360 # global_1deg

    log_dir = log_dir = Path(__file__).resolve().parent.parent / 'logs_correction' / args.center / args.model_name
    version_dir = log_dir / 'lightning_logs' / f'version_{args.version}'
    config_filepath = version_dir / 'hparams.yaml'
    save_path = version_dir / 'correction_plots'
    os.makedirs(save_path, exist_ok=True)

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    with open(config_filepath, 'r') as config_file:
        hyperparams = yaml.load(config_file, Loader=yaml.FullLoader)
    model_args = hyperparams['model_args']
    data_args = hyperparams['data_args']

    
    ckpt_path = version_dir / 'checkpoints'
    ckpt_file = os.listdir(ckpt_path)[0]
    print("Loading checkpoint from:", ckpt_path, ckpt_file)
    if args.model_name == 'graphcast':
        baseline = model.NWPCorrection.load_from_checkpoint(ckpt_path / ckpt_file)
    else:
        baseline = model.NWPCorrection.load_from_checkpoint(ckpt_path / ckpt_file, map_location = device)
    baseline.eval()

    test_dataset = dataset.NWP_Dataset(
        data_dir = data_args['data_dir'],
        center = data_args['center'],
        input_vars = data_args['input_vars'],
        input_cons = data_args['input_cons'],
        output_vars = data_args['output_vars'],
        status = 'test',
        crop_size = data_args['crop_size'],
        is_normalized_nwp = True,
        is_normalized_era5 = False,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size = data_args['batch_size'],
        shuffle = False,
        num_workers = data_args['num_workers'],
    )

    norm_mean_np = np.load(Path(data_args['data_dir']) / 'ERA5_monthly_mean/global_1deg/processed_data' / 'norm_mean.npz')
    norm_std_np = np.load(Path(data_args['data_dir']) / 'ERA5_monthly_mean/global_1deg/processed_data' / 'norm_std.npz')
    norm_mean = {
        var: torch.from_numpy(norm_mean_np[var]).to(device) for var in data_args['output_vars']
    }
    norm_std = {
        var: torch.from_numpy(norm_std_np[var]).to(device) for var in data_args['output_vars']
    }

    # load lat and lon for plotting + if center crop. lat dim : H, lon dim : W
    lat = np.load(Path(data_args['data_dir']) / 'ERA5_monthly_mean/global_1deg/processed_data' / 'lat_lon.npz', allow_pickle=True)['lat']
    lon = np.load(Path(data_args['data_dir']) / 'ERA5_monthly_mean/global_1deg/processed_data' / 'lat_lon.npz', allow_pickle=True)['lon']
    lon = ((lon + 180) % 360) - 180 # convert to [-180, 179]
    lon = np.sort(lon)

    if data_args['crop_size'] != 0:
        top = (H_ - data_args['crop_size'][0]) // 2
        left = (W_ - data_args['crop_size'][1]) // 2
        lat = lat[top:top + data_args['crop_size'][0]]
        lon = lon[left:left + data_args['crop_size'][1]]

    grouped = {}
    input_var_flat = data_args['input_vars']['pressure_levels'] + data_args['input_vars']['single_level'] + data_args['input_cons']
    input_var_flat = ['tp' if var == 'tprate' else var for var in input_var_flat]
    var_to_idx = {var: i for i, var in enumerate(input_var_flat)}

    with torch.no_grad():
        for batch in tqdm(test_dataloader, total=len(test_dataloader), desc='Inference'):
            x_nwp, x_cons, y_true, time = batch
            x_nwp_ = copy.deepcopy(x_nwp)
            x_nwp_ = x_nwp_.to(device)

            B, C, T, H, W = x_nwp.shape
            C_ = x_cons.shape[1]
            x_nwp = x_nwp.reshape(B, -1, H, W).to(device)
            x_cons = x_cons.expand(B, C_, T, H, W).reshape(B, -1, H, W).to(device)
            time = list(zip(*time)) 
            x = torch.cat([x_nwp, x_cons], dim=1)
            
            y_pred = baseline(x)
            x_nwp_ori = x_nwp_[:, baseline.used_var_indices, ...].reshape(B, -1, H, W)
            y_pred = y_pred + x_nwp_ori
            y_pred = y_pred.reshape(B, len(data_args['output_vars']), T, H, W) 

            x_nwp_denorm = {}

            for var_idx, var in enumerate(data_args['output_vars']):
                y_pred[:, var_idx] = y_pred[:, var_idx] * norm_std[var] + norm_mean[var]
                if var in var_to_idx:
                    idx = var_to_idx[var]
                    x_nwp_var = x_nwp[:, idx*T:(idx+1)*T]
                    x_nwp_var = x_nwp_var * norm_std[var] + norm_mean[var]
                    x_nwp_denorm[var] = x_nwp_var.cpu().numpy()

            x_nwp_denorm = np.stack(list(x_nwp_denorm.values()), axis=1) 

            # group by forecast_reference_time (first element in time tuple)
            time_keys = [t[0] for t in time]
            for i, key in enumerate(time_keys):
                if key not in grouped:
                    grouped[key] = {'pred': [], 'true': [], 'nwp': []}
                grouped[key]['pred'].append(y_pred[i].cpu().numpy())
                grouped[key]['nwp'].append(x_nwp_denorm[i]) 
                grouped[key]['true'].append(y_true[i]) 
    
    # plot the EA region lat = [60, 8], lon = [58, 163]
    lat_idx = np.where((lat <= 60) & (lat >= 8))[0]
    lon_idx = np.where((lon <= 163) & (lon >= 58))[0]
    lat_crop = lat[lat_idx]
    lon_crop = lon[lon_idx]

    for ref_time, group in grouped.items():
        pred_stack = np.stack(group['pred'], axis=0) # [Ens, C, T, H, W]
        true_stack = np.stack(group['true'], axis=0) # [Ens, C, T, H, W]
        nwp_stack = np.stack(group['nwp'], axis=0) # [Ens, C, T, H, W]
        pred_mean = np.mean(pred_stack, axis=0) # [C, T, H, W]
        true_mean = np.mean(true_stack, axis=0) # [C, T, H, W]
        nwp_mean = np.mean(nwp_stack, axis=0) # [C, T, H, W]

        pred_mean = pred_mean[:, :, lat_idx][:, :, :, lon_idx]
        true_mean = true_mean[:, :, lat_idx][:, :, :, lon_idx]
        nwp_mean = nwp_mean[:, :, lat_idx][:, :, :, lon_idx]


        for var_idx, var in enumerate(data_args['output_vars']):
            if var == 'tp': cmap = 'Blues'
            elif var in ['t2m', 't_1000','t_850','t_700','t_500','t_200']: cmap = 'coolwarm'
            else: cmap = 'viridis'

            fig, axes = plt.subplots(3, T, figsize=(3*T, 9), subplot_kw={'projection': ccrs.PlateCarree()})

            vmin = min(pred_mean[var_idx].min(), true_mean[var_idx].min(), nwp_mean[var_idx].min())
            vmax = max(pred_mean[var_idx].max(), true_mean[var_idx].max(), nwp_mean[var_idx].max())

            for t in range(T):
                if var == 'tp': # do not use the global vmin and vmax. Set vmin and vmax for each time step
                    vmin = min(pred_mean[var_idx][t].min(), true_mean[var_idx][t].min(), nwp_mean[var_idx][t].min())
                    vmax = max(pred_mean[var_idx][t].max(), true_mean[var_idx][t].max(), nwp_mean[var_idx][t].max())

                pred_map = pred_mean[var_idx, t] # [H, W]
                true_map = true_mean[var_idx, t]
                nwp_map = nwp_mean[var_idx, t]
                ax_nwp = axes[0, t]
                ax_pred = axes[1, t]
                ax_true = axes[2, t]
                for ax, data, title in zip([ax_nwp, ax_pred, ax_true], [nwp_map, pred_map, true_map], ['ENS10', 'ENS10 Correction', 'Ground Truth']):
                    ax.set_extent([lon_crop.min(), lon_crop.max(), lat_crop.min(), lat_crop.max()], ccrs.PlateCarree())
                    lon2d, lat2d = np.meshgrid(lon_crop, lat_crop)
                    im = ax.pcolormesh(lon2d, lat2d, data, cmap=cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
                    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, aspect=50, extend='both')
                    ax.coastlines()
                    ax.add_feature(cfeature.BORDERS, linestyle=':')
                    ax.set_title(f"{title} - Step {t+1}", fontsize=10)
            
            current_time = ref_time
            plt.suptitle(f'{var} - {current_time}', fontsize=12)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(save_path / f'{var}_{current_time}.png', dpi=300)
            print(f'{var}_{current_time}.png saved')
            plt.close(fig)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Plot corrections for all output variables for a given model and dataset')
    parser.add_argument('--device', type=int, default=0, help='GPU device number')
    parser.add_argument('--center', type=str, required=True, help='Name of the center (cmcc)')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model (fno, vit)')
    parser.add_argument('--version', type=int, required=True, help='Version of the model')

    args = parser.parse_args()
    plot_prediction(args)