import os
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
from seasonbench.prediction import dataset, model

N_STEPS = 6
BATCH_SIZE = 24

def plot_prediction(args):
    """
    Plot prediction and ground truth for a given model and given variables on the test dataset
    Example usage:
        python plot_prediction.py --device 1 --dataset_name china_025deg --model_name fno --version 1 --vars t2m t_850 z_500 q_700 tp
    """
    if args.dataset_name == 'china_025deg':
        H, W = 209, 421
    elif args.dataset_name == 'global_1deg':
        H, W = 181, 360
    elif args.dataset_name == 'global_025deg':
        H, W = 721, 1440
    else:
        raise ValueError(f"Dataset {args.dataset_name} not recognized. Choose from ['china_025deg', 'global_1deg', 'global_025deg']")

    log_dir = log_dir = Path(__file__).resolve().parent.parent / 'logs_prediction' / args.dataset_name / args.model_name
    version_dir = log_dir / 'lightning_logs' / f'version_{args.version}'
    config_filepath = version_dir / 'hparams.yaml'
    save_path = version_dir / 'prediction_plots'
    os.makedirs(save_path, exist_ok=True)

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    with open(config_filepath, 'r') as config_file:
        hyperparams = yaml.load(config_file, Loader=yaml.FullLoader)
    model_args = hyperparams['model_args']
    data_args = hyperparams['data_args']

    
    ckpt_path = version_dir / 'checkpoints'
    ckpt_file = os.listdir(ckpt_path)[0]
    baseline = model.SeasonalPred.load_from_checkpoint(ckpt_path / ckpt_file, map_location = device)
    baseline.eval()

    test_dataset = dataset.ERA5_Dataset(
        data_dir = data_args['data_dir'],
        input_vars = model_args['input_vars'],
        input_cons = model_args['input_cons'],
        output_vars = model_args['output_vars'],
        status = 'test',
        pred_step = N_STEPS,
        lead_step = data_args['lead_step'],
        crop_size = data_args['crop_size'],
        is_normalized = False
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size = BATCH_SIZE,
        shuffle = False,
        num_workers = model_args['num_workers'],
    )

    norm_mean_np = np.load(Path(data_args['data_dir']) / 'norm_mean.npz')
    norm_std_np = np.load(Path(data_args['data_dir']) / 'norm_std.npz')
    norm_mean = {
        var: torch.from_numpy(norm_mean_np[var]).to(device) for var in model_args['output_vars']
    }
    norm_std = {
        var: torch.from_numpy(norm_std_np[var]).to(device) for var in model_args['output_vars']
    }

    # load lat and lon for plotting + if center crop. lat dim : H, lon dim : W
    lat, lon = np.load(Path(data_args['data_dir']) / 'lat_lon.npz')['lat'], np.load(Path(data_args['data_dir']) / 'lat_lon.npz')['lon']
    if data_args['crop_size'] != 0:
        top = (H - data_args['crop_size'][0]) // 2
        left = (W - data_args['crop_size'][1]) // 2
        lat = lat[top:top + data_args['crop_size'][0]]
        lon = lon[left:left + data_args['crop_size'][1]]

    with torch.no_grad():
        for batch in tqdm(test_dataloader, total=len(test_dataloader)):
            x, cons, y_true, timestamp = batch
            timestamp = list(zip(*timestamp))
            x = x.to(device)
            cons = cons.to(device)
            y_true = y_true.to(device)
            y_true = y_true.permute(0, 2, 1, 3, 4)
            batch_size = y_true.shape[0]

            preds = []
            current_x = x
            for step in range(N_STEPS):
                model_input = torch.cat([current_x, cons], dim=1)
                if args.model_name == 'vae':
                    pred, _, _ = baseline(model_input)
                else:
                    pred = baseline(model_input)
                preds.append(pred)
                current_x = pred
            
            preds = torch.stack(preds, dim=1)
            denorm_preds = []
            for i, var in enumerate(model_args['output_vars']):
                denorm_preds.append(preds[:,:, i] * norm_std[var] + norm_mean[var])
            denorm_preds = torch.stack(denorm_preds, dim=2) 

            for var in args.vars:
                if var == 'tp': cmap = 'Blues'
                elif var in ['t2m', 't_1000','t_850','t_700','t_500','t_200']: cmap = 'coolwarm'
                else: cmap = 'viridis'

                if var not in model_args['output_vars']:
                    print(f"Variable {var} not in output variables")
                    continue

                var_idx = model_args['output_vars'].index(var)

                for b_idx in range(batch_size):
                    fig, axes = plt.subplots(2, N_STEPS, figsize=(3*N_STEPS, 6), subplot_kw={'projection': ccrs.PlateCarree()})
                    for step in range(N_STEPS):
                        pred_map = denorm_preds[b_idx, step, var_idx].cpu().numpy()
                        true_map = y_true[b_idx, step, var_idx].cpu().numpy()
                        vmin = min(pred_map.min(), true_map.min())
                        vmax = max(pred_map.max(), true_map.max())
                        ax_pred = axes[0, step]
                        ax_true = axes[1, step]
                        for ax, data, title in zip([ax_pred, ax_true], [pred_map, true_map], ['Prediction', 'Ground Truth']):
                            ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], ccrs.PlateCarree())
                            lon2d, lat2d = np.meshgrid(lon, lat)
                            im = ax.pcolormesh(lon2d, lat2d, data, cmap=cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
                            cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, aspect=50, extend='both')
                            ax.coastlines()
                            ax.add_feature(cfeature.BORDERS, linestyle=':')
                            ax.set_title(f"{title} - Step {step+1}", fontsize=10)

                    time = timestamp[b_idx][0]
                    month, year = int(time[5:7]), int(time[:4])
                    year = year - 1 if month == 1 else year
                    month = 12 if month == 1 else month - 1
                    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                    plt.suptitle(f'{var} - {year}-{month:02d}', fontsize=12)
                    plt.savefig(save_path / f'{var}_{year}-{month:02d}.png', dpi=300)
                    print(f'{var}_{year}-{month:02d}.png saved')
                    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot predictions for selected variables for a given model and dataset')
    parser.add_argument('--device', type=int, default=0, help='GPU device number')
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset (china_025deg, global_1deg, global_025deg)')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model (fno, vit)')
    parser.add_argument('--version', type=int, required=True, help='Version of the model')
    parser.add_argument('--vars', nargs='+', required=True, help='List of variables to plot')

    args = parser.parse_args()
    plot_prediction(args)