"""
    evaluate all the ensemble members for a given nwp model (center name) with ERA5 1 deg data.
    All the evaluation are made within the EA region lat = [60, 8], lon = [58, 163]
    Ensemble Mean for RMSE, bias, acc, es and csi. Independent members for 
"""
import os
import yaml
import copy
import torch
import numpy as np
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from torch.utils.data import DataLoader

import argparse
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

from seasonbench import criterion
from seasonbench.correction import dataset, model

N_STEPS = 6

def save_metrics(args, final_metrics, log_dir):
    """
    Args:
        args:
        final_metrics:
            {
                'rmse': {'z_500': [step1, step2, ...], 'tp': [...]},
                'bias': {'z_500': [step1, step2, ...], 'tp': [...]},
                'acc': {'z_500': [step1, step2, ...], 'tp': [...]},
                'es': {'z_500': [step1, step2, ...], 'tp': [...]},
                'csi': {'tp': [step1, step2, ...]}
            }
        N_STEPS
    """
    save_dict = {}
    for var in final_metrics['rmse'].keys():
        save_dict[f'rmse_{var}'] = np.array(final_metrics['rmse'][var], dtype=np.float32)
        save_dict[f'bias_{var}'] = np.array(final_metrics['bias'][var], dtype=np.float32)
        save_dict[f'wi_{var}'] = np.array(final_metrics['wi'][var], dtype=np.float32)
        save_dict[f'acc_{var}'] = np.array(final_metrics['acc'][var], dtype=np.float32)
        save_dict[f'es_{var}'] = np.array(final_metrics['es'][var], dtype=np.float32)
        save_dict[f'es_gt_{var}'] = np.array(final_metrics['es_gt'][var], dtype=np.float32)
        if var == 'tp':
            save_dict[f'csi_{var}'] = np.array(final_metrics['csi'][var], dtype=np.float32)
        
        save_dict[f'rank_hist_{var}'] = np.stack(final_metrics['rank_hist'][var], axis=0) # [N_STEPS, N_bins]
        save_dict[f'crps_{var}'] = np.array(final_metrics['crps'][var], dtype=np.float32)
        save_dict[f'ssr_{var}'] = np.array(final_metrics['ssr'][var], dtype=np.float32)
        
        save_dict['pred_step'] = np.arange(1, N_STEPS + 1, dtype=np.int32)
        save_path = log_dir / f'{args.center}_metrics.npz'
        np.savez(save_path, **save_dict)


def main(args):
    """
    Example usage: 
            1). python eval_nwp.py --item prediction --center cmcc (eval the original NWP performance)
            2). python eval_nwp.py --item correction --center cmcc --model_name fno --version 0 (eval the NWP performance (10 ens) after correction
            3). python eval_nwp.py --item climatology --center cmcc (eval the climatology performance)
    if the metric is deterministic, the ensemble mean is used for evaluation.
    if the metric is probabilistic, all the ensemble members are used for evaluation.
    """
    H_, W_ = 181, 360 # global_1deg
    pl.seed_everything(42)
    IS_PRED, IS_CORR, IS_CLIM = False, False, False

    if args.item == 'prediction':
        if args.center == 'cmcc': ensemble_count = 40
        elif args.center == 'dwd': ensemble_count = 30
        elif args.center == 'eccc': ensemble_count = 10
        elif args.center == 'ecmwf': ensemble_count = 25
        elif args.center == 'meteo_france': ensemble_count = 25
        log_dir = Path('logs_correction') / args.center / 'metrics'
        os.makedirs(log_dir, exist_ok=True)
        config_filepath = Path('seasonbench') / 'prediction' / 'nwp_config.yaml'
        device = torch.device('cpu')
    
    elif args.item == 'climatology':
        ensemble_count = 10
        log_dir = Path('logs_correction') / 'climatology' / 'metrics'
        os.makedirs(log_dir, exist_ok=True)
        config_filepath = Path('seasonbench') / 'correction' / 'climatology_config.yaml'
        device = torch.device('cpu')


    elif args.item == 'correction':
        ensemble_count = 10
        log_dir = Path('logs_correction') / args.center / args.model_name / 'lightning_logs' / f'version_{args.version}'
        config_filepath = log_dir / 'hparams.yaml'
        device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')

    else:
        raise ValueError('item should be either prediction or correction')


    with open(config_filepath, 'r') as config_file:
        hyperparams = yaml.load(config_file, Loader=yaml.FullLoader)
    model_args = hyperparams['model_args']
    data_args = hyperparams['data_args']

    # load model 
    if args.item == 'correction':
        ckpt_path = log_dir / 'checkpoints'
        ckpt_file = list(ckpt_path.glob('*.ckpt'))[0]
        print("Loading checkpoint from:", ckpt_file)
        if args.model_name == 'graphcast':
            baseline = model.NWPCorrection.load_from_checkpoint(ckpt_file)
        else:
            baseline = model.NWPCorrection.load_from_checkpoint(ckpt_file, map_location = device)
        baseline.eval()

    print("Device:", device)

    ####################################### Set Data. #######################################
    
    if args.item == 'prediction':
        IS_PRED = True
        test_dataset = dataset.NWP_Dataset(
            data_dir = data_args['data_dir'],
            center = args.center,
            input_vars = data_args['input_vars'],
            input_cons = data_args['input_cons'],
            output_vars = data_args['output_vars'],
            status = 'test',
            crop_size = data_args['crop_size'],
            is_normalized_nwp = False,
            is_normalized_era5 = False,
            ens_count = ensemble_count,
            nwp_subfolder = 'test_ens',
        )

        test_dataloader = DataLoader(
            test_dataset,
            batch_size = data_args['batch_size'],
            shuffle = False,
            num_workers = data_args['num_workers'],
        )
    
    elif args.item == 'climatology':
        IS_CLIM = True
        test_dataset = dataset.NWP_Dataset(
            data_dir = data_args['data_dir'],
            center = args.center,
            input_vars = data_args['input_vars'],
            input_cons = data_args['input_cons'],
            output_vars = data_args['output_vars'],
            status = 'test',
            crop_size = data_args['crop_size'],
            is_normalized_nwp = False,
            is_normalized_era5 = False,
            ens_count = ensemble_count,
            nwp_subfolder = 'test_ens',
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size = data_args['batch_size'],
            shuffle = False,
            num_workers = data_args['num_workers'],
        )

    
    elif args.item == 'correction':
        IS_CORR = True
        test_dataset = dataset.NWP_Dataset(
            data_dir = data_args['data_dir'],
            center = args.center,
            input_vars = data_args['input_vars'],
            input_cons = data_args['input_cons'],
            output_vars = data_args['output_vars'],
            status = 'test',
            crop_size = data_args['crop_size'],
            is_normalized_nwp = True, # for inference
            is_normalized_era5 = False,
            ens_count = ensemble_count,
            nwp_subfolder = 'processed_data_10ens',
        )

        test_dataloader = DataLoader(
            test_dataset,
            batch_size = data_args['batch_size'],
            shuffle = False,
            num_workers = data_args['num_workers'],
        )


    ####################################### Load Mean, Std, Climatology, Set Metrics. #######################################


    # load mean, std and climatology
    norm_mean_np = np.load(Path(data_args['data_dir']) / 'ERA5_monthly_mean/global_1deg/processed_data' / 'norm_mean.npz')
    norm_std_np = np.load(Path(data_args['data_dir']) / 'ERA5_monthly_mean/global_1deg/processed_data' / 'norm_std.npz')
    norm_mean = {
        var: torch.from_numpy(norm_mean_np[var]).to(device) for var in data_args['output_vars']
    }
    norm_std = {
        var: torch.from_numpy(norm_std_np[var]).to(device) for var in data_args['output_vars']
    }

    # first crop, then select the region.
    lat = np.load(Path(data_args['data_dir']) / 'ERA5_monthly_mean/global_1deg/processed_data' / 'lat_lon.npz', allow_pickle=True)['lat']
    lon = np.load(Path(data_args['data_dir']) / 'ERA5_monthly_mean/global_1deg/processed_data' / 'lat_lon.npz', allow_pickle=True)['lon']
    lon = ((lon + 180) % 360) - 180
    lon = np.sort(lon)
    if data_args['crop_size'][0] != 181:
        print(data_args['crop_size'])
        top = (H_ - data_args['crop_size'][0]) // 2
        left = (W_ - data_args['crop_size'][1]) // 2
        lat = lat[top:top + data_args['crop_size'][0]]
        lon = lon[left:left + data_args['crop_size'][1]]
    lat_idx = np.where((lat <= 60) & (lat >= 8))[0]
    lon_idx = np.where((lon >= 58) & (lon <= 163))[0]
    lat = lat[lat_idx]
    lon = lon[lon_idx]


    climatology_path = Path(data_args['data_dir']) / 'ERA5_monthly_mean/global_1deg/processed_data' / 'climatology.npz'
    climatology_data = np.load(climatology_path)
    tp_threshold_path = Path(data_args['data_dir']) / 'ERA5_monthly_mean/global_1deg/processed_data' / 'tp_quantiles.npz'
    tp_threshold = np.load(tp_threshold_path)
    tp_threshold = np.stack([tp_threshold['q50'], tp_threshold['q75'], tp_threshold['q90'], tp_threshold['q95'], tp_threshold['q99']], axis=0) # [5, H, W]

    # inital criterion
    RMSE = criterion.RMSE()
    Bias = criterion.Bias()
    WI = criterion.Willmott_Index()
    ES = criterion.Energy_Spectral()
    ACC = criterion.ACC(climatology=climatology_data, is_weight=False, crop_size=data_args['crop_size'], lat_idx=lat_idx, lon_idx=lon_idx)
    CSI = criterion.CSI(thresholds=tp_threshold, crop_size=data_args['crop_size'], lat_idx=lat_idx, lon_idx=lon_idx)
    Rank_Hist = criterion.Rank_Histogram()
    CRPS = criterion.CRPS()
    SSR = criterion.Spread_Skill_Ratio()
    
    # Evaluation
    metrics = {
        'rmse': {var:[[] for _ in range(N_STEPS)] for var in data_args['output_vars']},
        'bias': {var:[[] for _ in range(N_STEPS)] for var in data_args['output_vars']},
        'wi': {var:[[] for _ in range(N_STEPS)] for var in data_args['output_vars']},
        'acc': {var:[[] for _ in range(N_STEPS)] for var in data_args['output_vars']},
        'es': {var:[[] for _ in range(N_STEPS)] for var in data_args['output_vars']},
        'es_gt': {var:[[] for _ in range(N_STEPS)] for var in data_args['output_vars']},
        'csi': {var:[[] for _ in range(N_STEPS)] for var in ['tp']}, # only for tp
        'rank_hist': {var:[[] for _ in range(N_STEPS)] for var in data_args['output_vars']},
        'crps': {var:[[] for _ in range(N_STEPS)] for var in data_args['output_vars']},
        'ssr': {var:[[] for _ in range(N_STEPS)] for var in data_args['output_vars']},
    }

    input_var_flat = data_args['input_vars']['pressure_levels'] + data_args['input_vars']['single_level'] + data_args['input_cons']
    input_var_flat = ['tp' if var == 'tprate' else var for var in input_var_flat]
    var_to_idx = {var: i for i, var in enumerate(input_var_flat)}


####################################### Eval Below. #######################################

    all_true = [[] for _ in range(N_STEPS)]
    grouped = defaultdict(lambda: {'pred':[], 'true': []})

    if IS_PRED:
        with torch.no_grad():
            for batch in tqdm(test_dataloader, total=len(test_dataloader), desc='Inference'):
                x_nwp, x_cons, y_true, timestamp = batch
                x_nwp, y_true = x_nwp.to(device), y_true.to(device)
                timestamp = list(zip(*timestamp)) # [B,T]
                B = x_nwp.shape[0]

                for i in range(B):
                    ref_time = timestamp[i][0]
                    grouped[ref_time]['pred'].append(x_nwp[i].detach().cpu())
                    grouped[ref_time]['true'].append(y_true[i].detach().cpu())

                for t in range(N_STEPS):
                    all_true[t].append(y_true[:, :, t, :, :].detach().cpu()) # [ENS, C_out, H, W]

    if IS_CLIM:
        output_var = data_args['output_vars']
        with torch.no_grad():
            for batch in tqdm(test_dataloader, total=len(test_dataloader), desc='Inference'):
                _, _, y_true, timestamp = batch # y_true shape [B, C_out, T, H, W]
                timestamp = list(zip(*timestamp)) # [B,T]
                B, T = y_true.shape[0], y_true.shape[2]
                for i in range(B):
                    climatology_list = []
                    for ti in range(T):
                        clim_list_per_time = []
                        month_idx = int(timestamp[i][ti].split('-')[1]) - 1
                        for var in output_var:
                            clim_var = climatology_data[var][month_idx] # [H, W]
                            if data_args['crop_size'][0] != 181:
                                clim_var = clim_var[top:top + data_args['crop_size'][0], left:left + data_args['crop_size'][1]]
                            clim_list_per_time.append(clim_var)
                        clim_list_per_time = np.stack(clim_list_per_time, axis=0) # [C_out, H, W]
                        climatology_list.append(clim_list_per_time)
                    climatology_list = np.stack(climatology_list, axis=0) # [T, C_out, H, W]
                    ref_time = timestamp[i][0]
                    grouped[ref_time]['pred'].append(torch.tensor(climatology_list).permute(1, 0, 2, 3))
                    grouped[ref_time]['true'].append(y_true[i].detach().cpu())
                for t in range(N_STEPS):
                    all_true[t].append(y_true[:, :, t, :, :].detach().cpu())


    elif IS_CORR:
        with torch.no_grad():
            for batch in tqdm(test_dataloader, total=len(test_dataloader), desc='Inference'):
                x_nwp, x_cons, y_true, timestamp = batch
                x_nwp_ = copy.deepcopy(x_nwp) # used in learning residual # [B,C,T,H,W]
                x_nwp_ = x_nwp_.to(device)

                B, C, T, H, W = x_nwp.shape
                assert T == N_STEPS, f'The number of time steps should be {N_STEPS}'
                C_ = x_cons.shape[1]
                x_nwp, x_cons, y_true = x_nwp.to(device), x_cons.to(device), y_true.to(device)
                timestamp = list(zip(*timestamp))
                x_nwp = x_nwp.reshape(B, -1, H, W)
                x_cons = x_cons.expand(B, C_, T, H, W).reshape(B, -1, H, W)
                x = torch.cat([x_nwp, x_cons], dim=1)
                if args.model_name == 'vae':
                    preds, _, _ = baseline(x)
                else:
                    preds = baseline(x)

                # learning residual
                x_nwp_ori = x_nwp_[:, baseline.used_var_indices, ...].reshape(x_nwp_.shape[0], -1, x_nwp_.shape[-2], x_nwp_.shape[-1])
                preds = preds + x_nwp_ori

                preds = preds.reshape(B, len(data_args['output_vars']),T, H, W)
                # denorm
                for i, var in enumerate(data_args['output_vars']):
                    preds[:, i, :, :, :] = preds[:, i, :, :, :] * norm_std[var] + norm_mean[var]

                for i in range(B):
                    ref_time = timestamp[i][0]
                    grouped[ref_time]['pred'].append(preds[i].detach().cpu())
                    grouped[ref_time]['true'].append(y_true[i].detach().cpu())

                for t in range(N_STEPS):
                    all_true[t].append(y_true[:, :, t, :, :].detach().cpu())

    # used in willmott index as obs_mean
    all_true_mean = [torch.mean(torch.concatenate(all_true[t], axis=0), axis=0) for t in range(N_STEPS)]

    for ref_time, group in grouped.items():
        pred = torch.stack(group['pred'], axis=0)[:,:,:,lat_idx][:,:,:,:,lon_idx]
        true = torch.stack(group['true'], axis=0)[:,:,:,lat_idx][:,:,:,:,lon_idx]
        print(ref_time, pred.shape, true.shape)

        for kk in range(1,ensemble_count):
            assert torch.allclose(true[0], true[kk]), 'y_true should be the same for all ensemble members' # a simple assertion

        pred_mean = torch.mean(pred, axis=0)
        true_mean = torch.mean(true, axis=0)


        for var_idx, var in enumerate(data_args['output_vars']):
            if var not in var_to_idx:
                print(f'{var} not in input variables')
                continue

            if IS_PRED:
                input_idx = var_to_idx[var]
                pred_var = pred[:, input_idx, :, :, :]
                pred_mean_var = pred_mean[input_idx, :, :, :]
            elif IS_CORR:
                pred_var = pred[:, var_idx, :, :, :] 
                pred_mean_var = pred_mean[var_idx, :, :, :] 
            elif IS_CLIM:
                pred_var = pred[:, var_idx, :, :, :]
                pred_mean_var = pred_mean[var_idx, :, :, :] 
                
            true_var = true[:, var_idx, :, :, :] 
            true_mean_var = true_mean[var_idx, :, :, :] 

            for t in range(N_STEPS):
                year_, month_ = map(int, ref_time.split('-'))
                step_month = month_ + t
                step_year = year_ + (step_month - 1) // 12
                step_month = (step_month - 1) % 12 + 1
                valid_time = f"{step_year:04d}-{step_month:02d}"

                p = pred_var[:, t, :, :]
                gt = true_var[:, t, :, :]
                p_mean = pred_mean_var[t, :, :].squeeze()
                gt_mean = true_mean_var[t, :, :].squeeze()

                
                metrics['rmse'][var][t].append(RMSE(p_mean, gt_mean))
                metrics['bias'][var][t].append(Bias(p_mean, gt_mean))
                all_true_mean_t = all_true_mean[t][var_idx, :, :].squeeze()
                metrics['wi'][var][t].append(WI(p_mean, gt_mean, all_true_mean_t[lat_idx,:][:, lon_idx]))

                metrics['acc'][var][t].append(ACC(p_mean, gt_mean, [valid_time], var))
                metrics['es'][var][t].append(ES(p_mean))
                metrics['es_gt'][var][t].append(ES(gt_mean))
                if var == 'tp':
                    metrics['csi'][var][t].append(CSI(p_mean, gt_mean))
                
                metrics['rank_hist'][var][t].append(Rank_Hist(p, gt_mean))
                metrics['crps'][var][t].append(CRPS(p, gt_mean))
                metrics['ssr'][var][t].append(SSR(p, gt_mean))
    
    final_metrics = {
        'rmse': {var: [np.mean(metrics['rmse'][var][t]) for t in range(N_STEPS)] for var in data_args['output_vars']},
        'bias': {var: [np.mean(metrics['bias'][var][t]) for t in range(N_STEPS)] for var in data_args['output_vars']},
        'wi': {var: [np.mean(metrics['wi'][var][t]) for t in range(N_STEPS)] for var in data_args['output_vars']},
        'acc': {var: [np.mean(metrics['acc'][var][t]) for t in range(N_STEPS)] for var in data_args['output_vars']},
        'es': {var: [np.mean(np.concatenate(metrics['es'][var][t], axis=0), axis=0) for t in range(N_STEPS)] for var in data_args['output_vars']},
        'es_gt': {var: [np.mean(np.concatenate(metrics['es_gt'][var][t], axis=0), axis=0) for t in range(N_STEPS)] for var in data_args['output_vars']},
        'csi':{var: [np.mean(np.stack(metrics['csi'][var][t], axis=0), axis=0) for t in range(N_STEPS)] for var in ['tp']},
        'rank_hist': {var: [np.sum(metrics['rank_hist'][var][t], axis=0) for t in range(N_STEPS)] for var in data_args['output_vars']},
        'crps': {var: [np.mean(metrics['crps'][var][t]) for t in range(N_STEPS)] for var in data_args['output_vars']},
        'ssr': {var: [np.mean(metrics['ssr'][var][t]) for t in range(N_STEPS)] for var in data_args['output_vars']},
    }

    # save metrics
    save_metrics(args, final_metrics, log_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate NWP model')
    parser.add_argument('--item', type=str, choices=['prediction', 'correction', 'climatology'], required=True, help='item to evaluate')
    parser.add_argument('--center', type=str, required=True, help='NWP center name (cmcc, dwd, eccc, ecmwf, meteo_france)')
    parser.add_argument('--model_name', type=str, default='fno', help='model name (default: fno)')
    parser.add_argument('--version', type=int, default=0, help='version number (default: 0)')
    args = parser.parse_args()
    main(args)
