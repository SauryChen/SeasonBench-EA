"""
    To evaluate the hindcast of NWP and its correction. Consider start month of march.
    1). Inference on the "test set" to get the prediction results and ground truth for total precipitation in [June, July, and August].
    2). Save the predicted results and ground truth in the "test set" for the three months.
"""
import os
import copy
from collections import defaultdict
import datetime
import argparse
import yaml
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from tqdm import tqdm
import torch
from collections import defaultdict
from torch.utils.data import DataLoader, TensorDataset
import lightning.pytorch as pl
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import ModelCheckpoint

import sys
sys.path.append('..')
import warnings
warnings.filterwarnings('ignore')

from seasonbench import criterion
from seasonbench.correction import dataset, model

N_STEPS = 6
ensemble_count = 10

def stack_and_average(d: dict) -> np.ndarray:
    for k in d.keys():
        print(k, len(d[k]), d[k][0].shape)
    return np.stack([np.mean(d[k], axis=0) for k in sorted(d.keys())], axis=0)

def main(args):
    """
    Example usage:
        python save_hindcast_corr.py --center cmcc --model_name graphcast --version_num 0 # for eval the correction
    
    Note, only support batch size = 1 for now.
    """
    pl.seed_everything(42)
    print(f'Evaluating {args.model_name} model')
    log_dir = Path('logs_hindcast_corr/') / args.center / args.model_name
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # =============== Set config and load model =============== #
    config_filepath = Path(f'logs_hindcast_corr/{args.center}/{args.model_name}/lightning_logs/version_{args.version_num}/hparams.yaml')
    with open(config_filepath, 'r') as config_file:
        hyperparams = yaml.load(config_file, Loader=yaml.FullLoader)
    model_args = hyperparams['model_args']
    data_args = hyperparams['data_args']

    ckpt_path = log_dir / f'lightning_logs/version_{args.version_num}/checkpoints'
    ckpt_file = list(ckpt_path.glob('*.ckpt'))[0]
    if args.model_name == 'graphcast':
        baseline = model.NWPCorrection.load_from_checkpoint(ckpt_file)
    else:
        baseline = model.NWPCorrection.load_from_checkpoint(ckpt_file, map_location = device)
    baseline.eval()

    # ============== Load data =============== #
    norm_mean_np = np.load(Path(data_args['data_dir']) / 'ERA5_monthly_mean/global_1deg/processed_data' / 'norm_mean.npz')
    norm_std_np = np.load(Path(data_args['data_dir']) / 'ERA5_monthly_mean/global_1deg/processed_data' / 'norm_std.npz')
    print(norm_mean_np['tp'], norm_std_np['tp'])

    test_dataset_graphcast = dataset.NWP_Dataset(
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

    graphcast_loader = DataLoader(
        test_dataset_graphcast,
        batch_size = 1,
        shuffle = False,
        num_workers = data_args['num_workers'],
    )

    # =============== Inference the test =============== # 
    print('Inference the testing dataset')
    tp_idx = test_dataset_graphcast.output_vars.index('tp')
    print("tp_idx:", tp_idx)
    print("baseline.used_var_indices:", baseline.used_var_indices)

    tp_test_june = defaultdict(list)
    tp_test_july = defaultdict(list)
    tp_test_august = defaultdict(list)
    tp_test_gt_june = defaultdict(list)
    tp_test_gt_july = defaultdict(list)
    tp_test_gt_august = defaultdict(list)

    with torch.no_grad():
        for batch in tqdm(graphcast_loader, total = len(graphcast_loader)):
            x_nwp, x_cons, y_true, timestamp = batch
            x_nwp_ = copy.deepcopy(x_nwp)
            x_nwp_ = x_nwp_.to(device)

            B, C, T, H, W = x_nwp.shape
            assert B == 1, "Only support batch size = 1"
            C_ = x_cons.shape[1]
            x_nwp, x_cons, y_true = x_nwp.to(device), x_cons.to(device), y_true.to(device)
            timestamp = list(zip(*timestamp)) # [B, T]

            ref_time = timestamp[0][0] # forecasting start month, e.g. "2013-03"
            start_month = ref_time.split('-')[1]

            if start_month != '03':
                continue
            
            x_nwp = x_nwp.reshape(B, -1, H, W)
            x_cons = x_cons.expand(B, C_, T, H, W).reshape(B, -1, H, W)
            x = torch.cat([x_nwp, x_cons], dim=1)
            preds = baseline(x)

            x_nwp_ori = x_nwp_[:, baseline.used_var_indices, ...].reshape(B, -1, H, W)
            preds = preds + x_nwp_ori
            preds = preds.reshape(B, -1, T, H, W)

            for i in range(B):
                pred = preds[i, tp_idx, ...].cpu().numpy() # [T, H, W]
                pred = pred * norm_std_np['tp'] + norm_mean_np['tp']
                true = y_true[i, tp_idx, ...].cpu().numpy() # [T, H, W]
                tp_test_june[ref_time].append(pred[3])
                tp_test_gt_june[ref_time].append(true[3])
                tp_test_july[ref_time].append(pred[4])
                tp_test_gt_july[ref_time].append(true[4])
                tp_test_august[ref_time].append(pred[5])
                tp_test_gt_august[ref_time].append(true[5])
                

    tp_test_june = stack_and_average(tp_test_june)
    tp_test_july = stack_and_average(tp_test_july)
    tp_test_august = stack_and_average(tp_test_august)

    tp_test_gt_june = stack_and_average(tp_test_gt_june)
    tp_test_gt_july = stack_and_average(tp_test_gt_july)
    tp_test_gt_august = stack_and_average(tp_test_gt_august)
    print(tp_test_june.shape, tp_test_gt_june.shape, tp_test_july.shape, tp_test_gt_july.shape, tp_test_august.shape, tp_test_gt_august.shape)

    predict = {
        'test_june': tp_test_june,
        'test_july': tp_test_july,
        'test_august': tp_test_august,
    }
    truth = {
        'gt_june': tp_test_gt_june,
        'gt_july': tp_test_gt_july,
        'gt_august': tp_test_gt_august,
    }
    np.savez(log_dir / f'lightning_logs/version_{args.version_num}' / 'predict.npz', **predict)
    np.savez(log_dir / f'lightning_logs/version_{args.version_num}' / 'truth.npz', **truth)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hindcast correction')
    parser.add_argument('--center', type=str, default='cmcc', help='Center name')
    parser.add_argument('--model_name', type=str, default='vit', help='Model name')
    parser.add_argument('--version_num', type=int, default=0, help='Version number')
    args = parser.parse_args()

    main(args)