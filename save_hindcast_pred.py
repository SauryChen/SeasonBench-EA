"""
    To evaluate the hindcast of prediction model. Consider start month of february, which predicts 4-6 months ahead.
    1). Inference on the "test set" to get the prediction results and ground truth for total precipitation in [June, July, and August].
    2). Save the predicted results and ground truth in the "test set" for the three months.
"""
import os
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

from seasonbench.prediction import dataset, model
from seasonbench import criterion
N_STEPS = 6
BATCH_SIZE = 24

def main(args):
    """
    Example usage:
        python save_hindcast_pred.py --dataset_name china_025deg_2006-2020 --model_name vit --version_num 0
    """
    pl.seed_everything(42)
    print(f'Evaluating {args.model_name} model iteratively')
    log_dir = Path('logs_hindcast_pred/') / args.dataset_name / args.model_name
    device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')

    # =============== Set config and load model =============== #
    config_filepath = Path(f'logs_hindcast_pred/{args.dataset_name}/{args.model_name}/lightning_logs/version_{args.version_num}/hparams.yaml')
    with open(config_filepath, 'r') as config_file:
        hyperparams = yaml.load(config_file, Loader=yaml.FullLoader)
    model_args = hyperparams['model_args']
    data_args = hyperparams['data_args']

    ckpt_path = log_dir / f'lightning_logs/version_{args.version_num}/checkpoints'
    ckpt_file = list(ckpt_path.glob('*.ckpt'))[0]
    baseline = model.SeasonalPred.load_from_checkpoint(ckpt_file, map_location=device)
    baseline.eval()

    # ============== Load data =============== #
    norm_mean_np = np.load(Path(data_args['data_dir']) / 'norm_mean.npz')
    norm_std_np = np.load(Path(data_args['data_dir']) / 'norm_std.npz')

    test_dataset = dataset.ERA5_Dataset(
        data_dir = data_args['data_dir'],
        input_vars = model_args['input_vars'],
        input_cons = model_args['input_cons'],
        output_vars = model_args['output_vars'],
        status = 'test',
        pred_step = N_STEPS,
        lead_step = data_args['lead_step'],
        crop_size = data_args['crop_size'],
        is_normalized= False
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # =============== Inference the test =============== # 
    print('Inference the testing dataset')
    tp_idx = test_dataset.output_vars.index('tp')
    tp_test_june, tp_test_july, tp_test_august = [], [], []
    tp_test_gt_june, tp_test_gt_july, tp_test_gt_august = [], [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, total = len(test_loader)):
            x, cons, y_true, timestamp = batch
            timestamp = list(zip(*timestamp))
            x = x.to(device)
            cons = cons.to(device)
            batch_size = len(timestamp)

            original_x, original_cons = x.clone(), cons.clone()
            preds = None

            for step_idx in range(N_STEPS):
                current_x = original_x if step_idx == 0 else preds
                model_input = torch.cat([current_x, original_cons], dim=1)
                preds = baseline.model(model_input)
                preds_tp = preds[:, tp_idx]
                for b in range(batch_size):
                    start_month = int(timestamp[b][0].split('-')[1]) - 1
                    if start_month == 2:
                        if step_idx == 3:
                            tp_test_june.append(preds_tp[b].cpu().numpy() * norm_std_np['tp'] + norm_mean_np['tp'])
                            tp_test_gt_june.append(y_true[b, tp_idx, step_idx])
                        elif step_idx == 4:
                            tp_test_july.append(preds_tp[b].cpu().numpy() * norm_std_np['tp'] + norm_mean_np['tp'])
                            tp_test_gt_july.append(y_true[b, tp_idx, step_idx])
                        elif step_idx == 5:
                            tp_test_august.append(preds_tp[b].cpu().numpy() * norm_std_np['tp'] + norm_mean_np['tp'])
                            tp_test_gt_august.append(y_true[b, tp_idx, step_idx])
                        else: 
                            continue
                    else:
                        continue
    tp_test_june = np.stack(tp_test_june, axis=0)
    tp_test_july = np.stack(tp_test_july, axis=0)
    tp_test_august = np.stack(tp_test_august, axis=0)
    tp_test_gt_june = np.stack(tp_test_gt_june, axis=0)
    tp_test_gt_july = np.stack(tp_test_gt_july, axis=0)
    tp_test_gt_august = np.stack(tp_test_gt_august, axis=0)
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
    parser = argparse.ArgumentParser(description='Hindcast prediction')
    parser.add_argument('--dataset_name', type=str, default='china_025-1996_2000', help='Dataset name')
    parser.add_argument('--model_name', type=str, default='vit', help='Model name')
    parser.add_argument('--version_num', type=int, default=0, help='Version number')
    args = parser.parse_args()

    main(args)