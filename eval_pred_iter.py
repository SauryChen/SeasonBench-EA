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

def save_metrics(args, final_metrics, N_STEPS):
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
    log_dir = Path('logs_prediction') / args.dataset_name / args.model_name
    if args.model_name in ['climatology', 'persistence']:
        metrics_dir = Path(log_dir) / 'metrics'
    else:
        metrics_dir = log_dir / f'lightning_logs/version_{args.version_num}'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    save_dict = {}
    for var in final_metrics['rmse'].keys():
        save_dict[f'rmse_{var}'] = np.array(final_metrics['rmse'][var], dtype=np.float32)
        save_dict[f'bias_{var}'] = np.array(final_metrics['bias'][var], dtype=np.float32)
        save_dict[f'acc_{var}'] = np.array(final_metrics['acc'][var], dtype=np.float32)
        save_dict[f'es_{var}'] = np.array(final_metrics['es'][var], dtype=np.float32)
        if 'es_gt' in final_metrics: # only for climatology
            save_dict[f'es_gt_{var}'] = np.array(final_metrics['es_gt'][var], dtype=np.float32)
        if var == 'tp':
            save_dict[f'csi_{var}'] = np.array(final_metrics['csi'][var], dtype=np.float32)
    save_dict['pred_step'] = np.arange(1, N_STEPS + 1, dtype=np.int32)
    save_path = metrics_dir / f'{args.model_name}_metrics.npz'
    np.savez(save_path, **save_dict)

def main(args):
    """
    Evaluate script given .yaml config and trained model checkpoint iteratively
    Example usage:
        (Climatology)   0) `python eval_pred_iter.py --dataset_name china_025deg --model_name climatology`
        (Persistence)   1) `python eval_pred_iter.py --dataset_name china_025deg --model_name persistence`
        (NWP Ensemble)  2)  omit in this script, use eval_nwp.py instead.
        (AI models)     3) `python eval_pred_iter.py --dataset_name china_025deg --model_name fno --version_num 0` version_num is the version in logs
    """
    pl.seed_everything(42)
    print(f'Evaluating {args.model_name} model iteratively')

    IS_CLIMATOLOGY, IS_PERSISTENCE, IS_AI_MODEL = False, False, False

    log_dir = Path('logs_prediction') / args.dataset_name / args.model_name
    device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')

    # Case 0: Climatology
    if args.model_name == 'climatology':
        IS_CLIMATOLOGY = True
        
        # hyperparameters 
        config_filepath = Path(f'seasonbench/prediction/climatology_config.yaml')
        with open(config_filepath, 'r') as config_file:
            hyperparams = yaml.load(config_file, Loader=yaml.FullLoader)
        data_args = hyperparams['data_args']
        model_args = hyperparams['model_args']
        
        # load data
        climatology_path = Path(data_args['data_dir']) / 'climatology.npz'
        climatology_data = np.load(climatology_path)
        climatology_dict = {}
        for var in data_args['used_vars']:
            assert var in climatology_data, f"Variable {var} not found in climatology data"
            var_data = climatology_data[var]
            assert var_data.shape[0] == 12, f"Variable {var} should have 12 months of data, but got {var_data.shape[0]} months"
            climatology_dict[var] = var_data
        
        test_dataset = dataset.ERA5_Dataset(
            data_dir = data_args['data_dir'],
            input_vars = data_args['used_vars'], # Needed otherwise error in dataset
            input_cons = data_args['input_cons'], # Needed otherwise error in dataset
            output_vars = data_args['used_vars'],
            status = 'test',
            pred_step = N_STEPS,
            lead_step = data_args['lead_step'],
            crop_size = data_args['crop_size'],
            is_normalized= False
        )
        test_dataloader = DataLoader(
            test_dataset,
            num_workers = data_args['num_workers'],
            batch_size = BATCH_SIZE,
            shuffle = False
        )

    # Case 1: Persistence
    elif args.model_name == 'persistence':
        IS_PERSISTENCE = True
        
        # hyperparameters
        config_filepath = Path(f'seasonbench/prediction/persistence_config.yaml')
        with open(config_filepath, 'r') as config_file:
            hyperparams = yaml.load(config_file, Loader=yaml.FullLoader)
        data_args = hyperparams['data_args']
        model_args = hyperparams['model_args']

        # load data
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
        test_dataloader = DataLoader(
            test_dataset,
            num_workers = model_args['num_workers'],
            batch_size = BATCH_SIZE,
            shuffle = False
        )

    # Case 2: NWP Ensemble
    elif args.model_name == 'nwp':
        raise NotImplementedError("NWP model is omitted in this script, use eval_nwp.py instead.")

    # Case 3: AI models
    elif args.model_name in ['fno', 'vit', 'vae', 'unet']:
        IS_AI_MODEL = True

        # hyperparameters
        config_filepath = Path(f'logs_prediction/{args.dataset_name}/{args.model_name}/lightning_logs/version_{args.version}/hparams.yaml')
        with open(config_filepath, 'r') as config_file:
            hyperparams = yaml.load(config_file, Loader=yaml.FullLoader)
        model_args = hyperparams['model_args']
        data_args = hyperparams['data_args']

        # load checkpoint
        ckpt_path = log_dir / f'lightning_logs/version_{args.version_num}/checkpoints'
        ckpt_file = list(ckpt_path.glob('*.ckpt'))[0]
        baseline = model.SeasonalPred.load_from_checkpoint(ckpt_file, map_location = device)
        baseline.eval()

        # load data
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
        test_dataloader = DataLoader(
            test_dataset,
            num_workers = model_args['num_workers'],
            batch_size = BATCH_SIZE,
            shuffle = False
        )
    else: 
        raise ValueError(f"Invalid model name {args.model_name}, choose from ['climatology', 'persistence', 'nwp names', 'ai model names']")

    # load mean and std for denormalization
    norm_mean_np = np.load(Path(data_args['data_dir']) / 'norm_mean.npz')
    norm_std_np = np.load(Path(data_args['data_dir']) / 'norm_std.npz')
    norm_mean = {
        var: torch.from_numpy(norm_mean_np[var]).to(device) for var in model_args['output_vars']
    }
    norm_std = {
        var: torch.from_numpy(norm_std_np[var]).to(device) for var in model_args['output_vars']
    }


    # load climatology
    climatology_path = Path(data_args['data_dir']) / 'climatology.npz'
    climatology_data = np.load(climatology_path)
    # load threshold
    tp_threshold_path = Path(data_args['data_dir']) / 'tp_quantiles.npz'
    tp_threshold = np.load(tp_threshold_path)
    tp_threshold = np.stack([tp_threshold['q50'], tp_threshold['q75'], tp_threshold['q90'], tp_threshold['q95'], tp_threshold['q99']], axis=0) # [5, H, W]

    # initialize criterion
    RMSE = criterion.RMSE()
    Bias = criterion.Bias()
    ES = criterion.Energy_Spectral()
    ACC = criterion.ACC(climatology=climatology_data, is_weight=False, crop_size=data_args['crop_size'])
    CSI = criterion.CSI(thresholds=tp_threshold, crop_size=data_args['crop_size'])

    # Evaluation
    metrics = {
        'rmse': {var:[[] for _ in range(N_STEPS)] for var in model_args['output_vars']},
        'bias': {var:[[] for _ in range(N_STEPS)] for var in model_args['output_vars']},
        'acc': {var:[[] for _ in range(N_STEPS)] for var in model_args['output_vars']},
        'es': {var:[[] for _ in range(N_STEPS)] for var in model_args['output_vars']},
        'csi': {var:[[] for _ in range(N_STEPS)] for var in ['tp']}, # only for tp
        'count': [[] for _ in range(N_STEPS)] # count the number of samples in each batch
    }

    if IS_CLIMATOLOGY:
        metrics['es_gt'] = {var:[[] for _ in range(N_STEPS)] for var in model_args['output_vars']}
        clim_tensor_dict = {
            var: torch.from_numpy(climatology_dict[var]).to(device) for var in model_args['output_vars'] # [12, H, W]
        }
        with torch.no_grad():
            for batch in tqdm(test_dataloader, total=len(test_dataloader)):
                _, _, y_true, timestamp = batch # timestamp format: 'YYYY-MM'
                timestamp = list(zip(*timestamp))
                y_true = y_true.permute(0, 2, 1, 3, 4).to(device) # [B, T, C, H, W]
                batch_size = y_true.shape[0]

                for b in range(y_true.shape[0]):
                    for step in range(y_true.size(1)):

                        metrics['count'][step].append(1) # count the number of samples for each step

                        clim_month = int(timestamp[b][step][5:7]) - 1
                        for i, var in enumerate(model_args['output_vars']):
                            clim_value = clim_tensor_dict[var][clim_month] #[H, W]
                            target = y_true[b, step, i] # [H, W]
                            rmse = RMSE(clim_value, target).cpu().numpy()
                            bias = Bias(clim_value, target).cpu().numpy()
                            acc = ACC(clim_value, target, [timestamp[b][step]], var).cpu().numpy()
                            es = ES(clim_value).cpu().numpy()
                            es_gt = ES(target).cpu().numpy()
                            if var == 'tp':
                                csi = CSI(clim_value, target).cpu().numpy()
                            metrics['rmse'][var][step].append(rmse)
                            metrics['bias'][var][step].append(bias)
                            metrics['acc'][var][step].append(acc)
                            metrics['es'][var][step].append(es)
                            metrics['es_gt'][var][step].append(es_gt)
                            if var == 'tp':
                                metrics['csi'][var][step].append(csi)
        final_metrics = {
            'rmse': {
                var: [
                    np.sum(np.array(step_vals) * np.array(metrics['count'][step])) / np.sum(metrics['count'][step])
                    for step, step_vals in enumerate(step_lists)
                ]
                for var, step_lists in metrics['rmse'].items()
            },
            'bias': {
                var: [
                    np.sum(np.array(step_vals) * np.array(metrics['count'][step])) / np.sum(metrics['count'][step])
                    for step, step_vals in enumerate(step_lists)
                ]
                for var, step_lists in metrics['bias'].items()
            },
            'acc': {
                var: [
                    np.sum(np.array(step_vals) * np.array(metrics['count'][step])) / np.sum(metrics['count'][step])
                    for step, step_vals in enumerate(step_lists)
                ]
                for var, step_lists in metrics['acc'].items()
            },
            'es': {
                var: [
                    np.mean(np.concatenate(step_vals, axis=0), axis=0) # concatenate along batch, and average along batch.
                    for step_vals in step_lists
                ]
                for var, step_lists in metrics['es'].items()
            },
            'es_gt': {
                var: [
                    np.mean(np.concatenate(step_vals, axis=0), axis=0)
                    for step_vals in step_lists
                ]
                for var, step_lists in metrics['es_gt'].items()
            },
            'csi': {
                var: [
                    np.sum(np.stack(step_vals, axis=0) * np.array(metrics['count'][step])[:, None], axis=0) / np.sum(metrics['count'][step])
                    for step_vals in step_lists
                ]
                for var, step_lists in metrics['csi'].items()
            } # return a list containing 6 sublists, with each sublist containing 5 values.
        }
        save_metrics(args, final_metrics, N_STEPS)

    else:
        with torch.no_grad():
            for batch in tqdm(test_dataloader, total=len(test_dataloader)):
                x, cons, y_true, timestamp = batch # NOTE, y is the original data without normalization, with the shape of [B, C, T, H, W]
                timestamp = list(zip(*timestamp)) # [B, T]
                x = x.to(device) # [B, C, H, W]
                cons = cons.to(device)
                y_true = y_true.to(device)
                y_true = y_true.permute(0, 2, 1, 3, 4)
                batch_size = y_true.shape[0]

                original_x, original_cons = x.clone(), cons.clone()
                preds = None

                assert y_true.size(1) == N_STEPS, f"y.size(1) should be {N_STEPS}, but got {y.size(1)}"
                for step in range(y_true.size(1)):

                    metrics['count'][step].append(batch_size)

                    if IS_AI_MODEL:
                        current_x = original_x if step == 0 else preds.detach()
                        model_input = torch.cat([current_x, original_cons], dim=1)
                        if args.model_name == 'vae':
                            preds, _, _ = baseline(model_input)
                        else:
                            preds = baseline(model_input)
                    elif IS_PERSISTENCE:
                        preds = original_x
                    else:
                        raise NotImplementedError

                    denorm_preds = []
                    for i ,var in enumerate(model_args['output_vars']):
                        denorm_preds.append(preds[:, i,] * norm_std[var] + norm_mean[var])
                    denorm_preds = torch.stack(denorm_preds, dim=1)

                    target = y_true[:, step,]
                    for i, var in enumerate(model_args['output_vars']):
                        rmse = RMSE(denorm_preds[:, i], target[:, i]).cpu().numpy()
                        bias = Bias(denorm_preds[:, i], target[:, i]).cpu().numpy()
                        timestamps_for_this_step = [ts[step] for ts in timestamp]
                        acc = ACC(denorm_preds[:, i], target[:, i], timestamps_for_this_step, var).cpu().numpy()
                        es = ES(denorm_preds[:, i]).cpu().numpy()
                        if var == 'tp':
                            csi = CSI(denorm_preds[:, i], target[:, i]).cpu().numpy()

                        metrics['rmse'][var][step].extend([rmse])
                        metrics['bias'][var][step].extend([bias])
                        metrics['acc'][var][step].extend([acc])
                        metrics['es'][var][step].extend([es])
                        if var == 'tp':
                            metrics['csi'][var][step].extend([csi])

                        if var in ['tp']:
                            print(f"Step {step+1}/{N_STEPS}, Variable: {var}, CSI: {csi[0]:.2f} / {csi[-1]:.2f}, RMSE: {rmse:.2f}, ACC: {acc:.2f}")

        final_metrics = {
            'rmse': {
                var: [
                    np.sum(np.array(step_vals) * np.array(metrics['count'][step])) / np.sum(metrics['count'][step])
                    for step, step_vals in enumerate(step_lists)
                ]
                for var, step_lists in metrics['rmse'].items()
            },
            'bias': {
                var: [
                    np.sum(np.array(step_vals) * np.array(metrics['count'][step])) / np.sum(metrics['count'][step])
                    for step, step_vals in enumerate(step_lists)
                ]
                for var, step_lists in metrics['bias'].items()
            },
            'acc': {
                var: [
                    np.sum(np.array(step_vals) * np.array(metrics['count'][step])) / np.sum(metrics['count'][step])
                    for step, step_vals in enumerate(step_lists)
                ]
                for var, step_lists in metrics['acc'].items()
            },
            'es': {
                var: [
                    np.mean(np.concatenate(step_vals, axis=0), axis=0)
                    for step_vals in step_lists
                ]
                for var, step_lists in metrics['es'].items()
            },
            'csi': {
                var: [
                    np.sum(np.stack(step_vals, axis=0) * np.array(metrics['count'][step])[:, None], axis=0) / np.sum(metrics['count'][step])
                    for step, step_vals in enumerate(step_lists)
                ]
                for var, step_lists in metrics['csi'].items()
            } # return a list containing T sublists, with each sublist containing 5 values.
        }
        if IS_AI_MODEL or IS_PERSISTENCE:
            save_metrics(args, final_metrics, N_STEPS)
        else:
            pass # TODO

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate script.")
    parser.add_argument("--dataset_name", type=str, default="china_025deg", help="Dataset name")
    parser.add_argument("--model_name", type=str, default="fno", help="Model name")
    parser.add_argument("--version_num", type=int, default=0, help="Version number")
    args = parser.parse_args()
    main(args)
