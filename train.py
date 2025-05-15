import os
import yaml
import argparse
from pathlib import Path

import torch
torch.set_float32_matmul_precision('medium')
# torch.set_float32_matmul_precision('high') # for graphcast, use 'high'. !!
import lightning.pytorch as pl 
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
pl.seed_everything(42)

def main(args):
    """
    Training script given .yaml config
    Example usage:
        1) python train.py --item prediction --config seasonbench/prediction/fno_config.yaml
        2) python train.py --item correction --config seasonbench/correction/fno_config.yaml
        3) python train.py --item hindcast_pred --config seasonbench/hindcast/vit_config.yaml
        4) python train.py --item hindcast_corr --config seasonbench/hindcast/graphcast_config.yaml
    """

    with open(args.config, 'r') as config_filepath:
        hyperparams = yaml.load(config_filepath, Loader=yaml.FullLoader)

    model_args = hyperparams['model_args']
    data_args = hyperparams['data_args']

    if args.item == 'prediction':
        log_dir = Path(f'logs_{args.item}') / data_args['dataset_name'] / model_args['model_name']
    
    elif args.item == 'hindcast_pred':
        log_dir = Path(f'logs_{args.item}') / data_args['dataset_name'] / model_args['model_name']

    elif args.item == 'correction':
        log_dir = Path(f'logs_{args.item}') / data_args['center'] / model_args['model_name']
    
    elif args.item == 'hindcast_corr':
        log_dir = Path(f'logs_{args.item}') / data_args['center'] / model_args['model_name']

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=log_dir)
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', mode='min')

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=model_args['patience'],
        verbose=True,
        mode='min'
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    if args.item == 'prediction' or args.item == 'hindcast_pred':
        trainer = pl.Trainer(
            devices = -1,
            accelerator = 'gpu',
            strategy = 'ddp',
            max_epochs = model_args['max_epochs'],
            logger = tb_logger,
            callbacks = [checkpoint_callback, early_stop_callback, lr_monitor],
        )
    elif args.item == 'correction' or args.item == 'hindcast_corr':
        trainer = pl.Trainer(
            devices = -1,
            accelerator = 'gpu',
            strategy = 'ddp',
            max_epochs = model_args['max_epochs'],
            logger = tb_logger,
            callbacks = [checkpoint_callback, early_stop_callback, lr_monitor],
            use_distributed_sampler=False,
            # gradient_clip_val = model_args['gradient_clip_val'], # for unet.
        )


    if args.item == 'prediction' or args.item == 'hindcast_pred':
        from seasonbench.prediction.model import SeasonalPred
        baseline = SeasonalPred(model_args = model_args, data_args = data_args)
        baseline.setup()
    
    elif args.item == 'correction' or args.item == 'hindcast_corr':
        from seasonbench.correction.datamodule import NWPDataModule
        from seasonbench.correction.model import NWPCorrection
        data_module = NWPDataModule(data_args)
        baseline = NWPCorrection(model_args = model_args, data_args = data_args)
    else:
        raise ValueError(f"Invalid item {args.item}, choose from ['prediction', 'correction', 'hindcast_pred']")

    # print("Model Architecture: \n", baseline)

    if args.item == 'prediction' or args.item == 'hindcast_pred':
        trainer.fit(baseline)
    elif args.item == 'correction' or args.item == 'hindcast_corr':
        trainer.fit(baseline, data_module)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--item', type=str, required=True, help='prediction or correction or hindcast_pred')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    main(args)