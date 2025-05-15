import os 
import torch
import numpy as np
import pandas as pd 
from typing import List, Dict
from torch.utils.data import Dataset

class ERA5_Dataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        input_vars: List[str],
        input_cons: List[str],
        output_vars: List[str],
        status: str,
        lead_step: int = 1,
        pred_step: int = 1,
        crop_size: tuple = (200, 400),
        is_normalized: bool = True, # whether to normalize the output variables
    ) -> None:
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"{data_dir} doesn't exist.")

        self.data_dir = data_dir
        self.input_vars = input_vars
        self.input_cons = input_cons
        self.output_vars = output_vars
        self.status = status
        self.lead_step = lead_step
        self.pred_step = pred_step
        self.crop_h, self.crop_w = crop_size
        self.is_normalized = is_normalized

        # load data and normalization parameters.
        self.normalize_mean: Dict[str, np.ndarray] = dict(
            np.load(os.path.join(data_dir, 'norm_mean.npz'))
        )
        self.normalize_std: Dict[str, np.ndarray] = dict(
            np.load(os.path.join(data_dir, 'norm_std.npz'))
        )
        self.data: Dict[str, np.ndarray] = self._load_data()

        time_steps = self.data[self.input_vars[0]].shape[0]
        self.indices = list(range(time_steps - self.pred_step - self.lead_step))

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        t = self.indices[idx]
        target_t = [t + self.lead_step+i for i in range(self.pred_step)]

        x = []
        for var in self.input_vars:
            arr = self.data[var][t]
            arr = (arr - self.normalize_mean[var]) / self.normalize_std[var]
            arr = self.center_crop(arr)
            x.append(arr)
        
        c = []
        for const in self.input_cons:
            arr = self.data[const]
            if arr.ndim == 3:
                arr = arr[0]
            arr = (arr - self.normalize_mean[const]) / self.normalize_std[const]
            arr = self.center_crop(arr)
            c.append(arr)

        y = []
        for var in self.output_vars:
            arr = np.stack([self.data[var][ts] for ts in target_t], axis=0) # [T, H, W]
            if self.is_normalized:
                arr = (arr - self.normalize_mean[var]) / self.normalize_std[var]
            arr = self.center_crop(arr)
            y.append(arr)

        valid_time = [
            pd.to_datetime(self.data['valid_time'][ts]).strftime("%Y-%m")
            for ts in target_t
        ]

        return (
            torch.FloatTensor(np.stack(x, axis=0)),
            torch.FloatTensor(np.stack(c, axis=0)),
            torch.FloatTensor(np.stack(y, axis=0)),
            list(valid_time),
        )
    
    def _load_data(self) -> Dict[str, np.ndarray]:
        data_path = os.path.join(self.data_dir, f'{self.status}.npz')
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"{data_path} doesn't exist.")
        with np.load(data_path) as data:
            data_dict = {k: data[k] for k in data}
        return data_dict
    
    def center_crop(self, arr: np.ndarray) -> np.ndarray:
        h, w = arr.shape[-2], arr.shape[-1]
        top = (h - self.crop_h) // 2
        left = (w - self.crop_w) // 2
        return arr[..., top:top + self.crop_h, left:left + self.crop_w]


if __name__ == "__main__":
    dataset = ERA5_Dataset(
        data_dir = '/root/Weather_Data/flood_season_data/ERA5_monthly_mean/china_025deg/processed_data',
        input_vars = ['msl', 't2m', 'z_500', 't_850'],
        input_cons = ['lsm', 'z'], 
        output_vars = ['z_500', 'tp'],
        status = 'test',
        lead_step = 1,
        pred_step = 2,
        crop_size = (200, 400),
    )
    x, cons, y, time = dataset[2]
    print(x.shape, cons.shape, y.shape)
    print(time)
    for i in range(len(x)):
        print(f"x[{i}]: {np.mean(x[i].numpy())}, {np.std(x[i].numpy())}")
    for i in range(len(cons)):
        print(f"cons[{i}]: {np.mean(cons[i].numpy())}, {np.std(cons[i].numpy())}")
    for i in range(len(y)):
        for j in range(len(y[i])):
            print(f"y[{i}][{j}]: {np.mean(y[i][j].numpy())}, {np.std(y[i][j].numpy())}")