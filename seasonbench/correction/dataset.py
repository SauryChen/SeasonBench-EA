"""
    The NWP file is saved in YYYY-M.npz or YYYY-MM.npz format, with some years missing in the training dataset. Just ignore the missing years.
    For each NWP file, it has the dim of [ensemble member=10, forecast_reference_time, forecastMonth, lat, lon].
    To increase the training samples, we view each of the ensemble member as a different sample.
    For each NWP sample [forecast_reference_time, forecastMonth=6, lat, lon], load the corresponding 6 time-step ERA5 data from train/val/test.npz.
    The mean and std for normalization is in the same filepath of ERA5 npz data, named as norm_mean.npz and norm_std.npz.
"""
import os
import torch
import numpy as np
import xarray as xr
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict
import torch.distributed as dist
from torch.utils.data import Dataset

START_YEAR, END_YEAR = 1993, 2024
VAL_YEARS = [2009, 2010, 2011]
# VAL_YEARS  = [2021, 2022, 2023, 2024] # for hindcast
TEST_YEARS = [2013, 2014, 2015, 2016]
# TEST_YEARS = [2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016] # for hindcast


class NWP_Dataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        center: str, # [cmcc, dwd, eccc, ecmwf, meteo_france]
        input_vars: Dict[str, List[str]], # {"pressure_levels": [...], "single_level": [...]}
        input_cons: List[str],
        output_vars: List[str],
        status: str, # [train, val, test]
        crop_size: tuple = (180, 360),
        is_normalized_nwp: bool = True, # whether to normalize the input variables
        is_normalized_era5: bool = True, # whether to normalize the output variables
        ens_count: int = 10,
        nwp_subfolder: str='processed_data_10ens' # ['processed_data_10ens', 'test_ens']
    ) -> None:
        super().__init__()

        self._init_distributed()

        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"{data_dir} doesn't exist.")

        self.data_dir = Path(data_dir)
        self.center = center
        self.input_vars = input_vars
        self.input_cons = input_cons
        self.output_vars = output_vars
        self.status = status
        self.crop_size = crop_size
        self.is_normalized_nwp = is_normalized_nwp
        self.is_normalized_era5 = is_normalized_era5
        self.ens_count = ens_count
        self.nwp_subfolder = nwp_subfolder

        self._init_era5_data()
        self._init_nwp_data()
        self._build_sample_index()

    def _init_distributed(self):
        self.rank = 0
        self.world_size = 1
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        
    def _init_era5_data(self):
        era5_path = self.data_dir / 'ERA5_monthly_mean/global_1deg/processed_data' / f'{self.status}.npz'
        # era5_path = self.data_dir / 'ERA5_monthly_mean/global_1deg/hindcast/2006-2016' / f'{self.status}.npz' # for hindcast
        with np.load(era5_path, allow_pickle=True) as era5_npz:
            self.era5_data = {key: era5_npz[key].copy() for key in era5_npz.files}
            self.era5_data['valid_time'] = era5_npz['valid_time']
    
        self.timestamps_era5 = np.array([str(t)[:7] for t in self.era5_data['valid_time']])
        self.sort_idx = np.argsort(self.timestamps_era5)
        self.timestamps_sorted = self.timestamps_era5[self.sort_idx]

    
        self.normalized_mean: Dict[str, np.ndarray] = dict(
            np.load(self.data_dir / 'ERA5_monthly_mean/global_1deg/processed_data' / 'norm_mean.npz')
            # np.load(self.data_dir / 'ERA5_monthly_mean/global_1deg/hindcast/2006-2016' / 'norm_mean.npz') # for hindcast
        )
        self.normalized_std: Dict[str, np.ndarray] = dict(
            np.load(self.data_dir / 'ERA5_monthly_mean/global_1deg/processed_data' / 'norm_std.npz')
            # np.load(self.data_dir / 'ERA5_monthly_mean/global_1deg/hindcast/2006-2016' / 'norm_std.npz') # for hindcast
        )

        # load lat and lon
        self.lat = np.load(self.data_dir / 'ERA5_monthly_mean/global_1deg/processed_data' / 'lat_lon.npz', allow_pickle=True)['lat']
        # self.lat = np.load(self.data_dir / 'ERA5_monthly_mean/global_1deg/hindcast/2006-2016' / 'lat_lon.npz', allow_pickle=True)['lat'] # for hindcast
        self.lon = np.load(self.data_dir / 'ERA5_monthly_mean/global_1deg/processed_data' / 'lat_lon.npz', allow_pickle=True)['lon']
        # self.lon = np.load(self.data_dir / 'ERA5_monthly_mean/global_1deg/hindcast/2006-2016' / 'lat_lon.npz', allow_pickle=True)['lon'] # for hindcast
        self.lon = ((self.lon + 180) % 360) - 180 # [-180, 179]
        self.lon = np.sort(self.lon)
    
    def _init_nwp_data(self):
        # NWP file list
        self.nwp_dir = Path(self.data_dir) / 'NWP_ensemble_monthly_mean' / self.center / self.nwp_subfolder
        self.file_keys = self._collect_files()

        residual = len(self.file_keys) % self.world_size
        if residual != 0:
            self.file_keys += self.file_keys[:self.world_size - residual]
        if len(self.file_keys) % self.world_size != 0:
            raise ValueError(f"Number of files {len(self.file_keys)} is not divisible by world size {self.world_size}.")
        file_splits = np.array_split(self.file_keys, self.world_size)

        self.file_keys = file_splits[self.rank].tolist()

        self.nwp_cache = {}
        for file_key in tqdm(self.file_keys, desc="Loading NWP data", disable=self.rank != 0):
            self.nwp_cache[file_key] = self._load_nwp_file(file_key)


    def _build_sample_index(self):
        self.sample_index = [
            (file_key, ens_idx)
            for file_key in self.file_keys
            for ens_idx in range(self.ens_count)
        ]

    def __len__(self):
        return len(self.sample_index)

    def __getitem__(self, idx):
        file_key, ens_idx = self.sample_index[idx]
        nwp_data = self.nwp_cache[file_key] # [Ensemble, Var, Time, H, W]
        x_nwp = nwp_data[ens_idx] # [Var, Time, H, W]
        y_target, x_const, time = self._choose_era5_data(file_key)
        return (
            torch.tensor(x_nwp, dtype=torch.float32),
            torch.tensor(x_const, dtype=torch.float32),
            torch.tensor(y_target, dtype=torch.float32),
            time
        )

    
    def _center_crop(self, data: np.ndarray) -> np.ndarray:
        h, w = data.shape[-2], data.shape[-1]
        assert h == 181 and w == 360, f"Invalid shape: {data.shape}. Expected (181, 360)."
        top = (h - self.crop_size[0]) // 2
        left = (w - self.crop_size[1]) // 2
        return data[..., top:top + self.crop_size[0], left:left + self.crop_size[1]]
    
    
    def _choose_era5_data(self, file_key: str):
        timestamp = file_key
        year, month = int(timestamp[:4]), int(timestamp[5:])
        steps = []
        for i in range(6):
            step_month = month + i
            step_year = year + (step_month - 1) // 12
            step_month  = (step_month - 1) % 12 + 1
            steps.append(f'{step_year:04d}-{step_month:02d}')
        
        out_vars = []
        search_idx = np.searchsorted(self.timestamps_sorted, steps)
        indices = self.sort_idx[search_idx]
        

        for var in self.output_vars:
            if var in self.era5_data.keys():
                var_data = self.era5_data[var]
            else:
                raise ValueError(f"Variable {var} not found in ERA5 data.")

            var_data_sel = var_data[indices]
            if self.is_normalized_era5:
                var_data_sel = (var_data_sel - self.normalized_mean[var]) / self.normalized_std[var]
            out_vars.append(var_data_sel)
        out_vars = np.stack(out_vars, axis=0)
        out_vars = self._center_crop(out_vars)

        cons_vars = []
        for const in self.input_cons:
            if const in self.era5_data.keys():
                const_data = self.era5_data[const]
            else:
                raise ValueError(f"Constant {const} not found in ERA5 data.")
            if self.is_normalized_nwp: # since the constant are used as the training data, rather than the output data.
                const_data = (const_data - self.normalized_mean[const]) / self.normalized_std[const]
            cons_vars.append(const_data)
        cons_vars = np.stack(cons_vars, axis=0) # [Const, H, W]
        cons_vars = self._center_crop(cons_vars)

        return out_vars, cons_vars, steps


    def _collect_files(self):
        pres_dir = self.nwp_dir / 'pressure_levels'
        all_files = sorted(pres_dir.glob('*.npz'))

        def get_year(fname):
            stem = fname.stem
            try:
                return int(stem.split('_')[0]) # Year
            except ValueError:
                raise ValueError(f"Invalid filename format: {fname}. Expected format YYYY-MM or YYYY-M.")
        
        if self.status == 'train':
            valid_years = list(range(START_YEAR, END_YEAR))
            valid_years = [year for year in valid_years if year not in VAL_YEARS and year not in TEST_YEARS]
        elif self.status == 'val':
            valid_years = VAL_YEARS
        elif self.status == 'test':
            valid_years = TEST_YEARS
        else:
            raise ValueError(f"Invalid status: {self.status}. Must be one of ['train', 'val', 'test'].")

        # delete the last year month 8-12 because of the ERA5 boundary.
        file_list = []
        for f in all_files:
            year, month = int(f.stem.split('_')[0]), int(f.stem.split('_')[1])
            if year not in valid_years:
                continue
            if year == max(valid_years) and month in [8, 9, 10, 11, 12]:
                continue
            file_list.append(f.stem)

        return file_list

    def _load_nwp_file(self, file_key: str):
        pressure_path  = self.nwp_dir / 'pressure_levels' / f'{file_key}.npz'
        single_path = self.nwp_dir / 'single_level' / f'{file_key}.npz'
        pressure_vars = self.input_vars['pressure_levels']
        single_vars = self.input_vars['single_level']

        data_list = []
        pressure_data = np.load(pressure_path, allow_pickle=True)
        for var in pressure_vars:
            if var in pressure_data.keys():
                var_data = pressure_data[var].squeeze() # [Ensemble, Time, H=181, W=360]
                if self.is_normalized_nwp:
                    var_data = (var_data - self.normalized_mean[var]) / self.normalized_std[var]
                data_list.append(var_data)
            else:
                raise ValueError(f"Variable {var} not found in NWP data.")
        
        single_data = np.load(single_path, allow_pickle=True)
        for var in single_vars:
            if var in single_data.keys():
                var_data = single_data[var].squeeze() # [Ensemble, Time, H=181, W=360]
                if self.is_normalized_nwp:
                    if var == 'tprate':
                        var_data = (var_data - self.normalized_mean['tp']) / self.normalized_std['tp']
                    else:
                        var_data = (var_data - self.normalized_mean[var]) / self.normalized_std[var]
                data_list.append(var_data)
            else:
                raise ValueError(f"Variable {var} not found in NWP data.")
        # stack the data
        data = np.stack(data_list, axis=1) # [Ensemble, Var, Time, H=181, W=360]
        data = self._center_crop(data) # [Ensemble, Var, Time, H=180, W=360]
        return data # [Ensemble, Var, Time, H=180, W=360]


if __name__ == "__main__":
    dataset = NWP_Dataset(
        data_dir = '/root/Weather_Data/flood_season_data',
        center = 'cmcc',
        input_vars = {
            'pressure_levels': ['z_500', 't_850', 'q_700'],
            'single_level': ['t2m', 'tprate', 'sst', 'msl'],
        },
        input_cons = ['lsm', 'z'],
        output_vars = ['z_500', 't_850', 'tp', 'q_700', 't2m'],
        status = 'train',
        crop_size = (180, 360),
        is_normalized_nwp = True,
        is_normalized_era5 = True,
    )

    print(len(dataset))
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
    )
    print(len(loader))
    exit()

    # check lon convert
    x_nwp, x_cons, y, time = dataset[10]
    x_nwp_plot = x_nwp.numpy()[3,0,:,:]
    y_plot = y.numpy()[4,0,:,:]
    # plot in a figure
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(x_nwp_plot, cmap='jet', origin='upper')
    ax[0].set_title('NWP data')
    ax[1].imshow(y_plot, cmap='jet', origin='upper')
    ax[1].set_title('ERA5 data')
    plt.savefig('check_lon_convert.png')


    # for i, (x_nwp, x_cons, y, time) in enumerate(loader):
    #     print(f"Batch {i}:")
    #     print(x_nwp.shape, x_cons.shape, y.shape, list(zip(*time)))
    #     if i > 10:
    #         break


    # for i in range(30):
    #     x_nwp, x_cons, y, time = dataset[i]
    #     print(x_nwp.shape, x_cons.shape, y.shape, time)
    
    # for i in range(len(x_nwp)):
    #     print(f"x[{i}]: {np.mean(x_nwp[i].numpy())}, {np.std(x_nwp[i].numpy())}")
    # for i in range(len(x_cons)):
    #     print(f"cons[{i}]: {np.mean(x_cons[i].numpy())}, {np.std(x_cons[i].numpy())}")
    # for i in range(len(y)):
    #     for j in range(len(y[i])):
    #         print(f"y[{i}][{j}]: {np.mean(y[i][j].numpy())}, {np.std(y[i][j].numpy())}")