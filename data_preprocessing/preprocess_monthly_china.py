import os
import argparse
import numpy as np
import xarray as xr

TRAIN_START, TRAIN_END = 1940, 2015
VAL_START, VAL_END = 2016, 2019
TEST_START, TEST_END = 2020, 2024
CLIM_START, CLIM_END = 1991, 2020

VARIABLES = {
    "atm_surface": {
        "names": ['2m_temperature', 'mean_sea_level_pressure', 'total_precipitation'],
        "vars": ['t2m', 'msl', 'tp'],
        "unit_convert": {'tp': lambda x: x * 1000}  # m/day -> mm/day
    },
    "atm_pressure": {
        "names": ['temperature', 'geopotential', 'specific_humidity', 'u_component_of_wind', 'v_component_of_wind'],
        "vars": ['t', 'z', 'q', 'u', 'v'],
        "levels": [1000, 850, 700, 500, 200]
    },
    "boundary": {
        "names": ['soil_temperature_level_1', 'boundary_layer_height', 'snow_albedo', 'snow_depth', 'surface_solar_radiation_downwards'],
        "vars": ['stl1', 'blh', 'asn', 'sd', 'ssrd']
    },
    "constant": {
        "names": ['land_sea_mask', 'soil_type', 'geopotential_surface'],
        "vars": ['lsm', 'slt', 'z'],
        "unit_convert": {'z': lambda x: x / 9.80665}  # m²/s² -> geopotential m
    }
}

DATA_PATH = '/root/Weather_Data/flood_season_data/ERA5_monthly_mean/'

def make_dir(dir_name):
    os.makedirs(dir_name, exist_ok=True)

def safe_file_access(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} doesn't exist.")
    return xr.open_dataset(file_path)

def process_variable_group(ds_name, status, start_day, end_day):
    np_vars = {}
    
    group = VARIABLES["atm_surface"]
    for var_name, var in zip(group["names"], group["vars"]):
        file_path = os.path.join(DATA_PATH, ds_name, f'{var_name}.nc')
        with safe_file_access(file_path) as ds:
            data = ds[var].sel(valid_time=slice(start_day, end_day)).values
            if var in group["unit_convert"]:
                data = group["unit_convert"][var](data)
            np_vars[var] = data
            if 'valid_time' not in np_vars:
                np_vars['valid_time'] = ds.valid_time.sel(valid_time=slice(start_day, end_day)).values
            print(f"[Surface] {var}: {data.shape}")

    group = VARIABLES["atm_pressure"]
    for var_name, var in zip(group["names"], group["vars"]):
        file_path = os.path.join(DATA_PATH, ds_name, f'{var_name}.nc')
        with safe_file_access(file_path) as ds:
            for level in group["levels"]:
                data = ds[var].sel(
                    valid_time=slice(start_day, end_day),
                    pressure_level=level
                ).values
                np_vars[f'{var}_{level}'] = data
                print(f"[Pressure] {var}_{level}: {data.shape}")

    group = VARIABLES["boundary"]
    for var_name, var in zip(group["names"], group["vars"]):
        file_path = os.path.join(DATA_PATH, ds_name, f'{var_name}.nc')
        with safe_file_access(file_path) as ds:
            data = ds[var].sel(valid_time=slice(start_day, end_day)).values
            np_vars[var] = data
            print(f"[Boundary] {var}: {data.shape}")

    group = VARIABLES["constant"]
    for var_name, var in zip(group["names"], group["vars"]):
        file_path = os.path.join(DATA_PATH, ds_name, f'{var_name}.nc')
        with safe_file_access(file_path) as ds:
            data = ds[var].values
            if var in group['unit_convert']:
                data = group['unit_convert'][var](data)
            np_vars[var] = data
            print(f"[Constant] {var}: {data.shape}")

    return np_vars

def convert_to_numpy(status, dataset_name):
    time_ranges = {
        'train': (f"{TRAIN_START}-01-01", f"{TRAIN_END}-12-31"),
        'val': (f"{VAL_START-1}-12-01", f"{VAL_END}-12-31"),
        'test': (f"{TEST_START-1}-12-01", f"{TEST_END}-12-31")
    }
    
    start_day, end_day = time_ranges[status]
    save_dir = os.path.join(DATA_PATH, dataset_name, 'processed_data')
    make_dir(save_dir)
    
    print(f"\nProcessing {status} data: {start_day} to {end_day}")
    np_vars = process_variable_group(dataset_name, status, start_day, end_day)
    
    np.savez(
        os.path.join(save_dir, f'{status}.npz'),
        **np_vars
    )
    print(f"Saved {status} data to {save_dir}")

def lat_lon(dataset_name):
    file_path = os.path.join(DATA_PATH, dataset_name, '2m_temperature.nc') # use 2m_temperature as reference
    with safe_file_access(file_path) as ds:
        lat = ds.latitude.values
        lon = ds.longitude.values
    np.savez(
        os.path.join(DATA_PATH, dataset_name, 'processed_data','lat_lon.npz'),
        lat=lat,
        lon=lon
    )
    print(f"Saved lat/lon data: {lat.shape}, {lon.shape}")

def calculate_statistics(dataset_name):
    train_path = os.path.join(DATA_PATH, dataset_name, 'processed_data', 'train.npz')
    with np.load(train_path) as data:
        stats = {
            'mean': {k: np.mean(data[k]) for k in data if k != 'valid_time'}, # we do not use nanmean here to test whether data contains nan
            'std': {k: np.std(data[k]) for k in data if k != 'valid_time'}
        }
    for k in stats['mean']:
        print(f"[Mean/Std] {k}: {stats['mean'][k].shape}, Mean = {stats['mean'][k]}, Std = {stats['std'][k]}")
    for stat_type in ['mean', 'std']:
        np.savez(
            os.path.join(DATA_PATH, dataset_name, 'processed_data', f'norm_{stat_type}.npz'),
            **stats[stat_type]
        )
    print("Calculated mean and std from training data")

def calculate_climatology(dataset_name):
    clim_data = {}
    
    group = VARIABLES["atm_surface"]
    for var_name, var in zip(group["names"], group["vars"]):
        file_path = os.path.join(DATA_PATH, dataset_name, f'{var_name}.nc')
        with safe_file_access(file_path) as ds:
            data = ds[var].sel(valid_time=slice(f"{CLIM_START}-01-01", f"{CLIM_END}-12-31"))
            if var in group["unit_convert"]:
                data = group["unit_convert"][var](data)
            monthly_mean = data.groupby('valid_time.month').mean(dim='valid_time').values
            clim_data[var] = monthly_mean
            print(f"[Climatology] {var}: {monthly_mean.shape}, Mean = {np.mean(monthly_mean)}")

    group = VARIABLES["atm_pressure"]
    for var_name, var in zip(group["names"], group["vars"]):
        file_path = os.path.join(DATA_PATH, dataset_name, f'{var_name}.nc')
        with safe_file_access(file_path) as ds:
            for level in group["levels"]:
                data = ds[var].sel(
                    valid_time=slice(f"{CLIM_START}-01-01", f"{CLIM_END}-12-31"),
                    pressure_level=level
                )
                monthly_mean = data.groupby('valid_time.month').mean(dim='valid_time').values
                clim_data[f'{var}_{level}'] = monthly_mean
                print(f"[Climatology] {var}_{level}: {monthly_mean.shape}, Mean = {np.mean(monthly_mean)}")
    
    group = VARIABLES["boundary"]
    for var_name, var in zip(group["names"], group["vars"]):
        file_path = os.path.join(DATA_PATH, dataset_name, f'{var_name}.nc')
        with safe_file_access(file_path) as ds:
            data = ds[var].sel(valid_time=slice(f"{CLIM_START}-01-01", f"{CLIM_END}-12-31"))
            monthly_mean = data.groupby('valid_time.month').mean(dim='valid_time').values
            clim_data[var] = monthly_mean
            print(f"[Climatology] {var}: {monthly_mean.shape}, Mean = {np.mean(monthly_mean)}")

    np.savez(
        os.path.join(DATA_PATH, dataset_name, 'processed_data', 'climatology.npz'),
        **clim_data
    )
    print("Saved climatology data")

def main(args):
    """
    Calculate the data, mean_std and climatology of individual variables.
    Climatology uses 30 years from 1991-2020.
    Mean and Std only uses the training data from 1940-2015.
    Usage Example:
        # conver data to numpy first because tp and ssrd encounter the problem of nan when purly using xarray. Don't know why.
        (1) Data convert to numpy: python preprocess_monthly_china.py --item data --dataset china_025deg
        (2) Mean and std: python preprocess_monthly_china.py --item mean_std --dataset china_025deg
        (3) Climatology: python preprocess_monthly_china.py --item climatology --dataset china_025deg
    The computed mean, std and climatology will be saved in separate nc files with each file containing all the used variables listed above.
    """

    operations = {
        'data': lambda: [
            convert_to_numpy('train', args.dataset),
            convert_to_numpy('val', args.dataset),
            convert_to_numpy('test', args.dataset),
            lat_lon(args.dataset)
        ],
        'mean_std': lambda: calculate_statistics(args.dataset),
        'climatology': lambda: calculate_climatology(args.dataset)
    }
    
    if args.item not in operations:
        raise ValueError(f"Invalid operation {args.item}, choose from {list(operations.keys())}")
    
    operations[args.item]()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess monthly ERA5 data')
    parser.add_argument('--item', required=True, 
                       choices=['data', 'mean_std', 'climatology'],
                       help='Item to compute: mean_std or climatology')
    parser.add_argument('--dataset', required=True,
                       help='Dataset name (e.g. china_025deg)')
    args = parser.parse_args()
    
    main(args)
