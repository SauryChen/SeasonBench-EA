"""
Convert .nc files to .npz files
read single vars from DATA_PATH/center_name/single_level/YYYY_MM.nc
read pressure vars from DATA_PATH/center_name/pressure_levels/YYYY_MM.nc
save single vars to DATA_PATH/center_name/processed_data/single_level/YYYY_MM.npz
save pressure vars to DATA_PATH/center_name/processed_data/pressure_levels/YYYY_MM.npz

for all variables:
    - open_dataset_with_fixed_longitude
    - regrid_nwp_data
    - convert the units if needed
    - replace nan if needed
"""
import os
import argparse
import numpy as np
import xarray as xr

VARIABLES = {
    "single_level":{
        "vars": ["t2m", "msl", "sst", "siconc", "sd", "stl1", "tprate"],
        "unit_convert":{'tprate': lambda x: x * 1000 * 60 * 60 * 24}, # m/s -> mm/day
        "replace_nan_with_zero": ["siconc", "sd"],
        "replace_nan_with_t2m": ["sst","stl1"],
    },
    "pressure_levels":{
        "vars": ["t", "z", "q", "u", "v"],
        "levels": [1000, 850, 700, 500, 200],
    },
}

DATA_PATH = "/root/Weather_Data/flood_season_data/NWP_ensemble_monthly_mean"

def make_dir(dir_name):
    os.makedirs(dir_name, exist_ok=True)

def safe_file_access(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} doesn't exist.")
    return xr.open_dataset(file_path)

def open_dataset_with_fixed_longitude(file_path):
    with safe_file_access(file_path) as ds:
        if 'longitude' in ds.coords:
            lon = ds['longitude']
            if lon.min() >= 0 and lon.max() > 180:
                ds['longitude'] = ((lon + 180) % 360) - 180
                ds = ds.sortby('longitude')
                ds['longitude'].attrs['units'] = 'degrees_east'
                ds['longitude'].attrs['standard_name'] = 'longitude'
                ds['longitude'].attrs['long_name'] = 'longitude'
        return ds

def regrid_nwp_data(ds, target_lon, target_lat):
    ds = ds.interp(longitude=target_lon, latitude=target_lat, method='linear', kwargs={'fill_value': 'extrapolate'})
    return ds

def process_variable_group(ds_name, target_lon, target_lat):
    np_vars = {}
    group = VARIABLES["single_level"]
    
    file_path = os.path.join(DATA_PATH, args.center_name, 'single_level', ds_name)
    ds = open_dataset_with_fixed_longitude(file_path)
    ds = regrid_nwp_data(ds, target_lon, target_lat)

    for var in group["vars"]:

        data = ds[var].isel(number=slice(0, 10)).values # only use the first 10 members
        if var in group["unit_convert"]:
            data = group["unit_convert"][var](data)
        if var in group["replace_nan_with_zero"]:
            data = np.nan_to_num(data, nan=0.0)
        elif var in group["replace_nan_with_t2m"]:
            data = np.nan_to_num(data, nan=ds['t2m'].isel(number=slice(0, 10)).values) # only use the first 10 members
        
        np_vars[var] = data
        if np.isnan(data).sum() > 0:
            print(f"[Single Level] {var}: {data.shape}, nans: {np.isnan(data).sum()}")
        # save
        save_path = os.path.join(single_save_dir, ds_name.replace('.nc', '.npz'))
        np.savez(save_path, **np_vars)

    np_vars = {}
    group = VARIABLES["pressure_levels"]
    file_path = os.path.join(DATA_PATH, args.center_name, 'pressure_levels', ds_name)
    ds = open_dataset_with_fixed_longitude(file_path)
    ds = regrid_nwp_data(ds, target_lon, target_lat)

    for var in group["vars"]:

        for level in group["levels"]:
            data = ds[var].isel(number=slice(0, 10)).sel(pressure_level=level).values # only use the first 10 members
            np_vars[f'{var}_{level}'] = data

            if np.isnan(data).sum() > 0:
                print(f"[Pressure Level] {var}_{level}: {data.shape}, nans: {np.isnan(data).sum()}")
        
        # save
        save_path = os.path.join(pressure_save_dir, ds_name.replace('.nc', '.npz'))
        np.savez(save_path, **np_vars)

def convert_to_numpy(nwp_file, lon, lat):    
    process_variable_group(nwp_file, lon, lat)
    print(f"Processed {nwp_file}")

if __name__ == "__main__":
    import multiprocessing as mp
    parser = argparse.ArgumentParser(description="Convert NWP data to numpy format")
    parser.add_argument('--center_name', type=str, required=True, help="Center name of the NWP data")
    args = parser.parse_args()

    pressure_save_dir = os.path.join(DATA_PATH, args.center_name, 'processed_data_10ens', 'pressure_levels')
    single_save_dir = os.path.join(DATA_PATH, args.center_name, 'processed_data_10ens', 'single_level')
    make_dir(pressure_save_dir)
    make_dir(single_save_dir)

    lat_lon = np.load('/root/Weather_Data/flood_season_data/ERA5_monthly_mean/global_1deg/processed_data/lat_lon.npz')
    lat, lon = lat_lon['lat'], lat_lon['lon']
    if lon.min() >= 0 and lon.max() > 180: # [0, 359]
        lon = ((lon + 180) % 360) - 180
        lon = np.sort(lon) # [-180, 179] # sort the longitude for interp !

    nwp_file_list = []
    for file_name in os.listdir(os.path.join(DATA_PATH, args.center_name, 'single_level')):
        if file_name.endswith('.nc') and not file_name.startswith('2024'):
            nwp_file_list.append(file_name)

    n_proc = min(16, len(nwp_file_list))
    tasks = [(f, lon, lat) for f in nwp_file_list]
    with mp.Pool(n_proc) as pool:
        pool.starmap(convert_to_numpy, tasks)
