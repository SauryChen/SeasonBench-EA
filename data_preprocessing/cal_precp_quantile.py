# Calculate the q50，q75, q90, q95, q99 of the precipitation data from the monthly mean
# calculate at each lat-lon grid
# if use hourly or daily data, the quantile should be calculated at each corresponding time step!
import numpy as np
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt

china_025 = Path('/root/Weather_Data/flood_season_data/ERA5_monthly_mean/china_025deg')
# global_1 = Path('/root/Weather_Data/flood_season_data/ERA5_monthly_mean/global_1deg')
# global_025 = Path('/root/Weather_Data/flood_season_data/ERA5_monthly_mean/global_025deg')
tp_file = 'total_precipitation.nc'
save_path = china_025 / 'processed_data'
# save_path = global_1 / 'processed_data'
save_path.mkdir(parents=True, exist_ok=True)

ds = xr.open_dataset(china_025 / tp_file)
# ds = xr.open_dataset(global_1 / tp_file)
tp = ds['tp'].values

def calculate_quantiles(data, quantiles):
    """
        data: [T, H, W]
        quantiles: [0.5, 0.75, 0.9, 0.95, 0.99]
    """
    quantile_values = {}
    for q in quantiles:
        quantile_values[f"q{int(q*100)}"] = np.quantile(data, q, axis=0)  # (H, W)
    return quantile_values

quantiles = [0.5, 0.75, 0.9, 0.95, 0.99]
quantile_values = calculate_quantiles(tp, quantiles)

np.savez(save_path / 'tp_quantiles.npz', **quantile_values)
print(f"Quantiles saved to {save_path / 'tp_quantiles.npz'}")


def plot_quantiles(quantile_values):
    n_quantiles = len(quantile_values)
    fig, axes = plt.subplots(1, n_quantiles, figsize=(4 * n_quantiles, 4), constrained_layout=True)

    if n_quantiles == 1:
        axes = [axes]

    vmax = max(qv.max() for qv in quantile_values.values()) * 1000 # [m] -> [mm]
    vmin = 0

    for ax, (q_name, q_value) in zip(axes, quantile_values.items()):
        img = ax.imshow(q_value * 1000, cmap='viridis', vmin=vmin, vmax=vmax) # [m] -> [mm]
        ax.set_title(f'{q_name}', fontsize=10)
        ax.axis('off')

    cbar = fig.colorbar(img, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label('Precipitation (mm/day)', fontsize=10)
    plt.savefig('tp_quantiles_china025monthly.png', dpi=300)

plot_quantiles(quantile_values)
