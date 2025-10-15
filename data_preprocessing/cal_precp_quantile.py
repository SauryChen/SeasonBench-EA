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
tp = ds['tp']

# for each month
months = np.arange(1, 13)
quantiles = [0.5, 0.75, 0.9, 0.95, 0.99]

def calculate_quantiles(data, months, quantiles):
    """
        data: [T, H, W]
        quantiles: [0.5, 0.75, 0.9, 0.95, 0.99]
    """
    monthly_quantiles = {}
    for m in months:
        tp_month = data.sel(valid_time = data['valid_time.month'] == m)
        q_values = np.quantile(tp_month, quantiles, axis=0)  # (len(quantiles), H, W)
        for i, q in enumerate(quantiles):
            monthly_quantiles[f'month_{m:02d}_q{int(q*100)}'] = q_values[i]
    return monthly_quantiles


quantile_values = calculate_quantiles(tp, months, quantiles)

np.savez(save_path / 'tp_monthly_quantiles.npz', **quantile_values)
print(f"Quantiles saved to {save_path / 'tp_monthly_quantiles.npz'}")



def plot_monthly_quantiles(quantile_values, save_dir):
    """
    quantile_values: dict, keys like 'month_01_q90'
    save_dir: output directory for figures
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    months = sorted({int(k.split('_')[1]) for k in quantile_values.keys()})
    quantiles = sorted({k.split('_')[-1] for k in quantile_values.keys()})
    print(f"Detected {len(months)} months, quantiles: {quantiles}")

    for m in months:
        monthly_data = {k: v for k, v in quantile_values.items() if f"month_{m:02d}_" in k}
        n_quantiles = len(monthly_data)
        fig, axes = plt.subplots(1, n_quantiles, figsize=(4 * n_quantiles, 4), constrained_layout=True)

        assert n_quantiles == len(quantiles), "Mismatch in number of quantiles"

        vmax = max(qv.max() for qv in monthly_data.values()) * 1000  # [m] -> [mm]
        vmin = 0

        for ax, (q_name, q_value) in zip(axes, monthly_data.items()):
            img = ax.imshow(q_value * 1000, cmap='viridis', vmin=vmin, vmax=vmax)
            ax.set_title(f'{q_name}', fontsize=10)
            ax.axis('off')

        cbar = fig.colorbar(img, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
        cbar.set_label('Precipitation (mm/month)', fontsize=10)

        fig.suptitle(f'Month {m:02d} Quantiles of Precipitation', fontsize=12)
        save_path = save_dir / f'tp_quantiles_month_{m:02d}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {save_path}")

# Example usage
plot_monthly_quantiles(quantile_values, save_dir='tp_monthly_quantile_figs')
