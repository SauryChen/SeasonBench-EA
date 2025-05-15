import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent

def plot_standard_metric(ax, x, values, label, metric_name, show_ylabel=True):
    ax.plot(x, values, label=label)
    ax.set_xlabel('Time Step (Month)')
    ax.set_xticks(x)
    ax.set_xticklabels(x, fontsize=8)
    if show_ylabel:
        ax.set_ylabel(metric_name.upper())

def plot_spectra_metric(ax, wavenumbers, spectra, label, show_ylabel=True):
    spectra = np.clip(spectra, 1e-8, None)

    ax.plot(wavenumbers, spectra, label=label, linewidth=1.5)
    ax.set_xscale('log')
    ax.set_yscale('log')

    if show_ylabel:
        ax.set_ylabel('Mean Power')

    ax.set_xlabel('Zonal Wavenumber')

    ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=[1, 3]))
    ax.xaxis.set_major_formatter(ticker.LogFormatterSciNotation(
        base=10, 
        labelOnlyBase=False,
        minor_thresholds=(10, 10),
    ))
    ax.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2,10)*0.1))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())

    ax.grid(True, which='major', linestyle='--', linewidth=0.5)
    ax.grid(True, which='minor', linestyle=':', linewidth=0.3, alpha=0.5)


def plot_metrics(args):
    """
    Plot metrics after evaluation.
    Example usage:
        python plot_metrics.py --item correction
                               --center cmcc
                               --model_names fno nwp # nwp is the original prediction
                               --version 0 0 # default version for nwp is should be set to 0
                               --metrics rmse bias
                               --vars t2m z_500 tp
    python plot_metrics_corr.py --item correction --center cmcc --model_names nwp unet vit --version 0 0 0 --metrics rmse bias wi acc es rank_hist crps ssr --vars t2m t_850 z_500 q_700 tp
    """
    sns.set(style="darkgrid")
    sns.set_context("paper")
    for metric in args.metrics:
        if metric == 'rank_hist':
            for t in range(6):
                # ================== plot NWP ==================
                fig_nwp, axes_nwp = plt.subplots(1, len(args.vars), figsize=(3*len(args.vars), 3), sharey=True)
                fig_nwp.suptitle(f'Rank Histogram (NWP) - Time Step {t+1}', y=1.05)

                if 'nwp' in args.model_names:
                    model_name = 'nwp'
                    metrics_dir = BASE_DIR /f'logs_{args.item}'/ args.center / 'metrics'
                    metrics_file = os.path.join(metrics_dir, f'{args.center}_metrics.npz')
                    if os.path.exists(metrics_file):
                        data = np.load(metrics_file, allow_pickle=True)
                        for i, var in enumerate(args.vars):
                            ax = axes_nwp[i] if len(args.vars) > 1 else axes_nwp
                            key = f'rank_hist_{var}'
                            if key not in data:
                                print(f"Key {key} not found in {metrics_file}.")
                                continue
                            hist_data = data[key][t] # [n_bins]
                            bins = np.arange(len(hist_data))
                            ax.bar(bins, hist_data, alpha = 0.8, label = f'{args.center}')
                            ax.set_title(f'{var}', fontsize=12)
                            ax.set_xlabel('Rank of observation')
                            if i == 0: ax.set_ylabel('Number')
                
                plt.tight_layout()
                save_dir = BASE_DIR / f'logs_{args.item}/{args.center}/plots'
                print(metric, save_dir)
                os.makedirs(save_dir, exist_ok=True)
                save_path = f'{save_dir}/hist_rank_nwp_{t+1}.png'
                plt.savefig(save_path, bbox_inches='tight')
                plt.close(fig_nwp)

                # ================== plot model ==================
                fig_other, axes_other = plt.subplots(1, len(args.vars), figsize=(3*len(args.vars), 3), sharey=True)
                fig_other.suptitle(f'Rank Histogram (Models) - Time Step {t+1}', y=1.05)
                for model_name, version in zip(args.model_names, args.version):
                    if model_name == 'nwp': continue
                    metrics_dir = BASE_DIR /f'logs_{args.item}'/ args.center / model_name /f'lightning_logs/version_{version}'
                    metrics_file = os.path.join(metrics_dir, f'{args.center}_metrics.npz')
                    if not os.path.exists(metrics_file):
                        continue
                    data = np.load(metrics_file, allow_pickle = True)
                    for i, var in enumerate(args.vars):
                        ax = axes_other[i] if len(args.vars) > 1 else axes_other
                        key = f'rank_hist_{var}'
                        if key in data:
                            hist_data = data[key][t]
                            bins = np.arange(len(hist_data))
                            ax.bar(bins, hist_data, alpha = 0.8, label=f'{model_name} - {version}')
                            ax.set_title(f'{var}', fontsize=12)
                            ax.set_xlabel('Rank of observation')
                            if i == 0: ax.set_ylabel('Number')
                handles, labels = axes_other[0].get_legend_handles_labels() if len(args.vars) > 1 else axes.get_legend_handles_labels()
                fig_other.legend(handles, labels, loc='lower center', ncol=len(args.model_names)-1, fontsize=10, bbox_to_anchor=(0.5, -0.2))
                save_dir = BASE_DIR / f'logs_{args.item}/{args.center}/plots'
                print(metric, save_dir)
                os.makedirs(save_dir, exist_ok=True)
                filtered_models = [name for name in args.model_names if name != 'nwp']
                save_path = f'{save_dir}/hist_rank_{filtered_models}_{t+1}.png'
                plt.savefig(save_path, bbox_inches='tight')
                plt.close(fig_other)
            continue


        if metric == 'es':
            for t in range(6):
                fig, axes = plt.subplots(1, len(args.vars), figsize=(3 * len(args.vars), 3), sharey=False)
                for i, var in enumerate(args.vars):
                    ax = axes[i] if len(args.vars) > 1 else axes

                    PLOT_ESGT = True
                    for model_name, version in zip(args.model_names, args.version):
                        if model_name == 'nwp':
                            metrics_dir = BASE_DIR /f'logs_{args.item}'/ args.center / 'metrics'
                        else:
                            metrics_dir = BASE_DIR /f'logs_{args.item}'/ args.center / model_name /f'lightning_logs/version_{version}'
                        metrics_file = os.path.join(metrics_dir, f'{args.center}_metrics.npz')
                        if not os.path.exists(metrics_file):
                            print(f"Metrics file {metrics_file} does not exist.")
                            continue
                        data = np.load(metrics_file, allow_pickle=True)
                        key = f'{metric}_{var}'
                        if key not in data:
                            print(f"Key {key} not found in {metrics_file}.")
                            continue
                        metric_values = data[key]
                        label = f'{model_name} - {version}' if model_name not in ['nwp'] else model_name
                        wavenumbers = np.arange(metric_values.shape[1])
                        show_ylabel = (i == 0)
                        plot_spectra_metric(ax, wavenumbers, metric_values[t], label, show_ylabel)

                        gt_key = f'es_gt_{var}'
                        if gt_key in data and PLOT_ESGT:
                            gt_spectra = data[gt_key]
                            plot_spectra_metric(ax, wavenumbers, gt_spectra[t], 'ERA5', show_ylabel=False)
                            PLOT_ESGT = False

                    ax.set_title(f'{var} - {t+1} month', fontsize=12)
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                handles, labels = axes[0].get_legend_handles_labels() if len(args.vars) > 1 else axes.get_legend_handles_labels()
                fig.legend(handles, labels, loc='lower center', 
                           ncol=len(args.model_names)+1 if 'nwp' in args.model_names else len(args.model_names),
                           fontsize=10, bbox_to_anchor=(0.5, -0.2))
                save_dir = BASE_DIR / f'logs_{args.item}/{args.center}/plots'
                print(metric, save_dir)
                os.makedirs(save_dir, exist_ok=True)
                save_path = f'{save_dir}/{metric}_{t+1}_{args.model_names}.png'
                plt.savefig(save_path, bbox_inches='tight')
                plt.close(fig)
            continue
        
        if metric == 'csi':
            fig, ax_list = plt.subplots(1, len(args.vars), figsize=(3 * len(args.vars), 3), sharey=True)
            for i, ax in enumerate(ax_list):
                ax.set_title(f'CSI - q{[50, 75, 90, 95, 99][i]}')
                ax.set_xlabel('Lead Time (Month)')
                if i == 0:
                    ax.set_ylabel('CSI')
                ax.grid(True, linestyle='--', linewidth=0.5)
                ax.tick_params(labelsize=8)

            for model_name, version in zip(args.model_names, args.version):
                # Load the data
                if model_name == 'nwp':
                    metrics_dir = BASE_DIR /f'logs_{args.item}'/ args.center / 'metrics'
                else:
                    metrics_dir = BASE_DIR /f'logs_{args.item}'/ args.center / model_name /f'lightning_logs/version_{version}'
                metrics_file = os.path.join(metrics_dir, f'{args.center}_metrics.npz')
                if not os.path.exists(metrics_file):
                    print(f"Metrics file {metrics_file} does not exist.")
                    continue

                data = np.load(metrics_file, allow_pickle=True)
                key = 'csi_tp'
                if key not in data:
                    print(f"Key {key} not found in {metrics_file}.")
                    continue

                csi_array = data[key] # (N_steps, 5)
                label = f'{model_name} - {version}' if model_name not in ['nwp'] else model_name
                x = np.arange(1, csi_array.shape[0] + 1)
                for j in range(5):
                    ax_list[j].plot(x, csi_array[:, j], label=label, linewidth=1.5)

            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            handles, labels = ax_list[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', ncol=len(args.model_names), fontsize=10, bbox_to_anchor=(0.5, -0.2))
            save_dir = BASE_DIR / f'logs_{args.item}/{args.center}/plots'
            print(metric, save_dir)
            os.makedirs(save_dir, exist_ok=True)
            save_path = f'{save_dir}/{metric}_{args.model_names}.png'
            plt.savefig(save_path, bbox_inches='tight')
            plt.close(fig)

            continue

        num_vars = len(args.vars)
        cols = num_vars
        rows = 1
        fig, axes = plt.subplots(rows, cols, figsize=(3 * len(args.vars), 3))
        if num_vars == 1:
            axes = [axes]

        for i, var in enumerate(args.vars):
            ax = axes[i]
            for model_name, version in zip(args.model_names, args.version):
                # Load the data
                if model_name in ['nwp']:
                    metrics_dir = BASE_DIR /f'logs_{args.item}'/ args.center / 'metrics'
                else:
                    metrics_dir = BASE_DIR /f'logs_{args.item}'/ args.center / model_name /f'lightning_logs/version_{version}'
                metrics_file = os.path.join(metrics_dir, f'{args.center}_metrics.npz')
                if not os.path.exists(metrics_file):
                    print(f"Metrics file {metrics_file} does not exist.")
                    continue

                data = np.load(metrics_file, allow_pickle=True)
                key = f'{metric}_{var}'

                if key not in data:
                    print(f"Key {key} not found in {metrics_file}.")
                    continue

                metric_values = data[key]
                label = f'{model_name} - {version}' if model_name not in ['nwp'] else model_name
                if isinstance(metric_values, np.ndarray):
                    x = np.arange(len(metric_values)) + 1
                    x = x[0:6]
                    show_ylabel = (i == 0)
                    plot_standard_metric(ax, x, metric_values[0:6], label, metric, show_ylabel)

            ax.set_title(f'{var}', fontsize=12)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        handles, labels = [], []
        first_ax = axes[0] if num_vars > 1 else axes
        handles, labels = first_ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', 
                   ncol=len(args.model_names) if metric != 'es' else len(args.model_names)+1,
                   fontsize=10, bbox_to_anchor=(0.5, -0.1))

        save_dir = BASE_DIR / f'logs_{args.item}/{args.center}/plots'
        print(metric, save_dir)
        os.makedirs(save_dir, exist_ok=True)
        save_path = f'{save_dir}/{metric}_{args.model_names}.png'
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--item', type=str, required=True, help='correction')
    parser.add_argument('--center', type=str, required=True)
    parser.add_argument('--model_names', nargs='+', required=True, help='List of model names')
    parser.add_argument('--version', nargs='+', required=True, help='List of versions for each model')
    parser.add_argument('--metrics', nargs='+', required=True, help='List of metrics to plot (e.g., rmse, bias)')
    parser.add_argument('--vars', nargs='+', required=True, help='List of variables to plot (e.g., t2m, z500)')

    args = parser.parse_args()
    plot_metrics(args)