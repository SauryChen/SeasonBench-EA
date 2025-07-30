# SeasonBench-EA: A multi-source benchmark for seasonal prediction and numerical model post-processing in East Asia
## 🌍 Overview

SeasonBench-EA is a multi-source benchmark dataset for advancing AI-based seasonal prediction with 1–6 month lead times. It integrates:

+ **ERA5 reanalysis data** at monthly, daily, and 6-hour resolutions.
+ **Ensemble forecasts** from multiple leading operational forecast centers (CMCC, DWD, ECMWF, ECCC, ECMWF, Météo-France) at monthly resolution.

The benchmark focuses on **East Asia** (8-60N, 58-163E) region and defines two tasks:

+ **Machine-learning based seasonal prediction using ERA5 reanalysis.** In baselines, models are trained using monthly reanalysis data from 1940 to 2015, validated on 2016 to 2019, and evaluated on 2020 to 2024. Models are trained and evaluated over East Asia.
+ **Post-processing of seasonal forecasts from numerical model ensembles.** In baselines, models are trained using numerical model results from 1993 to 2024, excluding validation (2009 to 2011) and test (2013 to 2016). Models are trained on global-scale to incorporate large-scale boundary information, and evaluated over East Asia.

The benchmark includes several evaluation metrics:

+ **Deterministic metrics:** Root mean square error (RMSE), Bias, Willmott's index of agreement (WI), Anomaly correlation coefficient (ACC), Energy spectrum, Critical success index (CSI)
+ **Probabilistic metrics:** Rank histogram, Continuous ranked probability score (CPRS), Spread-skill ratio (SSR)
+ **Hindcast evaluation:** Assessment of summer-season precipitation (June–August) over East Asia during the period 2006-2020, with ACC and TCC as metrics.

## 📊 Datasets

### Reanalysis Data Included in SeasonBench-EA

The reanalysis data is based on ERA5 data, obtained from the [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/). The following datasets are used:

[ERA5 hourly data on single levels from 1940 to present](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview)

[ERA5 hourly data on pressure levels from 1940 to present](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=overview)

[ERA5 monthly averaged data on single levels from 1940 to present](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-monthly-means?tab=overview)

[ERA5 monthly averaged data on pressure levels from 1940 to present](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels-monthly-means?tab=overview)

+ Temporal coverage and spatial resolution:
  + Monthly data: 1940-2024. 0.25 degree over East Asia (8-60N, 58-163E), 0.25 and 1 degree over the global
  + 6-hourly and daily data: 1991-2024. Daily variables are derived by averaging the 6-hourly data (except for total precipitation, which is accumulated from hourly values). 0.25 degree overEast Asia and 1 degree over the global.

> *Variables* are only available in the reanalysis dataset. All others are included in both the reanalysis and numerical model ensembles.

| **Type**                                          | **Variables**                                                |
| ------------------------------------------------- | ------------------------------------------------------------ |
| **Surface**                                       | 2m temperature, mean sea level pressure, total precipitation |
| **Pressure** <br />(1000, 850, 700, 500, 200 hPa) | temperature, u/v component of wind, geopotential, specific humidity |
| **Boundary**                                      | *boundary layer height*, *surface solar radiation downwards*, soil temperature, *volumetric soil water layers*, *snow albedo*, snow depth, sea surface temperature, sea ice cover |
| **Constant**                                      | geopotential at surface, land-sea mask, soil type            |

### Seasonal Forecasts from Numerical Model Ensembles

Numerical model ensembles are provided by the Copernicus Climate Change Service (C3S), and are obtained from [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/). The following datasets are used:

[Seasonal forecast monthly statistics on single levels](https://cds.climate.copernicus.eu/datasets/seasonal-monthly-single-levels?tab=overview)

[Seasonal forecast monthly statistics on pressure levels](https://cds.climate.copernicus.eu/datasets/seasonal-monthly-pressure-levels?tab=overview)

| Center                                    | CMCC [374GB]                       | DWD [325GB]                      | ECCC [95GB]   | ECMWF [306GB]                     | Meteo-France [251GB]             |
| ----------------------------------------- | ---------------------------------- | -------------------------------- | ------------- | --------------------------------- | -------------------------------- |
| System                                    | SPS 3.5                            | GCFS2.1                          | GEM5-NEMO     | SEAS5                             | System 8                         |
| Ensemble Members<br />hindcast / forecast | 40(1993-2016) <br />50 (2021-2024) | 30(1993-2019)<br />50(2021-2024) | 10(1990-2020)<br />10(2021-2024) | 25(1981-2016) <br />51(2017-2024) | 25(1993-2018)<br />51(2022-2024) |
| Missing Years                             | 2017-2020                          | 2020                             |               |                                   | 2019-2021                        |

These forecasts are provided at **monthly resolution** and **1.0° global coverage**, spanning **1993–2024**. Some years may be missing for specific models due to inconsistencies or changes in system versions; only uniform system periods are included to ensure comparability across models.

## ⚙️ Models

We provide a set of representative machine learning models for both tasks in SeasonBench-EA:

### Data-driven baselines

For machine-learning based seasonal prediction, U-Net, ViT, FNO, VAE

For post-processing: U-Net, ViT, SFNO (designed for global), VAE, GraphCast (designed for global)

### Physics baselines

Climatology, Persistence, NWP prediction

### Targeted variabels

2m temperature, total precipitation, temperature at 850 hPa, geopotential at 500 hPa, specific humidity at 700 hPa

## 📈 Leaderboard

For each variable and lead time, we highlight the **best and the second-best** performing models among these approaches.

### Prediction

Regarding RMSE, because all the models outperform the persistence baseline and underperform the climatology baseline, we report the results among the data-driven models.

| Variable [RMSE $\downarrow$] | Lead time 1                  | Lead time 2                  | Lead time 3                  | Lead time 4                  | Lead time 5                  | Lead time 6                  |
| ---------------------------- | ---------------------------- | ---------------------------- | ---------------------------- | ---------------------------- | ---------------------------- | ---------------------------- |
| tp [mm/day]                  | 2.00 (ViT) / 2.03 (Unet)     | 2.00 (ViT) / 2.14 (Unet)     | 2.15 (ViT) / 2.25 (Unet)     | 2.25 (ViT) / 2.36 (Unet)     | 2.32 (ViT) / 2.45 (Unet)     | 2.37 (ViT) / 2.48 (Unet)     |
| t2m [K]                      | 2.28 (Unet) / 2.32 (ViT)     | 2.74 (Unet) / 3.08 (ViT)     | 3.16 (Unet) / 4.06 (ViT)     | 3.52 (Unet) / 5.22 (ViT)     | 3.76 (Unet) / 5.70 (FNO)     | 3.89 (Unet) / 5.84 (FNO)     |
| t_850 [K]                    | 1.95 (Unet) / 1.95 (ViT)     | 2.24 (Unet) / 2.67 (ViT)     | 2.52 (Unet) / 3.63 (ViT)     | 2.80 (Unet) / 4.61 (FNO)     | 3.02 (Unet) / 5.09 (FNO)     | 3.17 (Unet) / 5.20 (FNO)     |
| z_500 [m²/s²]                | 362.20 (ViT) / 363.46 (Unet) | 392.71 (Unet) / 476.74 (FNO) | 431.39 (Unet) / 592.03 (FNO) | 476.75 (Unet) / 711.96 (FNO) | 505.98 (Unet) / 802.12 (FNO) | 516.84 (Unet) / 842.05 (FNO) |
| q_700 [g/kg]                 | 0.67 (ViT) / 0.71 (Unet)     | 0.75 (ViT) / 0.79 (Unet)     | 0.87 (ViT) / 0.89 (Unet)     | 0.99 (Unet) / 1.05 (ViT)     | 1.05 (Unet) / 1.19 (ViT)     | 1.06 (Unet) / 1.26 (ViT)     |

For ACC, since climatology always yields an ACC of zero, we include the persistence baseline in the comparison.

| Variable [ACC $\uparrow$] | Lead time 1                | Lead time 2                | Lead time 3                 | Lead time 4                   | Lead time 5                   | Lead time 6                |
| ------------------------- | -------------------------- | -------------------------- | --------------------------- | ----------------------------- | ----------------------------- | -------------------------- |
| tp                        | 0.17 (ViT) / 0.13 (Persis) | 0.12 (ViT) / 0.12 (Persis) | 0.11 (ViT) / 0.09 (Persis)  | 0.09 (ViT) / 0.06 (Persis)    | 0.07 (ViT) / 0.04 (VAE)       | 0.05 (ViT) / 0.03 (VAE)    |
| t2m                       | 0.21 (VAE) / 0.12 (Unet)   | 0.12 (VAE) / 0.00 (Unet)   | 0.07 (VAE) / -0.03 (Unet)   | 0.04 (VAE) / -0.04 (Unet)     | 0.04 (VAE) / -0.01 (Unet)     | 0.05 (VAE) / 0.02 (Unet)   |
| t_850                     | 0.18 (VAE) / 0.14 (Unet)   | 0.07 (VAE) / 0.00 (Unet)   | 0.02 (VAE) / -0.02 (Unet)   | 0.00 (VAE) / -0.04 (Unet)     | -0.01 (VAE) / -0.02 (Unet)    | 0.01 (VAE) / 0.00 (Unet)   |
| z_500                     | 0.18 (VAE) / 0.10 (FNO)    | 0.15 (VAE) / -0.03 (Unet)  | 0.09 (VAE) / -0.05 (Unet)   | 0.06 (VAE) / -0.08 (Unet)     | 0.05 (VAE) / -0.07 (Unet)     | 0.05 (VAE) / -0.03 (Unet)  |
| q_700                     | 0.25 (ViT) / 0.16 (Unet)   | 0.11 (ViT) / 0.03 (Unet)   | 0.00 (ViT) / -0.01 (Persis) | -0.06 (Unet) / -0.06 (Persis) | -0.07 (Persis) / -0.07 (Unet) | -0.05 (Unet) / -0.08 (VAE) |

Regarding bias, we report the absolute bias and include both persistence and climatology as reference points for a more comprehensive comparison.

| Variable [bias $\rightarrow$ 0] | Lead time 1                    | Lead time 2                    | Lead time 3                     | Lead time 4                     | Lead time 5                     | Lead time 6                     |
| ------------------------------- | ------------------------------ | ------------------------------ | ------------------------------- | ------------------------------- | ------------------------------- | ------------------------------- |
| tp [mm/day]                     | 0.01 (FNO) / 0.04 (Clim)       | 0.05 (Clim) / 0.13 (Persis)    | 0.06 (Clim) / 0.08 (ViT)        | 0.07 (Clim) / 0.08 (ViT)        | 0.08 (Clim) / 0.22 (ViT)        | 0.09 (Clim) / 0.30 (Persis)     |
| t2m [K]                         | 0.31 (Persis) / 0.38 (Unet)    | 0.59 (Clim) / 0.68 (Persis)    | 0.58 (Clim) / 0.96 (Unet)       | 0.57 (Clim) / 1.03 (Unet)       | 0.58 (Clim) / 0.99 (Unet)       | 0.60 (Clim) / 0.85 (Unet)       |
| t_850 [K]                       | 0.28 (Persis) / 0.28 (Unet)    | 0.47 (Clim) / 0.60 (Persis)    | 0.46 (Clim) / 0.81 (Unet)       | 0.45 (Clim) / 0.87 (Unet)       | 0.47 (Clim) / 0.86 (Unet)       | 0.48 (Clim) / 0.76 (Unet)       |
| z_500 [m²/s²]                   | 31.89 (Persis) / 109.68 (Unet) | 73.11 (Persis) / 135.50 (Clim) | 111.04 (Persis) / 132.71 (Clim) | 132.65 (Clim) / 139.60 (Persis) | 132.96 (Clim) / 148.76 (Persis) | 134.58 (Clim) / 137.06 (Persis) |
| q_700 [g/kg]                    | 0.04 (ViT) / 0.06 (Unet)       | 0.16 (Unet) / 0.17 (Clim)      | 0.18 (Clim) / 0.26 (Persis)     | 0.19 (Clim) / 0.30 (VAE)        | 0.20 (Clim) / 0.27 (VAE)        | 0.20 (Clim) / 0.23 (VAE)        |

### Correction

| Variable [RMSE $\downarrow$] | Lead time 1                | Lead time 2                 | Lead time 3                 | Lead time 4                  | Lead time 5                 | Lead time 6                 |
| ---------------------------- | -------------------------- | --------------------------- | --------------------------- | ---------------------------- | --------------------------- | --------------------------- |
| **tp [mm/day]**              | 1.69 (GC) / 1.69 (Clim)    | 1.74 (Clim) / 1.80 (GC)     | 1.77 (Clim) / 1.85 (ViT)    | 1.80 (Clim) / 1.86 (GC)      | 1.81 (Clim) / 1.87 (GC)     | 1.81 (Clim) / 1.88 (GC)     |
| **t2m [K]**                  | 1.40 (Clim) / 1.52 (GC)    | 1.39 (Clim) / 1.64 (GC)     | 1.39 (Clim) / 1.64 (GC)     | 1.39 (Clim) / 1.68 (GC)      | 1.41 (Clim) / 1.73 (GC)     | 1.44 (Clim) / 1.71 (GC)     |
| **t_850 [K]**                | 1.35 (GC) / 1.39 (Clim)    | 1.39 (Clim) / 1.49 (GC)     | 1.39 (Clim) / 1.50 (GC)     | 1.41 (Clim) / 1.52 (ViT)     | 1.42 (Clim) / 1.61 (GC)     | 1.44 (Clim) / 1.58 (GC)     |
| **z_500 [m²/s²]**            | 260.36 (GC) / 260.76 (ViT) | 290.75 (Clim) / 295.30 (GC) | 291.10 (Clim) / 298.57 (GC) | 293.33 (Clim) / 300.93 (ViT) | 292.41 (Clim) / 301.68 (GC) | 295.35 (Clim) / 307.79 (GC) |
| **q_700 [g/kg]**             | 0.51 (GC) / 0.52 (ViT)     | 0.56 (Clim) / 0.57 (GC)     | 0.56 (Clim) / 0.58 (GC)     | 0.57 (Clim) / 0.60 (GC)      | 0.57 (Clim) / 0.61 (GC)     | 0.57 (Clim) / 0.60 (GC)     |

| Variable [ACC $\uparrow$] | Lead time 1             | Lead time 2             | Lead time 3              | Lead time 4              | Lead time 5               | Lead time 6              |
| ------------------------- | ----------------------- | ----------------------- | ------------------------ | ------------------------ | ------------------------- | ------------------------ |
| **tp**                    | 0.26 (ENS) / 0.25 (GC)  | 0.14 (GC) / 0.13 (ViT)  | 0.10 (ViT) / 0.10 (GC)   | 0.11 (GC) / 0.10 (ViT)   | 0.10 (GC) / 0.08 (Unet)   | 0.10 (GC) / 0.07 (SFNO)  |
| **t2m**                   | 0.33 (GC) / 0.28 (SFNO) | 0.12 (SFNO) / 0.10 (GC) | 0.10 (ViT) / 0.09 (SFNO) | 0.14 (SFNO) / 0.05 (VAE) | 0.05 (ViT) / 0.03 (GC)    | 0.09 (GC) / 0.06 (SFNO)  |
| **t_850**                 | 0.34 (GC) / 0.32 (ViT)  | 0.12 (GC) / 0.10 (ViT)  | 0.15 (ViT) / 0.12 (SFNO) | 0.13 (ViT) / 0.13 (SFNO) | 0.08 (SFNO) / 0.07 (ViT)  | 0.10 (ViT) / 0.10 (SFNO) |
| **z_500**                 | 0.45 (ENS) / 0.40 (ViT) | 0.17 (ENS) / 0.17 (ViT) | 0.16 (ViT) / 0.15 (VAE)  | 0.18 (ViT) / 0.18 (SFNO) | 0.16 (SFNO) / 0.12 (VAE)  | 0.15 (SFNO) / 0.14 (VAE) |
| **q_700**                 | 0.38 (GC) / 0.36 (Unet) | 0.20 (GC) / 0.19 (ViT)  | 0.17 (GC) / 0.15 (ViT)   | 0.13 (GC) / 0.12 (VAE)   | 0.10 (Unet) / 0.09 (SFNO) | 0.11 (Unet) / 0.11 (GC)  |

For CRPS, we omit climatology in the comparison.

| Variable [CRPS $\downarrow$] | Lead time 1                | Lead time 2                 | Lead time 3                | Lead time 4                | Lead time 5                 | Lead time 6                |
| ---------------------------- | -------------------------- | --------------------------- | -------------------------- | -------------------------- | --------------------------- | -------------------------- |
| **tp [mm/day]**              | 0.90 (ViT) / 0.91 (GC)     | 0.99 (ViT) / 1.00 (GC)      | 1.02 (ENS) / 1.03 (ViT)    | 1.04 (ENS) / 1.04 (GC)     | 1.04 (ENS) / 1.06 (GC)      | 1.03 (ENS) / 1.03 (GC)     |
| **t2m [K]**                  | 0.84 (GC) / 0.91 (SFNO)    | 0.92 (GC) / 1.03 (SFNO)     | 0.92 (GC) / 1.04 (SFNO)    | 0.94 (GC) / 1.06 (SFNO)    | 0.98 (GC) / 1.07 (SFNO)     | 0.96 (GC) / 1.06 (SFNO)    |
| **t_850 [K]**                | 0.78 (GC) / 0.83 (SFNO)    | 0.87 (GC) / 0.93 (ViT)      | 0.89 (GC) / 0.90 (ViT)     | 0.88 (ViT) / 0.93 (GC)     | 0.95 (GC) / 0.97 (ViT)      | 0.92 (GC) / 0.95 (ViT)     |
| **z_500 [m²/s²]**            | 140.32 (GC) / 145.27 (ViT) | 168.83 (GC) / 185.89 (SFNO) | 172.05 (GC) / 177.47 (ViT) | 172.68 (ViT) / 175.49 (GC) | 170.78 (GC) / 181.98 (SFNO) | 173.61 (GC) / 182.97 (ViT) |
| **q_700 [g/kg]**             | 0.30 (GC) / 0.30 (ViT)     | 0.33 (ViT) / 0.34 (GC)      | 0.34 (GC) / 0.35 (ViT)     | 0.35 (GC) / 0.36 (Unet)    | 0.35 (Unet) / 0.36 (GC)     | 0.35 (GC) / 0.36 (Unet)    |

## 🔧 How to use

### Environment

Using requirements.txt to prepare for the environment.

### Data preparation

If you only wish to reproduce the baselines used in our paper (i.e., monthly reanalysis + CMCC), a subset can be downloaded from: [🔗 SeasonBench-EA (subset)](https://doi.org/10.7910/DVN/EPEUGO)

To access the full dataset, you can download it from:

 [🔗 Baidu netdisk](https://pan.baidu.com/s/1p78We3pCwU-eF3Xp-I0-Uw) (access code: xgq9)

After downloading, please organize the data as follows:

```
/root/Weather_Data/flood_season_data/
├── ERA5_monthly_mean/
│   ├── china_025deg/
│   └── global_1deg/
├── NWP_ensemble_monthly_mean/
│   ├── CMCC/
│   ├── .../
└── └── ECMWF/
```

### For prediction

+ **Data preprocessing** using scripts in `data_preprocessing` folder.

  + Convert the required variables from NetCDF format to .npz files. During preprocessing, longitudes are converted from the [0, 360) range to [-180, 180), and all NaN values are properly handled and replaced with appropriate fill values.

    ```python
    # convert netcdf to npz file:
    python preprocess_monthly_china.py --item data --dataset china_025deg
    # calculate mean and std: 
    python preprocess_monthly_china.py --item mean_std --dataset china_025deg
    # calculate climatology:
    python preprocess_monthly_china.py --item climatology --dataset china_025deg
    # calculate quantile for precipitation if CSI is used:
    python cal_precp_quantile.py
    ```

+ **Train** the model with `train.py`.

  + ```python
    python train.py --item prediction --config seasonbench/prediction/fno_config.yaml
    ```

  + If you add new models, please include `${MODEL}_config.yaml` and register the model under `SeasonalPred` class in `prediction/model.py`

+ **Evaluation** with `eval_pred_iter.py`

  + ```python
    # climatology:
    python eval_pred_iter.py --dataset_name china_025deg --model_name climatology
    # persistence:
    python eval_pred_iter.py --dataset_name china_025deg --model_name persistence
    # data-driven models:
    python eval_pred_iter.py --dataset_name china_025deg --model_name fno --version_num 0
    # --version_num corresponds to the log version in logs/prediction/china_025deg/{MODEL}/lightning_logs/version_{VERSION_NUM}.
    ```

+ **Visualize** the perdiction with scripts in `visualization` folder.

  + ```python
    # prediction results (for --vars, all variables listed in the model_args:output_vars in yaml file can be chosen):
    python plot_prediction.py --device 1 
                              --dataset_name china_025deg 
                              --model_name fno
                              --version 1 
                              --vars t2m t_850 z_500 q_700 tp
    ```

  + Example output (*will be similar to*):
    ![t2m_2021-01](/example_figs/t2m_2021-01.png)
    *2021-01* is the start month; step 1 to 6 correspond to forecasts from 2021-02 to 2021-07.

+ **Visualize** the metrics with scripts in `visualization` folder.

  + ```python
    python plot_metrics_pred.py --item prediction
                                --dataset_name china_025deg
                                --model_names fno vit vit persistence climatology 
                                --version 0 0 1 0 0 
                                --metrics rmse bias acc es csi 
                                --vars t2m t_850 z_500 q_700 tp
    # for persistence and climatology, the default version is 0.
    ```

  + Example output (*will be similar to*): 
    ![rmse](/example_figs/rmse.png)
    ![es_1](/example_figs/es_1.png)
    Each metric will generate .png plots with selected variables and models.

### For correction

+ **Data preprocessing** using using scripts in `data_preprocessing` folder

  + ```python
    # convert netcdf to npz file:
    python process_nwp.py --center_name cmcc
    # convert netcdf to npz file / mean and std / climatology:
    python preprocess_monthly_global.py --item data --dataset global_1deg
    python preprocess_monthly_global.py --item mean_std --dataset global_1deg
    python preprocess_monthly_global.py --item climatology --dataset global_1deg
    # calculate quantile for precipitation if CSI is used, change the file path:
    python cal_precp_quantile.py
    # convert numerical model netcdf to npz file (all test ensembles) for evaluation:
    python process_nwp_all.py --center_name cmcc
    ```

+ **Train**, **evaluation** and **visualization** are similar to steps in prediction, with files in `correction` folder.

  + ```python
    # train (for graphcast, set torch.set_float32_matmul_precision('high'))
    python train.py --item correction --config seasonbench/correction/fno_config.yaml
    # evaluation of the original NWP performance (all ensembles)
    python eval_nwp.py --item prediction --center cmcc
    # evaluation of the NWP performance (10 ensembles) after correction
    python eval_nwp.py --item correction --center cmcc --model_name fno --version 0
    # visualize correction results:
    python plot_correction.py --device 7 --center cmcc --model_name graphcast --version 0
    # visualize metrics:
    python plot_metrics_corr.py --item correction 
                                --center cmcc 
                                --model_names nwp unet vit 
                                --version 0 0 0 
                                --metrics rmse bias wi acc es rank_hist crps ssr 
                                --vars t2m t_850 z_500 q_700 tp
    ```

  + Example outputs (*will be similar to*):
    ![tp_2014-01](/example_figs/tp_2014-01.png)
    Step 1 ~ Step 6 refers to the 2014-01 ~ 2014-06.

  + Example outputs (*will be similar to*):
    ![acc](/example_figs/acc.png)
    ![es_6](/example_figs/es_6.png)
    ![hist_rank_6](/example_figs/hist_rank_6.png)
    Each metric will generate .png plots with selected variables and models.

## 📢 Reference & Statement

Part of the framework design is inspired by [ChaosBench](https://github.com/leap-stc/ChaosBench/tree/main). The GraphCast implementation is adapted from [NVIDIA - physicsnemo](https://github.com/NVIDIA/physicsnemo/tree/main), and SFNO is implemented based on [NVIDIA - torch-harmonics](https://github.com/NVIDIA/torch-harmonics/tree/main).

All experiments are conducted with NVIDIA H100 Tensor Core GPU.

SeasonBench-EA will be continuously updated to include additional data sources, variables, and evaluation protocols to support broader research in seasonal prediction.
