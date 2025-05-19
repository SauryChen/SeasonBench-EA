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

| Center           | CMCC [374GB] | DWD [325GB] | ECCC <br> [95GB]      | ECMWF [306GB] | Meteo-France [251GB] |
| ---------------- | ------------ | ----------- | ---------------- | ------------- | -------------------- |
| System           | SPS 3.5      | GCFS2.1     | GEM5-NEMO        | SEAS5         | System 8             |
| Ensemble Members | 40           | 30          | 10               | 25            | 25                   |
| Missing Years    | 2017-2020    | 2020        | 2021, 2023, 2024 | 2024          | 2019-2021            |

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

## 🔧 How to use

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
