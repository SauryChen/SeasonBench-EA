import os
import numpy as np

def compute_ACC(pred, obs, clim_pred, clim_obs):
    """
    Compute Anomaly Correlation Coefficient (ACC).
    
    Args:
        pred: [H, W] prediction values
        obs: [H, W] observation values
        clim_pred: [H, W] climatology of prediction (multi-year mean)
        clim_obs: [H, W] climatology of observation (multi-year mean)
    
    Returns:
        acc: float, spatially averaged anomaly correlation coefficient
    """
    pred_anom = pred - clim_pred
    obs_anom = obs - clim_obs
    pred_anom_mean = np.mean(pred_anom)
    obs_anom_mean = np.mean(obs_anom)
    pred_anom_ = pred_anom - pred_anom_mean
    obs_anom_ = obs_anom - obs_anom_mean
    pred_anom_flat = pred_anom_.flatten()
    obs_anom_flat = obs_anom_.flatten()
    numerator = np.sum(pred_anom_flat * obs_anom_flat)
    denominator = np.sqrt(np.sum(pred_anom_flat**2) * np.sum(obs_anom_flat**2))
    acc = numerator / denominator
    return float(acc)


def compute_TCC(pred, obs, pred_mean, obs_mean):
    """
    Compute Temporal Correlation Coefficient (TCC) at each grid point.

    Args:
        pred: [N, H, W] predicted values over N years
        obs: [N, H, W] observed values over N years
        pred_mean: [H, W] multi-year mean of predicted values
        obs_mean: [H, W] multi-year mean of observed values

    Returns:
        tcc: [H, W] temporal correlation at each grid point
    """
    pred_anom = pred - pred_mean[None, :, :]
    obs_anom = obs - obs_mean[None, :, :]
    numerator = np.sum(pred_anom * obs_anom, axis=0)  # shape: [H, W]

    pred_std = np.sqrt(np.sum(pred_anom ** 2, axis=0))
    obs_std = np.sqrt(np.sum(obs_anom ** 2, axis=0))
    denominator = pred_std * obs_std + 1e-8  # avoid division by zero

    tcc = numerator / denominator
    return tcc # shape: [H, W]

def main(args):
    """
    Example usage:
        python cal_metrics.py --item hind_corr --dir $DIR_TO_LOGS/logs_hindcast_corr/cmcc/graphcast/lightning_logs/version_0
    """
    save_dir = args.dir

    pred = np.load(os.path.join(save_dir, 'predict.npz'), allow_pickle=True)
    truth = np.load(os.path.join(save_dir, 'truth.npz'), allow_pickle=True)
    print(pred['test_june'].shape, truth['gt_june'].shape)

    test_june, test_july, test_aug = pred['test_june'], pred['test_july'], pred['test_august']
    gt_june, gt_july, gt_aug = truth['gt_june'], truth['gt_july'], truth['gt_august']
    clim_obs_june, clim_obs_july, clim_obs_aug = gt_june.mean(axis=0), gt_july.mean(axis=0), gt_aug.mean(axis=0)
    clim_pred_june, clim_pred_july, clim_pred_aug = test_june.mean(axis=0), test_july.mean(axis=0), test_aug.mean(axis=0)

    # combine three months into one
    clim_obs = (clim_obs_june * 30 + clim_obs_july * 31 + clim_obs_aug * 31) / 92
    clim_pred = (clim_pred_june * 30 + clim_pred_july * 31 + clim_pred_aug * 31) / 92
    test = (test_june * 30 + test_july * 31 + test_aug * 31) / 92
    gt = (gt_june * 30 + gt_july * 31 + gt_aug * 31) / 92

    if args.item == 'hind_pred':
        acc = [compute_ACC(test[i], gt[i], clim_pred, clim_obs) for i in range(test.shape[0])]
        print('ACC:', acc)
        tcc = compute_TCC(test, gt, clim_pred, clim_obs)
        print('Mean TCC:', np.mean(tcc))

    elif args.item == 'hind_corr':
        # crop the region
        crop_size = [181, 360] # change with the config file
        H, W = 181, 360
        lat = np.arange(90, -91, -1)
        lon = np.arange(-180, 180, 1)
        top = (H - crop_size[0]) // 2
        left = (W - crop_size[1]) // 2
        lat = lat[top:top + crop_size[0]]
        lon = lon[left:left + crop_size[1]]
        lat_idx = np.where((lat <= 60) & (lat >= 8))[0]
        lon_idx = np.where((lon >= 58) & (lon <= 163))[0]
        clim_obs = clim_obs[lat_idx][:, lon_idx]
        clim_pred = clim_pred[lat_idx][:, lon_idx]
        test = test[:, lat_idx][:, :, lon_idx]
        gt = gt[:, lat_idx][:, :, lon_idx]
        acc = [compute_ACC(test[i], gt[i], clim_pred, clim_obs) for i in range(test.shape[0])]
        print('ACC:', acc)
        tcc = compute_TCC(test, gt, clim_pred, clim_obs)
        print('Mean TCC:', np.mean(tcc))

    # save 
    np.savez(os.path.join(save_dir, 'ACC_TCC.npz'), acc=acc, tcc=tcc)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Calculate ACC and TCC")
    parser.add_argument('--item', type=str, required=True, help="Item to calculate metrics for (hind_corr or hind_pred)")
    parser.add_argument('--dir', type=str, required=True, help="Directory containing the prediction and ground truth files")
    args = parser.parse_args()
    main(args)