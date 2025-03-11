import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import torch
from lombScargle import LombScargleBatchMask
import seaborn as sns
import scipy
import os.path as op

def quantile_loss(target, forecast, q: float, eval_points) -> float:
    return 2 * torch.sum(
        torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
    )


def calc_denominator(target, eval_points):
    return torch.sum(torch.abs(target * eval_points))


def calc_quantile_CRPS(target, forecast, eval_points, mean_scaler, scaler):

    target = target * scaler + mean_scaler
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(torch.quantile(forecast[j : j + 1], quantiles[i], dim=1))
        q_pred = torch.cat(q_pred, 0)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom

    return CRPS.item() / len(quantiles)

def calc_quantile_CRPS_sum(target, forecast, eval_points, mean_scaler, scaler):

    eval_points = eval_points.mean(-1)
    target = target * scaler + mean_scaler
    target = target.sum(-1)
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = torch.quantile(forecast.sum(-1),quantiles[i],dim=1)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)

def get_quantile(samples,q,dim=1):
    return torch.quantile(samples,q,dim=dim).cpu().numpy()

def reconstruct_data(samples, all_target, all_evalpoint, all_observed):
    all_given = all_observed - all_evalpoint
    median_sample = samples.median(dim=1).values * (1 - all_given) + all_target * all_given
    return median_sample

def leadfreqs(freqs, spectra):
    return freqs[np.argmax(spectra, axis=1)]

def plot_leadfreqs(freqs, spectra, label=''):
    lead_freqs = leadfreqs(freqs, spectra)
    sns.kdeplot(lead_freqs, label=label)

def plot_all_leadfreqs(freqs, spectra_gt, spectra_pred, fname=None):
    B, K, L = spectra_gt.shape
    for k in range(K):
        plt.figure()
        plot_leadfreqs(freqs, spectra_gt[:, k, :], label='GT')
        plot_leadfreqs(freqs, spectra_pred[:, k, :], label='Pred')
        plt.title(f'Lead freqs for feature {k}')
        plt.legend()
        if fname is not None:
            plt.savefig(f'{fname}_{k}.png', dpi=300, bbox_inches="tight")
        plt.close()

def mean_confidence_interval(data, confidence=0.95, axis=0):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a, axis=axis), scipy.stats.sem(a, axis=axis)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def mean_spectrum(freqs, ps, normalize=True):
    dfreq = freqs[1] - freqs[0]
    ps_mean = ps.mean(axis=0)
    if normalize:
        factor = 1.0 / (ps_mean * dfreq).sum()
        return ps_mean * factor, factor
    return ps_mean, 1.0

def plot_spectra_mean(freqs, spectra, confidence=0.95, label=None):
    ps_mean, factor = mean_spectrum(freqs.numpy(), spectra)
    plt.plot(freqs, ps_mean, label=label)
    if confidence is not None:
        mean, lower, upper = mean_confidence_interval(spectra*factor, confidence=confidence, axis=0)
        plt.fill_between(freqs.numpy(), lower, upper, alpha=0.25)

def plot_all_spectra_mean(freqs, spectra_gt, spectra_pred, fname=None):
    B, K, L = spectra_gt.shape
    for k in range(K):
        plt.figure()
        plot_spectra_mean(freqs, spectra_gt[:, k, :], label='GT')
        plot_spectra_mean(freqs, spectra_pred[:, k, :], label='Pred')
        plt.title(f'Spectra for feature {k}')
        plt.legend()
        if fname is not None:
            plt.savefig(f'{fname}_{k}.png', dpi=300, bbox_inches="tight")
        plt.close()

def LS_omegas(t, samples_per_peak=1):
    dt_min = np.min(np.diff(t))
    omega_max = np.pi / dt_min
    # Nomegas_nyq = int(t.max() / (2 * dt_min))
    Nomegas_nyq = len(t)
    ls_omegas = np.linspace(1e-5, omega_max, samples_per_peak * Nomegas_nyq)
    return ls_omegas

def compute_spectral_loss(median_sample, all_target, all_evalpoint, all_observed, all_observed_time, fap=False, path='.', plot=True):
    suffix = 'fap' if fap else ''
    # ls_omegas = 2*np.pi*torch.linspace(1e-3, 5.0, 500)
    ls_omegas = torch.tensor(LS_omegas(all_observed_time[0].cpu().numpy(), samples_per_peak=1)) #*2*np.pi
    W = len(ls_omegas)
    B, L, K = median_sample.shape
    ls = LombScargleBatchMask(ls_omegas.to('cuda'))
    all_target_r = all_target.permute(0, 2, 1).reshape(B*K, L)
    all_observed_time_r = all_observed_time.repeat(K,1)
    all_observed_r = all_observed.permute(0, 2, 1).reshape(B*K, L)
    gt = ls(all_observed_time_r, all_target_r, mask=all_observed_r, fap=fap)
    gt_r = gt.reshape(B, K, W)

    median_sample_r = median_sample.permute(0, 2, 1).reshape(B*K, L)
    pred_masked = ls(all_observed_time_r, median_sample_r, mask=all_observed_r, fap=fap)
    pred_masked_r = pred_masked.reshape(B, K, W)
    if plot:
        plot_all_leadfreqs(ls_omegas/(2*np.pi), gt_r.cpu().numpy(), pred_masked_r.cpu().numpy(), fname=op.join(path, f'leadfreqs_masked_{suffix}'))
        plot_all_spectra_mean(ls_omegas/(2*np.pi), gt_r.cpu().numpy(), pred_masked_r.cpu().numpy(), fname=op.join(path, f'spectra_masked_{suffix}'))
    loss = torch.nn.MSELoss()(gt, pred_masked)
    loss_mae = torch.nn.L1Loss()(gt, pred_masked)
    pred_full = ls(all_observed_time_r, median_sample_r, fap=fap)
    pred_full_r = pred_full.reshape(B, K, W)
    if plot:
        plot_all_leadfreqs(ls_omegas/(2*np.pi), gt_r.cpu().numpy(), pred_full_r.cpu().numpy(), fname=op.join(path, f'leadfreqs_full_{suffix}'))
        plot_all_spectra_mean(ls_omegas/(2*np.pi), gt_r.cpu().numpy(), pred_full_r.cpu().numpy(), fname=op.join(path, f'spectra_full_{suffix}'))
    loss_full = torch.nn.MSELoss()(gt, pred_full)
    return loss.cpu().numpy(), loss_full.cpu().numpy(), loss_mae.cpu().numpy()

def plot_samples(samples, all_target, all_evalpoint, all_observed, idx_sample, save_path=None, model_name='model', nrows=1, ncols=3, figsize=(14.0, 3.0)):
    all_target_np = all_target.cpu().numpy()
    all_evalpoint_np = all_evalpoint.cpu().numpy()
    all_observed_np = all_observed.cpu().numpy()
    all_given_np = all_observed_np - all_evalpoint_np

    K = samples.shape[-1]  # feature
    L = samples.shape[-2]  # time length
    qlist = [0.05, 0.25, 0.5, 0.75, 0.95]
    quantiles_imp = []
    for q in qlist:
        quantiles_imp.append(get_quantile(samples, q, dim=1) * (1 - all_given_np) + all_target_np * all_given_np)

    ### healthcare ###
    dataind = idx_sample  # change to visualize a different time-series sample

    plt.rcParams["font.size"] = 16
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = np.array(axes)  # Ensure `axes` is a NumPy array
    if nrows == 1:
        axes = axes[np.newaxis, :]  # Add a new axis for consistency
    if ncols == 1:
        axes = axes[:, np.newaxis]

    # Remove unused subplots
    num_plots = nrows * ncols
    if num_plots > K:
        for i in range(K, num_plots):
            fig.delaxes(axes.flatten()[i])

    # for k in range(K):
    for k in range(num_plots):
        df = pd.DataFrame({"x": np.arange(0, L), "val": all_target_np[dataind, :, k], "y": all_evalpoint_np[dataind, :, k]})
        df = df[df.y != 0]
        df2 = pd.DataFrame({"x": np.arange(0, L), "val": all_target_np[dataind, :, k], "y": all_given_np[dataind, :, k]})
        df2 = df2[df2.y != 0]
        row = k // ncols
        col = k % ncols
        axes[row, col].plot(range(0, L), quantiles_imp[2][dataind, :, k], color='g', linestyle='solid', label=model_name)
        axes[row, col].fill_between(range(0, L), quantiles_imp[0][dataind, :, k], quantiles_imp[4][dataind, :, k],
                                    color='g', alpha=0.3)
        axes[row, col].plot(df.x, df.val, color='b', marker='o', linestyle='None')
        axes[row, col].plot(df2.x, df2.val, color='r', marker='x', linestyle='None')
        if col == 0:
            axes[row, 0].set_ylabel('value')
        if row == nrows - 1:
            axes[row, col].set_xlabel('time')
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()




def plot_samples_point_pred(samples, all_target, all_evalpoint, all_observed, idx_sample, save_path=None, model_name='model', nrows=1, ncols=3, figsize=(14.0, 3.0)):
    all_target_np = all_target.cpu().numpy()
    all_evalpoint_np = all_evalpoint.cpu().numpy()
    all_observed_np = all_observed.cpu().numpy()
    all_given_np = all_observed_np - all_evalpoint_np
    samples_np = samples.cpu().numpy()

    K = samples.shape[-1]  # feature
    L = samples.shape[-2]  # time length

    ### healthcare ###
    dataind = idx_sample  # change to visualize a different time-series sample

    plt.rcParams["font.size"] = 16
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = np.array(axes)  # Ensure `axes` is a NumPy array
    if nrows == 1:
        axes = axes[np.newaxis, :]  # Add a new axis for consistency
    if ncols == 1:
        axes = axes[:, np.newaxis]

    # Remove unused subplots
    num_plots = nrows * ncols
    if num_plots > K:
        for i in range(K, num_plots):
            fig.delaxes(axes.flatten()[i])

    for k in range(num_plots):
        df = pd.DataFrame({"x": np.arange(0, L), "val": all_target_np[dataind, :, k], "y": all_evalpoint_np[dataind, :, k]})
        df = df[df.y != 0]
        df2 = pd.DataFrame({"x": np.arange(0, L), "val": all_target_np[dataind, :, k], "y": all_given_np[dataind, :, k]})
        df2 = df2[df2.y != 0]
        row = k // ncols
        col = k % ncols
        axes[row, col].plot(range(0, L), samples_np[dataind, :, k], color='g', linestyle='solid', label=model_name)
        axes[row, col].plot(df.x, df.val, color='b', marker='o', linestyle='None')
        axes[row, col].plot(df2.x, df2.val, color='r', marker='x', linestyle='None')
        if col == 0:
            axes[row, 0].set_ylabel('value')
        if row == nrows - 1:
            axes[row, col].set_xlabel('time')
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()