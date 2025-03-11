import torch
import math
import numpy as np

def LS_omegas(t, samples_per_peak=1):
    dt_min = np.min(np.diff(t))
    omega_max = np.pi / dt_min
    # Nomegas_nyq = int(t.max() / (2 * dt_min))
    Nomegas_nyq = len(t) #not sure if this works for irregular timesteps.
    ls_omegas = np.linspace(1e-5, omega_max, samples_per_peak * Nomegas_nyq)
    return ls_omegas

class LombScargleBatchMask(torch.nn.Module):
    def __init__(self, omegas):
        super(LombScargleBatchMask, self).__init__()
        self.omegas = omegas  # Tensor of angular frequencies (ω)

    def compute_fap_weights(self, ps, eps=1e-5):
        M = ps.shape[-1]  # Number of independent frequencies
        #ps = np.clip(ps, 0, 10)
        fap = 1 - (1 - torch.exp(-ps))**M # FAP for each frequency
        #fap = np.clip(fap, eps, None)  # Avoid near-zero FAP
        weights = 1.0 / (fap + eps)  # Avoid division by zero
        return weights

    def forward(self, t, y, mask=None, fap=False, norm=True):
        """
        Forward pass of the Lomb-Scargle periodogram with batch support.

        Args:
            t (Tensor): Time values, shape [B, N].
            y (Tensor): Observed data values, shape [B, N].
            mask (Tensor): Optional boolean mask, shape [B, N] (1 = valid, 0 = invalid).

        Returns:
            Tensor: Lomb-Scargle periodogram values, shape [B, M].
        """
        B, N = t.shape
        M = self.omegas.shape[0]

        if mask is None:
            mask = torch.ones_like(t)  # Default to all valid

        # Reshape tensors
        t = t.unsqueeze(1)  # [B, 1, N]
        y = y.unsqueeze(1)  # [B, 1, N]
        mask = mask.unsqueeze(1)  # [B, 1, N]
        omega = self.omegas.unsqueeze(0).unsqueeze(2)  # [1, M, 1]

        # Compute tau for each frequency and batch
        two_omega_t = 2 * omega * t  # [B, M, N]
        sin_2wt = torch.sin(two_omega_t) * mask
        cos_2wt = torch.cos(two_omega_t) * mask

        tan_2omega_tau = sin_2wt.sum(dim=2) / (cos_2wt.sum(dim=2) + 1e-10)
        tau = torch.atan(tan_2omega_tau) / (2 * omega.squeeze())

        # Compute Lomb-Scargle periodogram
        omega_t_tau = omega * (t - tau.unsqueeze(2))  # [B, M, N]
        cos_omega_t_tau = torch.cos(omega_t_tau) * mask
        sin_omega_t_tau = torch.sin(omega_t_tau) * mask

        y_cos = y * cos_omega_t_tau
        y_sin = y * sin_omega_t_tau

        P_cos = (y_cos.sum(dim=2) ** 2) / (cos_omega_t_tau.pow(2).sum(dim=2) + 1e-10)
        P_sin = (y_sin.sum(dim=2) ** 2) / (sin_omega_t_tau.pow(2).sum(dim=2) + 1e-10)

        P = 0.5 * (P_cos + P_sin)  # [B, M]

        if fap:
            weights = self.compute_fap_weights(P)
            P = P * weights
        
        if norm: # int[P(f)df] = 1
            #dfreq = torch.diff(torch.tensor(self.omegas/(2*math.pi)))
            dfreq = torch.diff(self.omegas/(2*math.pi))
            avg_heights = (P[:, :-1] + P[:, 1:]) / 2.0
            integral = torch.sum(avg_heights * dfreq, axis=-1) + 1e-10
            P = P / integral[:, None]

        return P
