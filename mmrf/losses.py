"""Losses and evaluation metrics. Fields u are displacements in voxels, (B, 3, D, H, W)."""
import math

import torch
import torch.nn.functional as F


def lncc_loss(pred, target, win=9):
    """negative local NCC (squared correlation form, 9^3 box window): the teacher's image term"""
    n = win ** 3; p = win // 2; filt = torch.ones(1, 1, win, win, win, device=pred.device)
    s = lambda x: F.conv3d(x, filt, padding=p)
    I, J = target, pred
    I_sum, J_sum = s(I), s(J); u_I, u_J = I_sum / n, J_sum / n
    cross = s(I * J) - u_J * I_sum - u_I * J_sum + u_I * u_J * n
    I_var = s(I * I) - 2 * u_I * I_sum + u_I * u_I * n
    J_var = s(J * J) - 2 * u_J * J_sum + u_J * u_J * n
    return -torch.mean(cross * cross / (I_var * J_var + 1e-5))


def lncc2d(a, b, win=9):
    """local squared NCC of planes a, b (B, 1, H, W), mean per sample (B,): the refinement's image term"""
    f = torch.ones(1, 1, win, win, device=a.device); n = win * win; p = win // 2
    s = lambda x: F.conv2d(x, f, padding=p)
    ua, ub = s(a) / n, s(b) / n; cr = s(a * b) - ub * s(a) - ua * s(b) + ua * ub * n
    va = s(a * a) - 2 * ua * s(a) + ua ** 2 * n; vb = s(b * b) - 2 * ub * s(b) + ub ** 2 * n
    return (cr ** 2 / (va * vb + 1e-5)).mean((1, 2, 3))


def smooth_loss(u):
    """mean squared forward differences of the field"""
    dz = u[:, :, 1:] - u[:, :, :-1]; dy = u[:, :, :, 1:] - u[:, :, :, :-1]; dx = u[:, :, :, :, 1:] - u[:, :, :, :, :-1]
    return (dz.pow(2).mean() + dy.pow(2).mean() + dx.pow(2).mean()) / 3


def jac_det(u):
    """det(I + grad u) (B, D, H, W), central differences"""
    g = [torch.gradient(u[:, i], dim=(1, 2, 3)) for i in range(3)]
    J = [[g[i][j] + (1.0 if i == j else 0.0) for j in range(3)] for i in range(3)]
    return J[0][0] * (J[1][1] * J[2][2] - J[1][2] * J[2][1]) - J[0][1] * (J[1][0] * J[2][2] - J[1][2] * J[2][0]) + J[0][2] * (J[1][0] * J[2][1] - J[1][1] * J[2][0])


def fold_penalty(u, margin=0.05):
    """mean relu(margin - det(I + grad u)): the folding penalty, matched to the folding metric"""
    return torch.relu(margin - jac_det(u)).mean()


@torch.no_grad()
def fold_pct(u, mask=None):
    """% voxels with det(I + grad u) <= 0 (within mask (D, H, W) if given)"""
    d = jac_det(u.float())[0]
    return float(((d[mask] if mask is not None else d) <= 0).float().mean() * 100)


# ---- evaluation metrics: each frame t compares V_0 warped by the field with V_t ----
def lncc_metric(pred, target, win=9):
    return float(-lncc_loss(pred, target, win))


def ssim_metric(x, y, win=7):
    """3D SSIM, 7^3 Gaussian window (sigma 1.5), zero padding"""
    c = torch.arange(win, dtype=torch.float32, device=x.device) - win // 2
    g = torch.exp(-c ** 2 / 4.5); g /= g.sum()
    k = (g[:, None, None] * g[None, :, None] * g[None, None, :])[None, None]; p = win // 2; C1, C2 = 1e-4, 9e-4
    m1, m2 = F.conv3d(x, k, padding=p), F.conv3d(y, k, padding=p)
    s1 = F.conv3d(x * x, k, padding=p) - m1 ** 2; s2 = F.conv3d(y * y, k, padding=p) - m2 ** 2; s12 = F.conv3d(x * y, k, padding=p) - m1 * m2
    return float(((2 * m1 * m2 + C1) * (2 * s12 + C2) / ((m1 ** 2 + m2 ** 2 + C1) * (s1 + s2 + C2) + 1e-8)).mean())


@torch.no_grad()
def metrics(warped, target):
    """lNCC, SSIM (both volumes rescaled with the target's range), PSNR (dB, volumes in [0, 1]), MSE"""
    mn, mx = target.min(), target.max(); r = max(float(mx - mn), 1e-8); mse = float(F.mse_loss(warped, target))
    ssim = ssim_metric(((warped - mn) / r).clamp(0, 1), ((target - mn) / r).clamp(0, 1))
    return {"lncc": lncc_metric(warped, target), "ssim": ssim, "psnr": 10 * math.log10(1 / mse) if mse > 1e-10 else float("inf"), "mse": mse}
