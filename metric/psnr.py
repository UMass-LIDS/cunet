import numpy as np
import torch


def compute_psnr_numpy_rgb(pred, gt):
    '''
    :param pred: (N, 3)
    :param gt: (N, 3)
    :return: float
    '''
    pred = pred.astype(float) * 255. + 1e-6  # for numerical stability
    gt = gt.astype(float) * 255.
    mse = np.mean((pred - gt) ** 2)
    psnr = 10 * np.log10(255.0 ** 2 / mse)
    return float(psnr)


def compute_psnr_torch_rgb(pred, gt):
    '''
    :param pred: (1, N, 3)
    :param gt: (1, N, 3)
    :return: float
    '''
    pred = pred.float() * 255. + 1e-6  # for numerical stability
    gt = gt.float() * 255.
    mse = torch.mean((pred - gt) ** 2)
    psnr = 10 * torch.log10(255.0 ** 2 / mse)
    return float(psnr)


