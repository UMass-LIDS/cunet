import numpy as np
import random
import torch


# randomly drop colors
# torch
class RandomColorSampler:
    def __init__(self, sample_rate):
        super(RandomColorSampler, self).__init__()
        self.sample_rate = sample_rate

    def __call__(self, points, colors):
        '''
        :param points: (B, N, 3)
        :param colors: (B, N, 3)
        :return: (B, N1, 3), (B, N2, 3), (B, N1, 3), (B, N2, 3)
        '''
        device = points.device
        num_colors = colors.shape[1]
        idx = np.arange(num_colors, dtype=int)
        np.random.shuffle(idx)
        idx = torch.from_numpy(idx).long().to(device)

        num_output = int(num_colors * self.sample_rate)
        idx1 = idx[:num_output]
        idx2 = idx[num_output:]
        points1 = points[:, idx1, :].float()
        points2 = points[:, idx2, :].float()
        colors1 = colors[:, idx1, :].float()
        colors2 = colors[:, idx2, :].float()
        return points1, points2, colors1, colors2


# numpy
def random_sampling_color(points, colors, sample_rate, require_idx=False):
    num_colors = colors.shape[0]
    idx = np.arange(num_colors, dtype=int)
    np.random.shuffle(idx)

    num_output = int(num_colors * sample_rate)
    idx1 = idx[:num_output]
    idx2 = idx[num_output:]
    points1 = points[idx1, :]
    points2 = points[idx2, :]
    colors1 = colors[idx1, :]
    colors2 = colors[idx2, :]
    if not require_idx:
        return points1, points2, colors1, colors2
    else:
        return points1, points2, colors1, colors2, idx1, idx2


# randomly drop points
def random_sampling_point(points, colors, sample_rate):
    num_colors = colors.shape[0]
    idx = np.arange(num_colors, dtype=int)
    np.random.shuffle(idx)
    num_output = int(num_colors * sample_rate)
    idx = idx[:num_output]
    return points[idx, :], colors[idx, :]


