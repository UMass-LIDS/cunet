import os
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import matplotlib.pyplot as plt

# Frequency-Selective Mesh-to-Mesh Resampling for Color Upsampling of Point Clouds, MMSP 2021

class FSMMRInterpolation:
    def __init__(self, scale, max_iter=500):
        super(FSMMRInterpolation, self).__init__()
        self.scale = scale
        self.max_iter = max_iter

    def __call__(self, points_lr, colors_lr, points_hr, idx_lr2hr=None):
        points_lr = points_lr.astype(np.float32)
        colors_lr = colors_lr.astype(np.float32)
        points_hr = points_hr.astype(np.float32)
        points_lr = points_lr * self.scale + self.scale / 2
        points_hr = points_hr + 1 / 2
        return fsmmr_interp(points_lr, colors_lr, points_hr)

def l2_distance(points):
    N = points.shape[0]
    points = points[None, :, :].repeat(N, axis=0)
    dist = np.sum((points - points.transpose((1, 0, 2))) ** 2, axis=-1)
    return dist

def sparse_to_list(mat: csr_matrix, num_point: int):
    result = [[] for _ in range(num_point)]
    entries = mat.nonzero()
    for i in range(num_point - 1):
        j = entries[0][i]
        k = entries[1][i]
        result[j].append(k)
        result[k].append(j)
    return result


def fsmmr_interp(points_lr, colors_lr, points_hr, max_iter=1000, transform_size=8):
    # project 3D coordinates to 2D
    num_lr = points_lr.shape[0]
    num_hr = points_hr.shape[0]
    points_all = np.concatenate([points_lr, points_hr], axis=0)
    points_all_2d = project_2d(points_all)

    # generate basis functions and weights
    basis_all = generate_basis_function(points_all_2d)  # (N, trans^2, 3)
    basis_lr = basis_all[:num_lr, :, :]  # (N_lr, trans^2, 3)
    basis_hr = basis_all[num_lr:, :, :]  # (N_hr, trans^2, 3)
    weight_freq = generate_basis_weights()  # (1, trans^2, 1)
    weight_spatial = np.ones((num_lr, 1, 1))  # (N_lr, 1, 1)

    # optimize
    res = colors_lr.copy()  # (N_lr, 3)
    model = np.zeros((transform_size**2, 3))  # (trans^2, 3)
    for iter in range(max_iter):
        numerator = np.sum(res[:, None, :] * basis_lr * weight_spatial, axis=0)  # (trans^2, 3)
        denominator = np.sum(basis_lr * basis_lr * weight_spatial, axis=0)  # (trans^2, 3)
        signal = numerator / denominator  # (trans^2, 3)
        delta_energy = signal * signal * denominator  # (trans^2, 3)
        idx = np.argmax(delta_energy * weight_freq[:, :], axis=0)  # (3, )
        tmp = np.arange(3)
        best_signal = signal[idx, tmp]  # (3, )
        model[idx, tmp] += best_signal
        res = res - best_signal[None, :] * basis_lr[:, idx, tmp]

    colors_hr = np.sum(model[None, :, :] * basis_hr, axis=1)
    return np.clip(colors_hr, a_min=0, a_max=1)


def generate_basis_function(pos, transform_size=8):
    # basis function weights
    w = np.sqrt(2 / transform_size) * np.ones((transform_size, 1))
    w[0, :] = np.sqrt(1 / transform_size)
    ww = w @ w.T
    ww = ww[None, :, :]  # (1, trans, trans)

    # basis functions
    N = pos.shape[0]
    x = pos[:, 0:1].repeat(transform_size, axis=1)
    y = pos[:, 1:2].repeat(transform_size, axis=1)
    sequence = np.arange(transform_size).reshape((1, transform_size))
    x_trans = np.cos(np.pi / (2 * transform_size) * (2 * (x - 1) + 1) * sequence)
    y_trans = np.cos(np.pi / (2 * transform_size) * (2 * (y - 1) + 1) * sequence)

    x_trans = x_trans[:, :, None].repeat(transform_size, axis=2)
    y_trans = y_trans[:, None, :].repeat(transform_size, axis=1)
    basis = ww * x_trans * y_trans  # (N, trans, trans)
    basis = basis.reshape((N, transform_size * transform_size, 1))  # (N, trans*trans, 1)
    return basis.repeat(3, axis=2)  # (N, trans*trans, 3)


def generate_basis_weights(sigma=0.5, transform_size=8):
    sequence = np.arange(transform_size)
    sequence = sequence[:, None].repeat(transform_size, axis=1)
    seq2 = sequence * sequence
    weight = np.power(sigma, np.sqrt(seq2 + seq2.T))  # (trans, trans)
    return weight.reshape((transform_size * transform_size, 1))  # (trans*trans, 1)


def project_2d(points_in):
    # absorb the dimension with the least variance
    # resort the dimensions s.t. the last dimension will be absorbed
    num_point = points_in.shape[0]
    variance = np.var(points_in, axis=0)
    absorb_idx = np.argmin(variance)
    points = []
    for i in range(3):
        if i != absorb_idx:
            points.append(points_in[:, i])
    points.append(points_in[:, absorb_idx])
    points = np.stack(points, axis=1)

    # construct minimum spanning tree
    distance = l2_distance(points)
    distance = np.clip(distance, a_min=1e-6, a_max=1e6)  # avoid rounding
    tree = minimum_spanning_tree(distance)
    tree = sparse_to_list(tree, num_point)

    # breath first search
    stack = []
    for child in tree[0]:
        stack.append((0, child, np.zeros((2, ))))
    flag = [False for _ in range(num_point)]
    flag[0] = True
    result = np.zeros((num_point, 2))

    while len(stack) > 0:
        parent_idx, current_idx, parent_2d = stack.pop(0)
        if flag[current_idx]:
            continue
        px = points[parent_idx, 0]
        py = points[parent_idx, 1]
        pz = points[parent_idx, 2]
        cx = points[current_idx, 0]
        cy = points[current_idx, 1]
        cz = points[current_idx, 2]
        dx = np.sign(cx - px) * np.sqrt((cx - px) ** 2 + (cz - pz) ** 2)
        dy = np.sign(cy - py) * np.sqrt((cy - py) ** 2 + (cz - pz) ** 2)
        current_2d = np.array([dx, dy]) + parent_2d
        result[current_idx] = current_2d
        flag[current_idx] = True
        for child in tree[current_idx]:
            stack.append((current_idx, child, current_2d))

    result -= np.min(result, axis=0)
    return result


if __name__ == '__main__':
    from utils import read_point_cloud_ply, draw_point_cloud, set_random_seed, read_mesh_ply, draw_mesh, normalize
    from sampling.random_sample import random_sampling_color
    from metric.psnr import compute_psnr_numpy_rgb

    set_random_seed(0)
    points, colors = read_point_cloud_ply("../demos/Asterix.ply", require_normal=False)
    # points = normalize(points)

    # print("remove half")
    # mask = (points[:, 1] < 0)
    # points = points[mask, :]
    # colors = colors[mask, :]

    points1, points2, colors1, colors2_gt = random_sampling_color(points, colors, sample_rate=0.5)

    colors2_pred = fsmmr_interp(points1, colors1, points2)

    print("num colored points", points1.shape[0], "uncolored points", points2.shape[0])
    points_new = np.concatenate([points1, points2], axis=0)
    colors_new = np.concatenate([colors1, colors2_pred], axis=0)
    psnr = compute_psnr_numpy_rgb(colors2_pred, colors2_gt)
    print("psnr", psnr)

    # draw_point_cloud(points_new, colors_new)


