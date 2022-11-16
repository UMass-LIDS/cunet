import numpy as np
import cvxpy as cp
from traditional.knn import nearest_interp, knn_interp, knn_index


# Super-Resolution of 3D Color Point Clouds via Fast Graph Total Variation, ICASSP 2020
# implementation different from the paper to avoid Out-Of-Memory error
class FGTVInterpolation():
    def __init__(self, scale, k=3, sigma_p=1.0, sigma_c=1.0, refine=None):
        '''
        :param k: number of neighbors
        :param sigma_p: importance of geometry
        :param sigma_c: importance of color
        :param refine: refine from
        '''
        super(FGTVInterpolation, self).__init__()
        self.scale = scale
        self.k = k
        self.sigma_p = sigma_p
        self.sigma_c = sigma_c
        self.refine = refine

    def __call__(self, points_lr, colors_lr, points_hr, idx_lr2hr):
        points_lr = points_lr.astype(np.float32)
        colors_lr = colors_lr.astype(np.float32)
        points_hr = points_hr.astype(np.float32)
        points_lr = points_lr * self.scale + self.scale / 2
        points_hr = points_hr + 1 / 2
        return fgtv_interp(points_lr, colors_lr, points_hr , self.k, self.sigma_p, self.sigma_c, self.refine)

def normalize_both(points1, points2):
    pos_min = np.min(points2, axis=0)
    pos_max = np.max(points2, axis=0)
    points1 = (points1 - pos_min) / (pos_max - pos_min)
    points2 = (points2 - pos_min) / (pos_max - pos_min)
    return points1, points2


def fgtv_interp(points1, colors1, points2, k=3, sigma_p=1.0, sigma_c=1.0, refine=None):

    points1, points2 = normalize_both(points1, points2)

    num_points1 = points1.shape[0]
    num_points2 = points2.shape[0]
    num_edges = num_points2 * k
    knn_idx = knn_index(points1, points2, k=k)

    if refine == "nearest":
        colors2_init = nearest_interp(points1, points2, colors1)
    elif refine == "knn":
        colors2_init = knn_interp(points1, points2, colors1, k=k)

    weight = np.zeros((num_edges, 1), dtype=np.float32)
    G_new = np.zeros((num_edges, num_points2), dtype=np.float32)  # corresponds to new points (points2)
    G_lr = np.zeros((num_edges, num_points1), dtype=np.float32)  # corresponds to LR points (points1)
    # use for loop instead of matrix operations to save memory
    for idx_new in range(num_points2):
        for j in range(k):
            idx_edge = idx_new * k + j
            idx_lr = knn_idx[idx_new, j]
            G_new[idx_edge, idx_new] = 1
            G_lr[idx_edge, idx_lr] = -1

            diff_p = points1[idx_lr] - points2[idx_new]
            diff_p_2 = diff_p @ diff_p
            if not refine:
                weight[idx_edge] = np.exp(- diff_p_2 / sigma_p)
            else:
                diff_c = colors1[idx_lr] - colors2_init[idx_new]
                diff_c_2 = diff_c @ diff_c
                weight[idx_edge] = np.exp(- diff_p_2 / sigma_p - diff_c_2 / sigma_c)

    colors2 = _solver(G_new, G_lr, weight, colors1)
    return colors2


def _solver(G_new, G_lr, weight, color_lr):
    num_edge = G_new.shape[0]
    num_new = G_new.shape[1]

    ones = np.ones((num_edge, 1), dtype=np.float32)
    zeros = np.zeros((num_edge, 1), dtype=np.float32)
    bound = cp.Variable((num_edge, 1))
    color_new = cp.Variable((num_new, 3))
    gtv = cp.multiply(weight, G_lr @ color_lr + G_new @ color_new)

    objective = cp.Minimize(ones.T @ bound)
    constraints = [-bound <= gtv,
                   gtv <= bound,
                   bound >= zeros]

    prob = cp.Problem(objective, constraints)
    prob.solve()

    # print("status:", prob.status)
    # print("optimal value:", prob.value)
    # print("solution:", color_new.value)
    return np.clip(color_new.value, a_min=0, a_max=1)



if __name__ == '__main__':
    from utils import read_point_cloud_ply, draw_point_cloud, set_random_seed, read_mesh_ply, draw_mesh, normalize
    from sampling.random_sample import random_sampling_color
    from metric.psnr import compute_psnr_numpy_rgb

    set_random_seed(0)
    points, colors = read_point_cloud_ply("../demos/man_norm.ply", require_normal=False)
    points = normalize(points)
    points1, points2, colors1, colors2_gt = random_sampling_color(points, colors, sample_rate=0.5)

    # colors2_pred = nearest_interp(points1, points2, colors1)  # 29.28
    # colors2_pred = knn_interp(points1, points2, colors1)  # 31.03


    colors2_pred = fgtv_interp(points1, colors1, points2)


    print("num colored points", points1.shape[0], "uncolored points", points2.shape[0])
    points_new = np.concatenate([points1, points2], axis=0)
    colors_new = np.concatenate([colors1, colors2_pred], axis=0)
    psnr = compute_psnr_numpy_rgb(colors2_pred, colors2_gt)
    print("psnr", psnr)

    # draw_point_cloud(points_new, colors_new)