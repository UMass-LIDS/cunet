import numpy as np
from sklearn.neighbors import NearestNeighbors

# Fractional Super-Resolution of Voxelized Point Clouds
def knn(source, target, k):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(source)
    distances, indices = nbrs.kneighbors(target)
    return distances, indices


class FractionalSRInterpolation:
    def __init__(self, scale):
        super(FractionalSRInterpolation, self).__init__()
        self.k = 7
        self.scale = scale
        self.threshold = np.sqrt((scale + 0.5) ** 2 + 0.5)

    def __call__(self, points_lr, colors_lr, points_hr, idx_lr2hr=None):
        points_lr = points_lr * self.scale + self.scale / 2
        points_hr = points_hr + 1 / 2
        knn_dist, knn_idx = knn(points_lr, points_hr, self.k)

        colors_knn = colors_lr[knn_idx]  # (N, 7, 3)
        colors_parent = colors_knn[:, 0, :]  # (N, 3)
        colors_uncles = colors_knn[:, 1:, :]  # (N, 6, 3)
        dist_uncles = knn_dist[:, 1:]  # (N, 6)

        # remove uncle nodes not sharing an edge with the children node
        mask = (dist_uncles < self.threshold)  # (N, 6)
        weight = mask * 1 / dist_uncles  # (N, 6)

        eta = dist_uncles[:, :1] * self.scale / 8  # (N, 1)
        colors_uncles_sum = np.sum(eta[:, :, None] * weight[:, :, None] * colors_uncles, axis=1)  # (N, 3)

        colors_hr = (colors_parent + colors_uncles_sum) / (1 + np.sum(eta * weight, axis=1, keepdims=True))
        return colors_hr



