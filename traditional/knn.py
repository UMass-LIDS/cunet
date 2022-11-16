import numpy as np
from sklearn.neighbors import NearestNeighbors
# from knn_cuda import KNN
import torch


def knn_index(source, target, k=1):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(source)
    distances, indices = nbrs.kneighbors(target)
    return indices

# nearest neighbor interpolation of colors
def nearest_interp(points1, points2, colors1):
    knn_idx = knn_index(points1, points2, k=1)
    colors2 = colors1[knn_idx].squeeze()
    return colors2


# k-nearest neighbors interpolation of colors
class KNNInterpolation:
    def __init__(self, k=3):
        super(KNNInterpolation, self).__init__()
        self.k = k

    def __call__(self, points1, points2, colors1):
        return knn_interp(points1, points2, colors1, k=self.k)


def knn_interp(points1, points2, colors1, k=3):
    knn_idx = knn_index(points1, points2, k=k)
    colors2 = colors1[knn_idx]
    colors2 = np.mean(colors2, axis=1)
    return colors2


class KNNVoxelInterpolation:
    def __init__(self, scale, k=3):
        super(KNNVoxelInterpolation, self).__init__()
        self.k = k
        self.scale = scale

    def __call__(self, points_lr, colors_lr, points_hr, idx_lr2hr):
        points_lr = points_lr * self.scale + self.scale / 2
        points_hr = points_hr + 1 / 2
        colors_hr = knn_interp(points_lr, points_hr, colors_lr, k=self.k)
        return colors_hr


