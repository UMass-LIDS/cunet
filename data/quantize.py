from itertools import repeat
from typing import List, Tuple, Union

import numpy as np


def quantize_point_cloud(points, colors, num_voxel_axis=1000):
    box_size = compute_box_size(points)
    voxel_size = box_size / num_voxel_axis
    points_vox, idx = sparse_quantize(points, voxel_size, return_index=True)
    colors_vox = colors[idx, :]
    return points_vox, colors_vox


def compute_box_size(points):
    max_pos = np.max(points, axis=0)
    min_pos = np.min(points, axis=0)
    axis_size = max_pos - min_pos
    box_size = np.max(axis_size)
    return box_size


# copied from TorchSparse
def ravel_hash(x: np.ndarray) -> np.ndarray:
    assert x.ndim == 2, x.shape

    x = x - np.min(x, axis=0)
    x = x.astype(np.uint64, copy=False)
    xmax = np.max(x, axis=0).astype(np.uint64) + 1

    h = np.zeros(x.shape[0], dtype=np.uint64)
    for k in range(x.shape[1] - 1):
        h += x[:, k]
        h *= xmax[k + 1]
    h += x[:, -1]
    return h


# copied from TorchSparse
def sparse_quantize(coords,
                    voxel_size: Union[float, Tuple[float, ...]] = 1,
                    *,
                    return_index: bool = False,
                    return_inverse: bool = False) -> List[np.ndarray]:
    if isinstance(voxel_size, (float, int)):
        voxel_size = tuple(repeat(voxel_size, 3))
    assert isinstance(voxel_size, tuple) and len(voxel_size) == 3

    voxel_size = np.array(voxel_size)
    coords = np.floor(coords / voxel_size).astype(np.int32)

    _, indices, inverse_indices = np.unique(ravel_hash(coords),
                                            return_index=True,
                                            return_inverse=True)
    coords = coords[indices]

    outputs = [coords]
    if return_index:
        outputs += [indices]
    if return_inverse:
        outputs += [inverse_indices]
    return outputs[0] if len(outputs) == 1 else outputs
