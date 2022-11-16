import numpy
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
import torchsparse.nn.functional as F
from tqdm import tqdm
from sklearn.cluster import KMeans, MiniBatchKMeans
from data.quantize import sparse_quantize

def knn_index(source, target, k=1):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(source)
    distances, indices = nbrs.kneighbors(target)
    return indices


def farthest_point_sample_numpy(points, num_centroid):
    N = points.shape[0]
    centroid_idx = np.zeros((num_centroid,), dtype=int)
    distance = np.ones((N, )) * 1e10
    farthest = np.random.randint(N)
    for i in range(num_centroid):
        centroid_idx[i] = farthest
        centroid = points[farthest:farthest+1, :]
        dist = np.sum((points - centroid) ** 2, -1)
        mask = (dist < distance)
        distance[mask] = dist[mask]
        farthest = np.argmax(distance)
    return centroid_idx


def farthest_point_sample_pytorch(points, num_centroid):
    N = points.shape[0]
    centroid_idx = torch.zeros((num_centroid,), dtype=torch.long)
    distance = torch.ones((N, )) * 1e10
    farthest = torch.randint(0, N, (1,), dtype=torch.long)
    for i in range(num_centroid):
        centroid_idx[i] = farthest
        centroid = points[farthest:farthest+1, :]
        dist = torch.sum((points - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroid_idx


class PatchInterpolation():
    def __init__(self, interp, scale, mode='average', num_point=4096, over_sample=3, block_size=8, verbose=False):
        super(PatchInterpolation, self).__init__()
        self.voxel_size = scale
        self.mode = mode
        self.num_point = num_point
        self.over_sample = over_sample
        self.block_size = block_size
        self.interp = interp
        self.verbose = verbose
        print("patch aggregation mode:", mode)


    def __call__(self, points_lr, colors_lr, points_hr, idx_lr2hr=None):
        if idx_lr2hr is None:
            is_numpy = False
            if not torch.is_tensor(points_hr):
                is_numpy = True
            if is_numpy:
                points_hr = torch.from_numpy(points_hr)
            points_hr = torch.cat([points_hr, torch.zeros((points_hr.shape[0], 1), dtype=torch.int32, device=points_hr.device)], dim=1)
            quant_hr = torch.cat([points_hr[:, :3] / self.voxel_size, points_hr[:, -1].view(-1, 1)], 1)
            quant_hr = torch.floor(quant_hr).int()
            hash_hr = F.sphash(quant_hr)
            _, idx_lr2hr = torch.unique(hash_hr, return_inverse=True)  # (N_hr)
            points_hr = points_hr[:, :3]
            if is_numpy:
                points_hr = points_hr.numpy()
                idx_lr2hr = idx_lr2hr.numpy()

        if self.mode == "average":
            return self._run_average(points_lr, colors_lr, points_hr, idx_lr2hr)
        elif self.mode == 'split':
            return self._run_split(points_lr, colors_lr, points_hr, idx_lr2hr)
        elif self.mode == 'block':
            return self._run_block(points_lr, colors_lr, points_hr, idx_lr2hr)
        else:
            raise NotImplementedError

    def _run_block(self, points_lr, colors_lr, points_hr, idx_lr2hr):
        colors_hr = np.zeros(points_hr.shape)
        _, block_idx = sparse_quantize(points_hr, self.block_size, return_index=False, return_inverse=True)
        num_block = np.max(block_idx)
        enumerator = tqdm(range(num_block)) if self.verbose else range(num_block)
        for i in enumerator:
            patch_idx = (block_idx == i)
            points_patch_hr = points_hr[patch_idx, :]
            patch_lr_idx = idx_lr2hr[patch_idx]
            patch_lr_idx, idx_lr2hr_patch = np.unique(patch_lr_idx, return_index=False, return_inverse=True)
            points_patch_lr = points_lr[patch_lr_idx, :]
            colors_patch_lr = colors_lr[patch_lr_idx, :]
            colors_patch_hr = self.interp(points_patch_lr, colors_patch_lr, points_patch_hr, idx_lr2hr_patch)
            colors_hr[patch_idx, :] = colors_patch_hr
        return colors_hr


    def _run_average(self, points_lr, colors_lr, points_hr, idx_lr2hr):
        num_centroid = int(points_hr.shape[0] / self.num_point) * self.over_sample
        patch_idx = self._crop_patch_average(points_hr, num_centroid)

        colors_hr_list = []
        enumerator = tqdm(range(num_centroid)) if self.verbose else range(num_centroid)
        for i in enumerator:
            points_patch_hr = points_hr[patch_idx[i, :], :]
            patch_lr_idx = idx_lr2hr[patch_idx[i, :]]
            patch_lr_idx, idx_lr2hr_patch = np.unique(patch_lr_idx, return_index=False, return_inverse=True)

            points_patch_lr = points_lr[patch_lr_idx, :]
            colors_patch_lr = colors_lr[patch_lr_idx, :]

            colors_patch_hr = self.interp(points_patch_lr, colors_patch_lr, points_patch_hr, idx_lr2hr_patch)
            colors_hr_list.append(colors_patch_hr)

        colors_hr = self._merge_patch_average(colors_hr_list, patch_idx, num_centroid, points_hr.shape[0])
        return colors_hr

    def _crop_patch_average(self, points, num_centroid):
            # sample centroids using farthest point sampling
            centroid_idx = farthest_point_sample_numpy(points, num_centroid)
            # collect k-nearest points around a centroid
            patch_idx = knn_index(points, points[centroid_idx, :], k=self.num_point)
            return patch_idx

    def _merge_patch_average(self, colors_hr_list, patch_idx, num_centroid, num_color):
        cnt = np.zeros((num_color, ))
        colors_hr = np.zeros((num_color, 3))
        for i in range(num_centroid):
            pidx = patch_idx[i, :]
            cnt[pidx] += 1
            colors_hr[pidx, :] += colors_hr_list[i]
        mask = (cnt > 0)
        colors_hr[mask, :] = colors_hr[mask, :] / cnt[mask, None]
        return colors_hr

    def _run_split(self, points_lr, colors_lr, points_hr, idx_lr2hr):
        patch_idx = self._crop_patch_split(points_hr)

        colors_hr_list = []
        enumerator = tqdm(range(len(patch_idx))) if self.verbose else range(len(patch_idx))
        for i in enumerator:
            points_patch_hr = points_hr[patch_idx[i], :]
            patch_lr_idx = idx_lr2hr[patch_idx[i]]
            patch_lr_idx, idx_lr2hr_patch = np.unique(patch_lr_idx, return_index=False, return_inverse=True)

            points_patch_lr = points_lr[patch_lr_idx, :]
            colors_patch_lr = colors_lr[patch_lr_idx, :]

            colors_patch_hr = self.interp(points_patch_lr, colors_patch_lr, points_patch_hr, idx_lr2hr_patch)
            colors_hr_list.append(colors_patch_hr)

        colors_hr = self._merge_patch_split(colors_hr_list, patch_idx, points_hr.shape[0])
        return colors_hr


    def _crop_patch_split(self, points):
        # recursively split points into 2 groups, until the group size <= num_point
        kmeans = MiniBatchKMeans(n_clusters=2, n_init=1)
        points_idx = np.arange(points.shape[0])
        patch_idx = []
        candidates = [[points, points_idx]]
        while len(candidates) > 0:
            candidate, candidate_idx = candidates.pop(0)
            cluster_idx = kmeans.fit_predict(candidate)
            num_cluster = np.max(cluster_idx) + 1
            for i in range(num_cluster):
                mask = (cluster_idx == i)
                patch = candidate[mask, :]
                pidx = candidate_idx[mask]
                if patch.shape[0] <= self.num_point:
                    patch_idx.append(pidx)
                else:
                    candidates.append([patch, pidx])
        return patch_idx

    def _merge_patch_split(self, colors_hr_list, patch_idx, num_color):
        colors_hr = np.zeros((num_color, 3))
        for i in range(len(patch_idx)):
            pidx = patch_idx[i]
            colors_hr[pidx, :] = colors_hr_list[i]
        return colors_hr







if __name__ == '__main__':
    from mpeg8i import read_voxel
    from traditional.knn import KNNVoxelInterpolation
    from metric.psnr import compute_psnr_numpy_rgb
    from utils import draw_point_cloud
    from traditional.fgtv import FGTVInterpolation
    from traditional.fsmmr import FSMMRInterpolation

    gt_scale=4
    scale = 8
    block_size = 8

    data_dir = "/work/lingdongwang_umass_edu/Datasets/MPEG8i/"
    name = "longdress"
    file_name = data_dir + name + "_viewdep_vox12.ply"

    # file_name = '/work/lingdongwang_umass_edu/Datasets/FaceScape/vox/1/10_dimpler.ply'

    data = read_voxel(file_name, scale=scale, gt_scale=gt_scale, gt_save_path='', avg_color=True)
    points_hr, colors_hr, points_lr, colors_lr, idx_hr2lr, idx_lr2hr = data
    # patch_interp = KNNVoxelInterpolation(k=3, scale=scale)
    # patch_interp = FGTVInterpolation(scale=scale, sigma_p=1.0, sigma_c=1.0)
    patch_interp = FSMMRInterpolation(scale=scale, max_iter=500)
    interp = PatchInterpolation(interp=patch_interp, scale=scale, mode='block', block_size=block_size, verbose=True)
    colors_hr_pred = interp(points_lr, colors_lr, points_hr, idx_lr2hr)

    psnr = compute_psnr_numpy_rgb(colors_hr_pred, colors_hr)
    print("psnr", psnr)
    draw_point_cloud(points_hr, colors_hr_pred, visualize=False, save_path='test_patch.ply')