import torch
from data.face_scape_voxel_dataset import FaceScapeVoxelDataset
from sampling.voxelize import Voxelizer
from torchsparse.utils.collate import sparse_collate
from time import time
from traditional.knn import KNNVoxelInterpolation
from traditional.fractional import FractionalSRInterpolation
import numpy as np
from traditional.fgtv import FGTVInterpolation
from traditional.fsmmr import FSMMRInterpolation
from patch import PatchInterpolation


def measure_latency(interp, gt_scale, scale, num_iter=65, warm_up=5):
    data_path = "/work/lingdongwang_umass_edu/Datasets/FaceScape/"
    dataset = FaceScapeVoxelDataset(data_path, "test", gt_scale=gt_scale)
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         collate_fn=sparse_collate,
                                         batch_size=1,
                                         shuffle=False,
                                         pin_memory=True)

    voxelizer = Voxelizer(scale)

    # data preparation
    print("preparing data")
    inputs = []
    cnt = 0
    for sparse_hr in loader:
        sparse_hr = sparse_hr.cuda()
        sparse_lr, idx_hr2lr, idx_lr2hr = voxelizer(sparse_hr)
        lr_points = sparse_lr.C.cpu()[:, :3].numpy().astype(int)
        lr_colors = sparse_lr.F.cpu().numpy().astype(np.float32)
        hr_colors = sparse_hr.C.cpu()[:, :3].numpy().astype(int)

        inputs.append([lr_points, lr_colors, hr_colors])
        cnt += 1
        if cnt == num_iter:
            break

    # model inference
    print("start inference")
    # torch.cuda.synchronize()
    start_time = time()
    for i in range(len(inputs)):
        if i == warm_up:
            start_time = time()
        points_lr, colors_lr, points_hr = inputs[i]
        colors_hr_pred = interp(points_lr, colors_lr, points_hr, None)

    # torch.cuda.synchronize()
    end_time = time()

    avg_time = (end_time - start_time) / (num_iter - warm_up)
    print("avg time", avg_time)
    return


if __name__ == '__main__':
    for gt_scale in [5, 2, 1]:
        scale = 10 // gt_scale
        print("gt scale", gt_scale, "scale", scale)
        print("knn")
        interp = KNNVoxelInterpolation(scale)
        measure_latency(interp, gt_scale=gt_scale, scale=scale)
        print("frac")
        interp = FractionalSRInterpolation(scale)
        measure_latency(interp, gt_scale=gt_scale, scale=scale)
        print("fgtv")
        patch_interp = FGTVInterpolation(scale=scale, sigma_p=1.0, sigma_c=1.0)
        interp = PatchInterpolation(interp=patch_interp, scale=scale, mode='average')
        measure_latency(interp, gt_scale=gt_scale, scale=scale, num_iter=1, warm_up=0)
        print("fsmmr")
        patch_interp = FSMMRInterpolation(scale=scale)
        interp = PatchInterpolation(interp=patch_interp, scale=scale, mode='block', block_size=8)
        measure_latency(interp, gt_scale=gt_scale, scale=scale, num_iter=1, warm_up=0)



