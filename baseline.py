import numpy as np
from data.face_scape_voxel_dataset import FaceScapeVoxelDataset
from utils import set_random_seed
from metric.psnr import compute_psnr_numpy_rgb
from sampling.voxelize import voxelize_avg_numpy, Voxelizer
from tqdm import tqdm
from traditional.devoxelize import devoxelize_interp
from traditional.knn import KNNVoxelInterpolation
from deep_interp import DeepInterpolation
from torch.utils.data import DataLoader
from traditional.fractional import FractionalSRInterpolation
import argparse

def run_baseline(dataset, interp, scale):
    total_psnr = 0
    for sparse_hr in tqdm(dataset):
        points_hr = sparse_hr.C
        colors_hr = sparse_hr.F
        points_lr, colors_lr, idx_lr2hr = voxelize_avg_numpy(sparse_hr.C, sparse_hr.F, scale)
        colors_hr_pred = interp(points_lr, colors_lr, points_hr, idx_lr2hr)
        psnr = compute_psnr_numpy_rgb(colors_hr_pred, colors_hr)
        total_psnr += psnr
    total_psnr /= len(dataset)
    print("Average PSNR:", total_psnr)
    return total_psnr


def knn_face_scape(scale, gt_scale, k=3):
    global face_scape_path
    interp = KNNVoxelInterpolation(scale, k=k)
    dataset = FaceScapeVoxelDataset(face_scape_path, "test", gt_scale=gt_scale, num_patch=1)
    run_baseline(dataset, interp, scale)


def devox_face_scape(scale, gt_scale):
    global face_scape_path
    interp = devoxelize_interp
    dataset = FaceScapeVoxelDataset(face_scape_path, "test", gt_scale=gt_scale, num_patch=1)
    run_baseline(dataset, interp, scale)


def frac_face_scape(scale, gt_scale):
    global face_scape_path
    interp = FractionalSRInterpolation(scale)
    dataset = FaceScapeVoxelDataset(face_scape_path, "test", gt_scale=gt_scale, num_patch=1)
    run_baseline(dataset, interp, scale)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="/work/lingdongwang_umass_edu/Datasets/FaceScape/")
    args = parser.parse_args()

    face_scape_path = args.data_dir
    set_random_seed(0)

    for gt_scale in [5, 2, 1]:
        scale = 10 // gt_scale

        print("Devoxelize")
        devox_face_scape(scale, gt_scale)

        print("KNN")
        knn_face_scape(scale, gt_scale)

        print("Fractional")
        frac_face_scape(scale, gt_scale)