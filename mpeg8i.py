import numpy as np
import torch.cuda
from utils import read_point_cloud_ply, draw_point_cloud, set_random_seed, read_mesh_ply, draw_mesh, normalize
from traditional.knn import KNNVoxelInterpolation
from metric.psnr import compute_psnr_numpy_rgb
from data.quantize import sparse_quantize
from sampling.voxelize import voxelize_avg_numpy
from traditional.devoxelize import devoxelize_interp
from deep_interp import DeepInterpolation
from traditional.fractional import FractionalSRInterpolation
from traditional.fgtv import FGTVInterpolation
from traditional.fsmmr import FSMMRInterpolation
from patch import PatchInterpolation
from time import time
import argparse


def read_voxel(file_name, scale, gt_scale, gt_save_path='', avg_color=False):
    points_hr, colors_hr = read_point_cloud_ply(file_name)
    points_hr = points_hr - np.min(points_hr, axis=0)
    points_hr = points_hr.astype(np.int32)
    if gt_scale > 1:
        points_hr, idx = sparse_quantize(points_hr, gt_scale, return_index=True, return_inverse=False)
        colors_hr = colors_hr[idx, :]
    if gt_save_path:
        draw_point_cloud(points_hr, colors_hr, visualize=False, save_path=gt_save_path)
    if avg_color:
        points_lr, colors_lr, idx_lr2hr = voxelize_avg_numpy(points_hr, colors_hr, scale)
        print("number of LR points:", points_lr.shape[0])
        print("number of HR points:", points_hr.shape[0])
        return points_hr, colors_hr, points_lr, colors_lr, None, idx_lr2hr
    else:
        points_lr, idx_hr2lr, idx_lr2hr = sparse_quantize(points_hr, scale, return_index=True, return_inverse=True)
        colors_lr = colors_hr[idx_hr2lr]
        colors_hr = colors_hr.astype(np.float32)
        print("number of LR points:", points_lr.shape[0])
        print("number of HR points:", points_hr.shape[0])
        return points_hr, colors_hr, points_lr, colors_lr, idx_hr2lr, idx_lr2hr


def demo_point_cloud(data, interp, save_path=''):
    points_hr, colors_hr, points_lr, colors_lr, idx_hr2lr, idx_lr2hr = data
    # draw_point_cloud(points_hr, colors_hr, visualize=False)
    start_time = time()
    colors_hr_pred = interp(points_lr, colors_lr, points_hr, idx_lr2hr)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time()
    print("latency", end_time - start_time)
    psnr = compute_psnr_numpy_rgb(colors_hr_pred, colors_hr)
    print("psnr", psnr)
    if save_path != '':
        draw_point_cloud(points_hr, colors_hr_pred, visualize=False, save_path=save_path)
    return


def draw_lr(data, scale, name):
    points_hr, colors_hr, points_lr, colors_lr, idx_hr2lr, idx_lr2hr = data
    points_lr = points_lr * scale
    # points_lr = points_lr + scale / 2
    draw_point_cloud(points_lr, colors_lr, visualize=False,
                     save_path="demos/{}_vox_{}_lr.ply".format(name, str(scale) + "x"))
    return


def evaluate(data_dir, file_names, scale, gt_scale=1, save_demo=False, fgtv=False, fsmmr=False):
    for name in file_names:
        print(name)

        file_name = data_dir + name + "_viewdep_vox12.ply"
        gt_save_path = "demos/{}_vox_{}_gt.ply".format(name, str(scale) + "x") if save_demo else ''
        data = read_voxel(file_name, scale=scale, gt_scale=gt_scale, gt_save_path=gt_save_path, avg_color=True)

        if save_demo:
            draw_lr(data, scale, name)

        print("knn")
        interp = KNNVoxelInterpolation(scale=scale)
        save_path = "demos/{}_vox_{}_knn.ply".format(name, str(scale) + "x") if save_demo else ''
        demo_point_cloud(data, interp, save_path)

        print("frac")
        interp = FractionalSRInterpolation(scale=scale)
        save_path = "demos/{}_vox_{}_frac.ply".format(name, str(scale) + "x") if save_demo else ''
        demo_point_cloud(data, interp, save_path)

        print("devox")
        interp = devoxelize_interp
        save_path = "demos/{}_vox_{}_devox.ply".format(name, str(scale)+"x") if save_demo else ''
        demo_point_cloud(data, interp, save_path)

        print("deep")
        model_path = "pretrained/face_vox_10x_b4c64_epoch_25.pth".format(str(scale) + "x")
        save_path = "demos/{}_vox_{}_deep.ply".format(name, str(scale)+"x") if save_demo else ''
        interp = DeepInterpolation(64, 4, scale, model_path)
        demo_point_cloud(data, interp, save_path)

        if fgtv:
            print("fgtv average")
            save_path = "demos/{}_vox_{}_fgtv_avg.ply".format(name, str(scale)+"x") if save_demo else ''
            patch_interp = FGTVInterpolation(scale=scale, sigma_p=1.0, sigma_c=1.0)
            interp = PatchInterpolation(interp=patch_interp, scale=scale, mode='average')
            demo_point_cloud(data, interp, save_path)

        if fsmmr:
            print("fsmmr block")
            save_path = "demos/{}_vox_{}_fsmmr_block.ply".format(name, str(scale) + "x") if save_demo else ''
            patch_interp = FSMMRInterpolation(scale=scale)
            interp = PatchInterpolation(interp=patch_interp, scale=scale, mode='block', block_size=args.block_size)
            demo_point_cloud(data, interp, save_path)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="/work/lingdongwang_umass_edu/Datasets/MPEG8i/")
    parser.add_argument('--gt_scale', type=int, default=4)
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--block_size', type=int, default=8)
    parser.add_argument('--fgtv', action='store_true', default=False)
    parser.add_argument('--fsmmr', action='store_true', default=False)
    args = parser.parse_args()

    print("gt scale", args.gt_scale, "scale", args.scale)

    if not torch.cuda.is_available():
        raise Exception("Please run with GPU.")
    set_random_seed(seed=0)

    # file_names = ["boxer", "longdress", "loot", "redandblack", "soldier", "Thaidancer"]

    file_names = ["longdress"]
    save_demo = True
    evaluate(args.data_dir, file_names, args.scale, save_demo=save_demo, gt_scale=args.gt_scale, fgtv=args.fgtv, fsmmr=args.fsmmr)

    file_names = ["boxer", "loot", "redandblack", "soldier", "Thaidancer"]
    save_demo = False
    evaluate(args.data_dir, file_names, args.scale, save_demo=save_demo, gt_scale=args.gt_scale, fgtv=args.fgtv, fsmmr=args.fsmmr)