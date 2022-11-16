import argparse
import os
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import time
import sys
from utils import Logger, display_config, set_random_seed
import numpy as np
from tqdm import tqdm
from torchsparse.utils.collate import sparse_collate
from data.face_scape_voxel_dataset import FaceScapeVoxelDataset
from sampling.voxelize import Voxelizer
from runner.voxel import run_voxel
from models.cunet import CUNet


parser = argparse.ArgumentParser()
# basic setting
parser.add_argument('--log_name', type=str, default='cunet_train', help="log name")
parser.add_argument('--dataset', type=str ,default='face_scape_vox', help="dataset name")
parser.add_argument('--data_dir', type=str ,default="/work/lingdongwang_umass_edu/Datasets/FaceScape/", help="dataset path")
parser.add_argument('--block', type=int, default=4, help='number of blocks in the model')
parser.add_argument('--channel', type=int, default=32, help='number of channels in the model')
parser.add_argument('--sampling', type=str, default="voxel", help="point sampling method")
parser.add_argument('--scale', type=int, default=2, help="input point cloud down-sampling scale")
parser.add_argument('--gt_scale', type=int, default=5, help="ground truth point cloud down-sampling scale")
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--log_path', type=str ,default='./result/log/', help="log path")
parser.add_argument('--save_model_path', type=str, default='./result/weight', help='location to save checkpoint models')
parser.add_argument('--verbose', action='store_true', default=False)

# training
parser.add_argument('--batch_size', type=int, default=16, help='training batch size')
parser.add_argument('--epochs', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=5, help='number of epochs to save a snapshot of model')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--step_size', type=int, default=10, help='frequency of learning rate decay')
parser.add_argument('--lr_decay', type=float, default=0.1 , help='learning rate decay')
parser.add_argument('--weight_decay', default=1e-04, type=float,help="weight decay")
parser.add_argument('--train_patch', type=int, default=1, help='number of patches sampled from an object')
parser.add_argument('--use_scaler', action='store_true', default=False)

# validation
parser.add_argument('--valid_every', type=int, default=5, help="frequency of validation")
parser.add_argument('--test_patch', type=int, default=1, help='number of patches sampled from an object')

args = parser.parse_args()


def main():
    sys.stdout = Logger(os.path.join(args.log_path, args.log_name + '.txt'))
    set_random_seed(args.seed)

    # gpu_devices = os.environ['CUDA_VISIBLE_DEVICES']
    # gpu_devices = gpu_devices.split(',')
    # print("Using GPU", gpu_devices)

    if not torch.cuda.is_available():
        raise Exception('No Gpu found, please run with gpu')

    cudnn.benchmark = True
    display_config(args)
    start_time = time.time()

    # dataset setting
    print('Loading Dataset ...')

    if args.dataset == "face_scape_vox":
        train_set = FaceScapeVoxelDataset(args.data_dir, "train", gt_scale=args.gt_scale, num_patch=args.train_patch)
        train_loader = DataLoader(dataset=train_set, num_workers=args.threads, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=sparse_collate, pin_memory=True, drop_last=True)
        valid_set = FaceScapeVoxelDataset(args.data_dir, "valid", gt_scale=args.gt_scale, num_patch=args.test_patch)
        valid_loader = DataLoader(dataset=valid_set, num_workers=args.threads, batch_size=1, shuffle=False,
                                  collate_fn=sparse_collate, pin_memory=True, drop_last=True)
        test_set = FaceScapeVoxelDataset(args.data_dir, "test", gt_scale=args.gt_scale, num_patch=args.test_patch)
        test_loader = DataLoader(dataset=test_set, num_workers=args.threads, batch_size=1, shuffle=False,
                                 collate_fn=sparse_collate, pin_memory=True, drop_last=True)
    else:
        raise NotImplementedError

    if args.sampling == "voxel":
        sampler = Voxelizer(voxel_size=args.scale, avg_color=True)
    else:
        raise NotImplementedError

    model = CUNet(voxel_size=args.scale, num_block=args.block, num_channel=args.channel)
    model = model.cuda()

    run_voxel(args, train_loader, valid_loader, test_loader, sampler, model)

    finish_time = time.time()
    print("Time consumption:", finish_time - start_time)
    return


if __name__ == '__main__':
    main()
