import sys
sys.path.append("..")
from typing import List, Tuple, Union
from torchsparse import SparseTensor
from torchsparse import nn as spnn
from torchsparse.utils.collate import sparse_collate_fn
from torchsparse.utils.quantize import sparse_quantize
import torchsparse.nn.functional as F
from utils import read_point_cloud_ply, draw_point_cloud
import math
import torch
import numpy as np
from torch import nn


class SparseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, use_res=True, use_bias=True):
        super().__init__()
        self.use_res = use_res
        self.main = nn.Sequential(
            spnn.Conv3d(in_channels,
                        out_channels,
                        kernel_size,
                        stride=stride,
                        bias=use_bias),
            spnn.BatchNorm(out_channels),
            spnn.ReLU(True),
            spnn.Conv3d(out_channels,
                        out_channels,
                        kernel_size,
                        stride=stride,
                        bias=use_bias),
            spnn.BatchNorm(out_channels)
        )
        self.relu = spnn.ReLU(True)

    def forward(self, x: SparseTensor) -> SparseTensor:
        if self.use_res:
            x = self.relu(self.main(x) + x)
        else:
            x = self.relu(self.main(x))
        return x


class CUNet(nn.Module):
    def __init__(self, voxel_size, num_channel=256, num_block=5, global_res=True, use_bias=True):
        '''
        :param voxel_size: voxel size, or up-sampling scale
        :param num_channel: number of channels for features
        :param num_block: number of sparse conv blocks
        :param global_res: use global residual
        :param use_bias: use bias in sparse conv blocks
        '''

        super(CUNet, self).__init__()
        self.voxel_size = voxel_size
        self.global_res = global_res

        self.color_block = nn.Sequential()
        self.color_block.append(SparseBlock(3, num_channel, use_res=False, use_bias=use_bias))
        for i in range(num_block-1):
            self.color_block.append(SparseBlock(num_channel, num_channel, use_res=True, use_bias=use_bias))
        self.out_block = nn.Sequential(
            nn.Linear(num_channel + 3, num_channel // 2),
            nn.ReLU(),
            nn.Linear(num_channel // 2, num_channel // 4),
            nn.ReLU(),
            nn.Linear(num_channel // 4, 3)
        )

    def forward(self, sparse_lr: SparseTensor, points_hr: torch.Tensor, idx_lr2hr=None) -> torch.Tensor:
        '''
        :param sparse_lr: (N_lr, 3), (N_lr, 3), LR point cloud with geometry and color
        :param points_hr: (N_hr, 3), HR point cloud geometry
        :param idx_lr2hr: (N_hr), LR-HR point cloud mapping indices, can be pre-computed for training efficiency
        :return: (N_hr, 3), HR point cloud color
        '''

        # extract LR color features
        sparse_color = self.color_block(sparse_lr)  # (N_lr, 3), (N_lr, C)

        # compute LR-HR mapping
        if idx_lr2hr is None:
            quant_hr = torch.cat([points_hr[:, :3] / self.voxel_size, points_hr[:, -1].view(-1, 1)], 1)
            quant_hr = torch.floor(quant_hr).int()
            hash_hr = F.sphash(quant_hr)
            _, idx_lr2hr = torch.unique(hash_hr, sorted=False, return_inverse=True)  # (N_hr)

        # expand LR features for HR points
        color_feat = sparse_color.feats  # (N_lr, C)
        color_feat = color_feat[idx_lr2hr, :]  # (N_hr, C)

        # compute query offsets
        points_lr = sparse_lr.C[idx_lr2hr, :3]  # (N_hr, 3)
        offset = points_hr[:, :3] - self.voxel_size * points_lr  # (N_hr, 3)
        offset = 2 * offset / (self.voxel_size - 1) - 1  # normalize offsets to [-1, 1]

        # predict HR colors
        out_feat = torch.cat([color_feat, offset], dim=1)  # (N_hr, C+3) / (N_hr, C+C_off)
        out = self.out_block(out_feat)  # (N_hr, 3)

        # global residual
        if self.global_res:
            out = out + sparse_lr.F[idx_lr2hr, :]  # (N_hr, 3)

        if not self.training:
            out = torch.clamp(out, min=0, max=1)
        return out


if __name__ == '__main__':
    from data.face_scape_voxel_dataset import FaceScapeVoxelDataset
    from torchsparse.utils.collate import sparse_collate
    from sampling.voxelize import Voxelizer

    batch_size = 8
    gt_scale = 5
    scale = 2
    training = True

    data_path = "/work/lingdongwang_umass_edu/Datasets/FaceScape/"
    dataset = FaceScapeVoxelDataset(data_path, "test", gt_scale=gt_scale)
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         collate_fn=sparse_collate,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         pin_memory=True)

    voxelizer = Voxelizer(scale)
    model = CUNet(scale).cuda()
    if not training:
        model.eval()
        torch.no_grad()

    for sparse_hr in loader:
        sparse_hr = sparse_hr.cuda()
        sparse_lr, idx_hr2lr, idx_lr2hr = voxelizer(sparse_hr)

        print("hr", sparse_hr.C.shape)
        print("lr", sparse_lr.C.shape)
        out = model(sparse_lr, sparse_hr.C, idx_lr2hr=idx_lr2hr)
        print("out", out.shape)

        if training:
            loss = torch.nn.functional.mse_loss(out, sparse_hr.F)
            loss.backward()
            print("backward pass")

        break