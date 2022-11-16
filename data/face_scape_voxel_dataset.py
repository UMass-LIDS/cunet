import sys
sys.path.append("..")
import json
from torch.utils.data import Dataset
import os
from torchvision import transforms
import torch
import torch.nn.functional as F
import numpy as np
from utils import read_point_cloud_ply, normalize
from data.quantize import sparse_quantize
from torchsparse import SparseTensor
from torchsparse.utils.collate import sparse_collate



class FaceScapeVoxelDataset(Dataset):
    def __init__(self, data_dir, split, gt_scale=1, require_patch=False, num_patch=1, num_point=4096):
        super(FaceScapeVoxelDataset, self).__init__()
        assert split in ["train", "valid", "test"]

        self.split = split
        self.gt_scale = gt_scale
        self.require_patch = require_patch
        self.num_patch = num_patch
        self.num_point = num_point

        if not os.path.exists(data_dir):
            raise Exception(f"[!] {data_dir} not existed")
        list_name = os.path.join(data_dir, split + ".txt")
        data_dir = os.path.join(data_dir, "vox")
        self.data_dir = data_dir

        self.list_file = open(list_name).read()
        self.dir_names = self.list_file.split(',')

        self.file_names = []
        for dir_name in self.dir_names:
            ply_names = os.listdir(os.path.join(self.data_dir, dir_name))
            for ply_name in ply_names:
                file_name = self.data_dir + '/' + dir_name + "/" + ply_name
                self.file_names.append(file_name)
        self.file_names.sort()
        # print("Number of samples:", len(self.file_names))

    def __getitem__(self, idx):
        file_idx = idx // self.num_patch if self.require_patch else idx
        file_name = os.path.join(self.data_dir, self.file_names[file_idx])
        points_hr, colors_hr = read_point_cloud_ply(file_name)
        points_hr = points_hr - np.min(points_hr, axis=0)
        points_hr = points_hr.astype(np.int32)
        if self.gt_scale > 1:
            points_hr, idx_gt = sparse_quantize(points_hr, self.gt_scale, return_index=True, return_inverse=False)
            colors_hr = colors_hr[idx_gt, :]

        if self.require_patch:
            points_hr, colors_hr = self._crop_patch(points_hr, colors_hr)

        colors_hr = colors_hr.astype(np.float32)
        sparse_hr = SparseTensor(coords=points_hr, feats=colors_hr)
        return sparse_hr


    def _crop_patch(self, points, colors):
        # randomly sample centroids
        centroid_idx = np.random.randint(points.shape[0])
        dist = np.sum((points[centroid_idx, :] - points) ** 2, axis=-1)
        # collect k-nearest points around a centroid
        patch_idx = np.argpartition(dist, self.num_point)[:self.num_point]
        points_patch = points[patch_idx, :]
        colors_patch = colors[patch_idx, :]
        return points_patch, colors_patch


    def __len__(self):
        if self.require_patch:
            return self.num_patch * len(self.file_names)
        else:
            return len(self.file_names)



if __name__ == '__main__':
    from utils import draw_point_cloud
    # data_path = "D:/Datasets/FaceScape/"
    data_path = "/work/lingdongwang_umass_edu/Datasets/FaceScape/"
    dataset = FaceScapeVoxelDataset(data_path, "test", gt_scale=5)
    print(len(dataset))
    sparse_hr = dataset[0]
    print("sparse", sparse_hr.C.shape, sparse_hr.F.shape)
    # draw_point_cloud(points_lr, colors_lr)

    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         collate_fn=sparse_collate,
                                         batch_size=2,
                                         shuffle=False,
                                         pin_memory=True)

    for data in loader:
        sparse_hr = data
        print("loader", sparse_hr.C.shape, sparse_hr.F.shape)
        break