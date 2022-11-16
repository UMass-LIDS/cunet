import sys
sys.path.append("..")
import numpy as np
import torch
import torchsparse.nn.functional as F
from torchsparse import SparseTensor


class Voxelizer:
    def __init__(self, voxel_size, avg_color=True):
        super(Voxelizer, self).__init__()
        self.voxel_size = voxel_size
        self.avg_color = avg_color  # use average / randomly-picked color in voxelization

    def __call__(self, sparse_hr:SparseTensor):
        coords_hr = sparse_hr.C
        feats_hr = sparse_hr.F
        device = coords_hr.device
        # print("hr", coords_hr.shape, feats_hr.shape)

        coords_lr = torch.cat([coords_hr[:, :3] / self.voxel_size, coords_hr[:, -1].view(-1, 1)], 1)
        coords_lr = torch.floor(coords_lr).int()
        hash_lr = F.sphash(coords_lr)
        hash_lr, idx_lr2hr = torch.unique(hash_lr, sorted=False, return_inverse=True)
        counts = F.spcount(idx_lr2hr.int(), len(hash_lr))

        idx_hr2lr = torch.arange(idx_lr2hr.size(0), dtype=idx_lr2hr.dtype, device=idx_lr2hr.device)
        idx_lr2hr, idx_hr2lr = idx_lr2hr.flip([0]), idx_hr2lr.flip([0])
        idx_hr2lr = idx_lr2hr.new_empty(hash_lr.size(0)).scatter_(0, idx_lr2hr, idx_hr2lr)
        idx_lr2hr = idx_lr2hr.flip([0])

        coords_lr = coords_lr[idx_hr2lr, :]
        if self.avg_color:
            feats_lr = F.spvoxelize(feats_hr, idx_lr2hr, counts)
        else:
            feats_lr = feats_hr[idx_hr2lr, :]
        # print("lr", coords_lr.shape, feats_lr.shape)
        sparse_lr = SparseTensor(coords=coords_lr, feats=feats_lr).to(device)
        return sparse_lr, idx_hr2lr, idx_lr2hr


def voxelize_avg_numpy(coords_hr, feats_hr, voxel_size):
    coords_lr = torch.from_numpy(coords_hr) / voxel_size
    coords_lr = torch.floor(coords_lr).int()
    coords_lr = torch.cat([coords_lr, torch.zeros((coords_lr.shape[0], 1), dtype=torch.int32)], dim=1)
    pc_hash = F.sphash(coords_lr)
    sparse_hash, idx_lr2hr = torch.unique(pc_hash, return_inverse=True)
    counts = F.spcount(idx_lr2hr.int(), len(sparse_hash))

    idx_hr2lr = torch.arange(idx_lr2hr.size(0), dtype=idx_lr2hr.dtype, device=idx_lr2hr.device)
    idx_lr2hr, idx_hr2lr = idx_lr2hr.flip([0]), idx_hr2lr.flip([0])
    idx_hr2lr = idx_lr2hr.new_empty(sparse_hash.size(0)).scatter_(0, idx_lr2hr, idx_hr2lr)
    idx_lr2hr = idx_lr2hr.flip([0])
    coords_lr = coords_lr[idx_hr2lr, :]

    feats_hr = torch.from_numpy(feats_hr).float()
    feats_lr = F.spvoxelize(feats_hr, idx_lr2hr, counts)
    coords_lr = coords_lr.numpy().astype(int)[:, :3]
    feats_lr = feats_lr.numpy().astype(float)
    idx_lr2hr = idx_lr2hr.numpy()
    # print("avg voxelize", coords_lr.shape, feats_lr.shape)
    # print(coords_lr)

    return coords_lr, feats_lr, idx_lr2hr


def test(scale=2, gt_scale=5):
    from data.face_scape_voxel_dataset import FaceScapeVoxelDataset
    from torchsparse.utils.collate import sparse_collate
    from utils import draw_point_cloud

    data_path = "/work/lingdongwang_umass_edu/Datasets/FaceScape/"
    dataset = FaceScapeVoxelDataset(data_path, "test", False, gt_scale=5)
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         collate_fn=sparse_collate,
                                         batch_size=1,
                                         shuffle=False,
                                         pin_memory=True)

    voxelizer = Voxelizer(scale)
    for data in loader:
        print(data.C.shape, data.F.shape)
        sparse_lr, idx_hr2lr, idx_lr2hr = voxelizer(data)
        print("out", sparse_lr.C.shape, sparse_lr.F.shape)
        points = sparse_lr.C.float()[:, :3]
        colors = sparse_lr.F.float()
        draw_point_cloud(points, colors, visualize=False, save_path='test.ply')

        break

if __name__ == '__main__':
    test()