import copy
import torch
import numpy as np
from models.cunet import CUNet
from torchsparse import SparseTensor

class DeepInterpolation:
    def __init__(self, num_channel, num_block, voxel_size, model_path, use_gpu=True):
        super(DeepInterpolation, self).__init__()
        self.model = CUNet(voxel_size, num_channel=num_channel, num_block=num_block)
        self.model.load_state_dict(torch.load(model_path))
        self.use_gpu = use_gpu
        if use_gpu:
            self.model.cuda()

    def __call__(self, points_lr, colors_lr, points_hr, idx_lr2hr=None):
        assert idx_lr2hr is not None, "Not Implemented"
        points_lr = expand_batch(points_lr)
        points_hr = expand_batch(points_hr)
        colors_lr = torch.from_numpy(colors_lr).float()
        sparse_lr = SparseTensor(coords=points_lr, feats=colors_lr)
        idx_lr2hr = torch.from_numpy(idx_lr2hr).long()

        if self.use_gpu:
            sparse_lr = sparse_lr.cuda()
            points_hr = points_hr.cuda()
            idx_lr2hr = idx_lr2hr.cuda()

        with torch.no_grad():
            self.model.eval()
            colors_pred = self.model(sparse_lr, points_hr, idx_lr2hr=idx_lr2hr)

        colors_pred = colors_pred.cpu().numpy()
        return colors_pred


def expand_batch(points_np):
    points = torch.from_numpy(points_np).int()
    batch = torch.zeros((points.shape[0], 1)).int()
    points = torch.cat([points, batch], dim=1)
    # print("points", points.shape)
    return points
