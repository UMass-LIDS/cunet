import torch
import torchsparse.nn.functional as F

def devoxelize_interp(points_lr, colors_lr, points_hr, idx_lr2hr):
    result = colors_lr[idx_lr2hr, :]
    return result


# devoxlization without pre-computed LR-to-HR mapping
class DevoxelizeInterpolation:
    def __init__(self, voxel_size):
        self.voxel_size = voxel_size

    def __call__(self, points_lr, colors_lr, points_hr, idx_lr2hr):
        if idx_lr2hr is None:
            device = points_hr.device
            points_hr = torch.cat([points_hr, torch.zeros((points_hr.shape[0], 1), dtype=torch.int32, device=device)], dim=1)
            quant_hr = torch.cat([points_hr[:, :3] / self.voxel_size, points_hr[:, -1].view(-1, 1)], 1)
            quant_hr = torch.floor(quant_hr).int()
            hash_hr = F.sphash(quant_hr)
            _, idx_lr2hr = torch.unique(hash_hr, return_inverse=True)  # (N_hr)
        result = colors_lr[idx_lr2hr, :]
        return result