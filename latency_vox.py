import torch
from data.face_scape_voxel_dataset import FaceScapeVoxelDataset
from torchsparse.utils.collate import sparse_collate
from sampling.voxelize import Voxelizer
from models.cunet import CUNet
from torchsparse import SparseTensor
from time import time
from traditional.devoxelize import DevoxelizeInterpolation


def measure_latency_gpu(gt_scale, scale, num_block=5, num_channel=256, num_iter=200, warm_up=5):
    data_path = "/work/lingdongwang_umass_edu/Datasets/FaceScape/"
    dataset = FaceScapeVoxelDataset(data_path, "test", gt_scale=gt_scale)
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         collate_fn=sparse_collate,
                                         batch_size=1,
                                         shuffle=False,
                                         pin_memory=True)

    voxelizer = Voxelizer(scale)
    model = CUNet(scale, num_block=num_block, num_channel=num_channel).cuda()
    model.eval()
    torch.no_grad()

    # data preparation
    print("preparing data")
    inputs = []
    cnt = 0
    for sparse_hr in loader:
        sparse_hr = sparse_hr.cuda()
        sparse_lr, idx_hr2lr, idx_lr2hr = voxelizer(sparse_hr)
        inputs.append([sparse_lr.C.cpu()[:, :3], sparse_lr.F.cpu(), sparse_hr.C.cpu()[:, :3]])
        cnt += 1
        if cnt == num_iter:
            break

    # model inference
    print("start inference")
    torch.cuda.synchronize()
    start_time = time()
    for i in range(len(inputs)):
        if i == warm_up:
            start_time = time()
        points_lr, colors_lr, points_hr = inputs[i]

        points_lr, colors_lr, points_hr = points_lr.cuda(), colors_lr.cuda(), points_hr.cuda()
        points_lr = torch.cat([points_lr, torch.zeros((points_lr.shape[0], 1), dtype=torch.int32, device="cuda")], dim=1)
        points_hr = torch.cat([points_hr, torch.zeros((points_hr.shape[0], 1), dtype=torch.int32, device="cuda")], dim=1)
        sparse_lr = SparseTensor(coords=points_lr, feats=colors_lr)

        colors_hr_pred = model(sparse_lr, points_hr)

        colors_hr_pred = colors_hr_pred.cpu()

    torch.cuda.synchronize()
    end_time = time()

    avg_time = (end_time - start_time) / (num_iter - warm_up)
    print("avg time", avg_time)
    return


def measure_latency_cpu(gt_scale, scale, num_block=5, num_channel=256, num_iter=50, warm_up=5):
    data_path = "/work/lingdongwang_umass_edu/Datasets/FaceScape/"
    dataset = FaceScapeVoxelDataset(data_path, "test", gt_scale=gt_scale)
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         collate_fn=sparse_collate,
                                         batch_size=1,
                                         shuffle=False,
                                         pin_memory=True)

    voxelizer = Voxelizer(scale)
    model = CUNet(scale, num_block=num_block, num_channel=num_channel)
    model.eval()
    torch.no_grad()

    # data preparation
    print("preparing data")
    inputs = []
    cnt = 0
    for sparse_hr in loader:
        sparse_hr = sparse_hr.cuda()
        sparse_lr, idx_hr2lr, idx_lr2hr = voxelizer(sparse_hr)
        inputs.append([sparse_lr.C.cpu()[:, :3], sparse_lr.F.cpu(), sparse_hr.C.cpu()[:, :3]])
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

        # points_lr, colors_lr, points_hr = points_lr.cuda(), colors_lr.cuda(), points_hr.cuda()
        points_lr = torch.cat([points_lr, torch.zeros((points_lr.shape[0], 1), dtype=torch.int32, device="cpu")], dim=1)
        points_hr = torch.cat([points_hr, torch.zeros((points_hr.shape[0], 1), dtype=torch.int32, device="cpu")], dim=1)
        sparse_lr = SparseTensor(coords=points_lr, feats=colors_lr)

        colors_hr_pred = model(sparse_lr, points_hr)

        colors_hr_pred = colors_hr_pred.cpu()

    # torch.cuda.synchronize()
    end_time = time()

    avg_time = (end_time - start_time) / (num_iter - warm_up)
    print("avg time", avg_time)
    return


def devox_latency(gt_scale, scale, num_iter=200, warm_up=5, use_gpu=True):
    data_path = "/work/lingdongwang_umass_edu/Datasets/FaceScape/"
    dataset = FaceScapeVoxelDataset(data_path, "test", gt_scale=gt_scale)
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         collate_fn=sparse_collate,
                                         batch_size=1,
                                         shuffle=False,
                                         pin_memory=True)

    voxelizer = Voxelizer(scale)
    interp = DevoxelizeInterpolation(scale)
    torch.no_grad()

    # data preparation
    print("preparing data")
    inputs = []
    cnt = 0
    for sparse_hr in loader:
        sparse_hr = sparse_hr.cuda()
        sparse_lr, idx_hr2lr, idx_lr2hr = voxelizer(sparse_hr)
        inputs.append([sparse_lr.C.cpu()[:, :3], sparse_lr.F.cpu(), sparse_hr.C.cpu()[:, :3]])
        cnt += 1
        if cnt == num_iter:
            break

    # model inference
    print("start inference")
    torch.cuda.synchronize()
    start_time = time()
    for i in range(len(inputs)):
        if i == warm_up:
            start_time = time()
        points_lr, colors_lr, points_hr = inputs[i]
        if use_gpu:
            points_lr, colors_lr, points_hr = points_lr.cuda(), colors_lr.cuda(), points_hr.cuda()
        colors_hr_pred = interp(points_lr, colors_lr, points_hr, None)
        colors_hr_pred = colors_hr_pred.cpu()

    torch.cuda.synchronize()
    end_time = time()

    avg_time = (end_time - start_time) / (num_iter - warm_up)
    print("avg time", avg_time)
    return



if __name__ == '__main__':
    for scale in [2, 5, 10]:
        gt_scale = 10 // scale
        print("scale", scale, "gt_scale", gt_scale)
        measure_latency_gpu(gt_scale=gt_scale, scale=scale, num_block=4, num_channel=32)
        measure_latency_cpu(gt_scale=gt_scale, scale=scale, num_block=4, num_channel=64)
        devox_latency(gt_scale, scale, use_gpu=True)
        devox_latency(gt_scale, scale, use_gpu=False)


