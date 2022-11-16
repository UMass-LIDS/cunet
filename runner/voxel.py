from tqdm import tqdm
import torch
import os
from torch import optim
from torch.optim import lr_scheduler
import sys
from metric.psnr import compute_psnr_torch_rgb
from torch.cuda import amp


def run_voxel(args, train_loader, valid_loader, test_loader, sampler, model):
    criterion = torch.nn.MSELoss()
    criterion = criterion.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.lr_decay)
    scaler = amp.GradScaler(enabled=args.use_scaler)

    for epoch in range(1, args.epochs + 1):
        train(train_loader, model, sampler, criterion, optimizer, scaler, epoch, verbose=args.verbose)
        scheduler.step()
        if epoch % args.valid_every == 0:
            valid(valid_loader, model, sampler, verbose=args.verbose)
        if epoch % (args.snapshots) == 0:
            checkpoint(args, model, epoch)
        sys.stdout.flush()
    print("Testing Result")
    valid(test_loader, model, sampler, verbose=args.verbose)
    return


def train(train_loader, model, sampler, criterion, optimizer, scaler, epoch, verbose=False):
    epoch_loss = 0
    model.train()
    data_enumerator = enumerate(tqdm(train_loader)) if verbose else enumerate(train_loader)
    for iteration, sparse_hr in data_enumerator:
        optimizer.zero_grad()
        sparse_hr = sparse_hr.cuda()
        sparse_lr, idx_hr2lr, idx_lr2hr = sampler(sparse_hr)
        colors_pred = model(sparse_lr, sparse_hr.C, idx_lr2hr=idx_lr2hr)
        loss = criterion(colors_pred, sparse_hr.F)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()
    print("Epoch {} Loss: {}".format(epoch, epoch_loss/len(train_loader)))


def valid(valid_loader, model, sampler, verbose=False):
    total_psnr = 0
    model.eval()
    data_enumerator = enumerate(tqdm(valid_loader)) if verbose else enumerate(valid_loader)
    with torch.no_grad():
        for iteration, sparse_hr in data_enumerator:
            sparse_hr = sparse_hr.cuda()
            sparse_lr, idx_hr2lr, idx_lr2hr = sampler(sparse_hr)
            colors_pred = model(sparse_lr, sparse_hr.C, idx_lr2hr=idx_lr2hr)
            psnr = compute_psnr_torch_rgb(colors_pred, sparse_hr.F)
            total_psnr += psnr
    total_psnr /= len(valid_loader)
    print('Valid PSNR(RGB): {}'.format(total_psnr))


def checkpoint(args, model, epoch):
    save_model_path = os.path.join(args.save_model_path, args.log_name)
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
    model_name = '{}'.format(args.log_name) + '_epoch_{}.pth'.format(epoch)
    torch.save(model.state_dict(), os.path.join(save_model_path, model_name))
    print('Checkpoint saved to {}'.format(os.path.join(save_model_path, model_name)))

