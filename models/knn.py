from torch import nn
import torch
import torch.nn.functional as F


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    return torch.sum((src[:, :, None] - dst[:, None]) ** 2, dim=-1)


class KNNTorch(nn.Module):
    def __init__(self, num_query=3, num_feat=320):
        super(KNNTorch, self).__init__()
        assert num_feat == 64 + 128 + 128
        self.num_query = num_query
        self.num_feat = num_feat

    def forward(self, points1, points2, colors1):
        B, N1, _ = points1.shape
        _, N2, _ = points2.shape

        # compute distances from points2 to points1
        dist = square_distance(points2, points1)  # (B, N2, N1)
        query_val, query_idx = torch.topk(dist, k=self.num_query, largest=False, dim=-1)  # (B, N2, K), (B, N2, K)

        # compute k-nn weighted average color
        query_color_idx = query_idx[:, :, :, None].expand((-1, -1, -1, 3))  # (B, N2, K, 3)
        colors1 = colors1[:, None, :, :].expand((-1, N2, -1, -1))  # (B, N2, N1, 3)
        knn_color = torch.gather(colors1, dim=2, index=query_color_idx)  # (B, N2, K, 3)
        knn_color = torch.mean(knn_color, dim=2)
        return knn_color



if __name__ == '__main__':
    batch_size = 1
    num_points = 2500
    scale = 16

    torch.no_grad()
    model = KNNTorch()
    model = model.cuda()
    points1 = torch.randn((batch_size, num_points, 3)).cuda()
    points2 = torch.randn((batch_size, num_points*(scale-1), 3)).cuda()
    colors1 = torch.randn((batch_size, num_points, 3)).cuda()
    colors2 = model(points1, points2, colors1)
    print(colors2.shape)