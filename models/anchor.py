"""完整共享的回归 MLP；各方法使用相同隐藏层和初始化。"""
import torch
from torch import nn


class NeighborResidual(nn.Module):
    """先执行扫描级逆距离加权，再学习坐标残差。"""
    def __init__(self, network):
        super().__init__()
        self.network = network
        nn.init.zeros_(network[-1].weight)
        nn.init.zeros_(network[-1].bias)

    def forward(self, x):
        points = x.reshape(len(x), -1, 3)
        weight = (points[:, :, 0]+1e-8).pow(-2)
        weight = weight/weight.sum(1, keepdim=True)
        prior = (weight[:, :, None]*points[:, :, 1:]).sum(1)
        return prior+self.network(x)


def make_network(width, hidden, seed, residual=False):
    torch.manual_seed(seed)
    layers = []
    for before, after in zip([width]+hidden, hidden+[2]):
        layers.append(nn.Linear(before, after))
        if after != 2:
            layers.append(nn.ReLU())
    model = nn.Sequential(*layers)
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    return NeighborResidual(model) if residual else model
