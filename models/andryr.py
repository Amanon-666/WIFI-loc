"""从指定版本 andryr notebook 提取的 MLP；来源见 BASELINE_PROVENANCE.md。"""

import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """原仓库的六隐藏层定位网络。"""
    def __init__(self, widths, num_outputs, bn_eps, bn_momentum, negative_slope):
        super(MLP, self).__init__()
        self.negative_slope = negative_slope
        
        self.linear1 = nn.Linear(520, widths[0])
        self.bn1 = nn.BatchNorm1d(widths[0], eps=bn_eps, momentum=bn_momentum)

        self.linear2 = nn.Linear(widths[0], widths[1])
        self.bn2 = nn.BatchNorm1d(widths[1], eps=bn_eps, momentum=bn_momentum)
        
        self.linear3 = nn.Linear(widths[1], widths[2])
        self.bn3 = nn.BatchNorm1d(widths[2], eps=bn_eps, momentum=bn_momentum)
        
        self.linear4 = nn.Linear(widths[2], widths[3])
        self.bn4 = nn.BatchNorm1d(widths[3], eps=bn_eps, momentum=bn_momentum)
        
        self.linear5 = nn.Linear(widths[3], widths[4])
        self.bn5 = nn.BatchNorm1d(widths[4], eps=bn_eps, momentum=bn_momentum)
        
        self.linear6 = nn.Linear(widths[4], widths[5])
        self.bn6 = nn.BatchNorm1d(widths[5], eps=bn_eps, momentum=bn_momentum)
    

        self.final = nn.Linear(widths[5], num_outputs)
      
    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.linear1(x)), self.negative_slope)
        x = F.leaky_relu(self.bn2(self.linear2(x)), self.negative_slope)
        x = F.leaky_relu(self.bn3(self.linear3(x)), self.negative_slope)
        x = F.leaky_relu(self.bn4(self.linear4(x)), self.negative_slope)
        x = F.leaky_relu(self.bn5(self.linear5(x)), self.negative_slope)
        x = F.leaky_relu(self.bn6(self.linear6(x)), self.negative_slope)
    
        x = self.final(x)
        return x
