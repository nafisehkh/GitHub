import torch
import torch.nn as nn
from e2nn import nn as e2nn
from e2nn import o3

class RConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 bias=True, mode=2):
        super(RConv2d, self).__init__()
        
        if mode == 1:
            in_type = e2nn.FieldType(o3.irreps('1x0e'), [in_channels])
        else:
            in_type = e2nn.FieldType(o3.irreps('1x1o'), [in_channels])
        
        out_type = e2nn.FieldType(o3.irreps('1x1o'), [out_channels])
        
        self.conv = e2nn.R2Conv(in_type, out_type, kernel_size, stride, padding, bias, initialize=True)
        self.in_type = in_type
        self.mode = mode
        
    def forward(self, x):
        x_geo = e2nn.GeometricTensor(x, self.in_type)
        return self.conv(x_geo).tensor

class VectorBatchNorm(nn.Module):
    def __init__(self, num_features):
        super(VectorBatchNorm, self).__init__()
        self.bn = nn.BatchNorm2d(num_features)
    def forward(self, x):
        return self.bn(x)

class Vector2Magnitude(nn.Module):
    def forward(self, x):
        batch, channels_times_2, h, w = x.shape
        channels = channels_times_2 // 2
        x_reshaped = x.view(batch, channels, 2, h, w)
        return torch.sqrt(x_reshaped[:, :, 0]**2 + x_reshaped[:, :, 1]**2)