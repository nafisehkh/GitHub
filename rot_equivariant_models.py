import torch.nn as nn
from rot_equivariant_layers import RConv2d, VectorBatchNorm, Vector2Magnitude

class RotEquivariantUNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=1):
        super(RotEquivariantUNet, self).__init__()
        
        self.enc1 = self._rot_block(in_channels, 32, mode=1)
        self.enc2 = self._rot_block(32, 64, mode=2)
        self.enc3 = self._rot_block(64, 128, mode=2)
        
        self.dec1 = self._rot_block(128, 64, mode=2)
        self.dec2 = self._rot_block(64, 32, mode=2)
        
        self.vector_to_scalar = Vector2Magnitude()
        self.final_conv = nn.Conv2d(32, out_channels, 1)
        
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    
    def _rot_block(self, in_channels, out_channels, mode=2):
        return nn.Sequential(
            RConv2d(in_channels, out_channels, 3, padding=1, mode=mode),
            VectorBatchNorm(out_channels),
            nn.ReLU(inplace=True),
            RConv2d(out_channels, out_channels, 3, padding=1, mode=2),
            VectorBatchNorm(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        
        d1 = self.dec1(self.upsample(e3))
        d2 = self.dec2(self.upsample(d1 + e2))
        
        scalar = self.vector_to_scalar(d2 + e1)
        return self.final_conv(scalar)