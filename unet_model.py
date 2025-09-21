
import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, image_size, channels, mask):
        super().__init__()
        self.down_convs = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        self.mask = mask.to('cpu')  # Ensure mask is on CPU for FFT operations

        # Downsampling
        in_ch = channels
        out_ch = 64
        for i in range(4):
            self.down_convs.append(DoubleConv(in_ch, out_ch))
            in_ch = out_ch
            out_ch *= 2

        # Bottleneck
        self.bottleneck = DoubleConv(in_ch, in_ch * 2)

        # Upsampling
        out_ch = in_ch * 2
        for i in range(4):
            self.up_convs.append(nn.ConvTranspose2d(out_ch, out_ch // 2, kernel_size=2, stride=2))
            self.up_convs.append(DoubleConv(out_ch, out_ch // 2))
            out_ch //= 2

        self.final_conv = nn.Conv2d(out_ch, channels, kernel_size=1)

    def forward(self, x, kspace, apply_dc=False):
        skips = []
        for down in self.down_convs:
            x = down(x)
            skips.append(x)
            x = nn.MaxPool2d(2)(x)

        x = self.bottleneck(x)

        skips = skips[::-1]
        for idx in range(0, len(self.up_convs), 2):
            x = self.up_convs[idx](x)
            skip = skips[idx // 2]
            x = torch.cat((skip, x), dim=1)
            x = self.up_convs[idx + 1](x)

        x = self.final_conv(x)

        if apply_dc:
            output_list = []
            device = x.device  # Store original device
            x = x.to('cpu')  # Move to CPU for complex operations
            kspace = kspace.to('cpu')  # Move kspace to CPU
            for b in range(x.size(0)):
                x_b = x[b:b+1]  # [1, 2, 640, 320]
                kspace_b = kspace[b:b+1]  # [1, 2, 640, 320]
                x_complex = x_b[:, 0, :, :] + 1j * x_b[:, 1, :, :]  # [1, 640, 320]
                x_k = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x_complex, dim=(-2, -1)), norm='ortho', dim=(-2, -1)), dim=(-2, -1))
                kspace_complex = kspace_b[:, 0, :, :] + 1j * kspace_b[:, 1, :, :]  # [1, 640, 320]
                x_k = x_k * (1 - self.mask) + kspace_complex * self.mask  # Apply DC
                x_b = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(x_k, dim=(-2, -1)), norm='ortho', dim=(-2, -1)), dim=(-2, -1))
                x_b = torch.stack((x_b.real, x_b.imag), dim=1)  # [1, 2, 640, 320]
                output_list.append(x_b)
            x = torch.cat(output_list, dim=0).to(device)  # Move back to original device

        return x

# Created on 07:41 AM MDT, Tuesday, September 02, 2025, Artifact ID: d1cf093c-2dd1-48c0-9981-bcf52e41eb68

