
import torch
import h5py
import os
import numpy as np
from utils import vdSampleMask

class fastMRI_knee_dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, device, sample_frac=0.2, out_dir=None, mask=None, verbose=False, middle_slice_only=False):
        self.data_dir = data_dir
        self.device = device
        self.sample_frac = sample_frac
        self.out_dir = out_dir
        self.mask = mask
        self.verbose = verbose
        self.middle_slice_only = middle_slice_only
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.h5')]
        if mask is None:
            self.mask = torch.from_numpy(vdSampleMask(smask=[640, 320], sigmas=[640/4, 320/4], numSamps=int(640 * 320 * sample_frac), maskType='laplace')).float()
        if self.verbose:
            print(f"Mask non-zero fraction: {self.mask.mean().item():.3f}, expected ~{sample_frac}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        with h5py.File(file_path, 'r') as f:
            kspace = f['kspace'][()]
        kspace = torch.from_numpy(kspace)  # Load on CPU
        kspace = torch.flip(kspace, dims=[-1])  # Flip k-space horizontally
        if self.verbose:
            print(f"File: {self.file_list[idx]}, Raw kspace shape: {kspace.shape}")
            print(f"Raw kspace magnitude range: {torch.abs(kspace).min().item():.6e}, {torch.abs(kspace).max().item():.6e}")
        if self.middle_slice_only:
            kspace = kspace[15]  # Middle slice
        else:
            slice_idx = np.random.randint(0, kspace.shape[0])
            kspace = kspace[slice_idx]
        # Ensure kspace is at least [640, 320]
        if kspace.shape[-2] < 640 or kspace.shape[-1] < 320:
            raise ValueError(f"kspace shape {kspace.shape} too small for cropping to [640, 320]")
        # Center crop to [640, 320] around zero frequency
        height, width = kspace.shape[-2], kspace.shape[-1]
        start_x = (height - 640) // 2 if height > 640 else 0
        start_y = (width - 320) // 2 if width > 320 else 0
        kspace = kspace[start_x:start_x + 640, start_y:start_y + 320]
        if self.verbose:
            print(f"Cropped kspace shape: {kspace.shape}")
            print(f"Cropped kspace magnitude range: {torch.abs(kspace).min().item():.6e}, {torch.abs(kspace).max().item():.6e}")
        # Compute fully-sampled image (target_img)
        img = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(kspace, dim=(-2, -1)), norm='ortho', dim=(-2, -1)), dim=(-2, -1))
        img = torch.stack((img.real, img.imag), dim=0)  # [2, 640, 320]
        # Normalize img by maximum magnitude
        img_mag = torch.sqrt(img[0]**2 + img[1]**2)  # [640, 320]
        max_mag = img_mag.max()
        img = img / max(max_mag, 1e-10)  # [2, 640, 320], max magnitude = 1
        if self.verbose:
            print(f"Normalized img magnitude range: {torch.sqrt(img[0]**2 + img[1]**2).min().item():.6e}, {torch.sqrt(img[0]**2 + img[1]**2).max().item():.6e}")
        # Compute masked k-space (input_kspace)
        img_complex = img[0] + 1j * img[1]  # [640, 320], complex
        kspace = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(img_complex, dim=(-2, -1)), norm='ortho', dim=(-2, -1)), dim=(-2, -1))
        # Apply mask to real and imaginary parts separately
        kspace_real = kspace.real * self.mask
        kspace_imag = kspace.imag * self.mask
        kspace = torch.stack((kspace_real, kspace_imag), dim=0)  # [2, 640, 320]
        if self.verbose:
            print(f"Masked kspace magnitude range: {torch.sqrt(kspace[0]**2 + kspace[1]**2).min().item():.6e}, {torch.sqrt(kspace[0]**2 + kspace[1]**2).max().item():.6e}")
        # Move to device
        kspace = kspace.to(self.device)
        img = img.to(self.device)
        return kspace, img

# Created on 08:01 AM MDT, Monday, September 01, 2025, Artifact ID: 0a0ea9a0-842b-41a4-811b-c3418f4f9ea6

