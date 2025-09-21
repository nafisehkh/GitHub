
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchmetrics.functional import structural_similarity_index_measure
from scipy.stats import laplace, norm

def compute_visualization_data(image, target=None, device='cpu'):
    image_cpu = image.to('cpu')
    mag = torch.sqrt(image_cpu[:, 0, :, :]**2 + image_cpu[:, 1, :, :]**2)
    mag = torch.clamp(mag, 0, 1).numpy()

    # Compute k-space
    complex_img = image_cpu[:, 0, :, :] + 1j * image_cpu[:, 1, :, :]
    kspace = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(complex_img, dim=(-2, -1)), norm='ortho', dim=(-2, -1)))
    kspace_mag = torch.abs(kspace)
    kspace_db = 20 * torch.log10(kspace_mag / (kspace_mag.max() + 1e-10) + 1e-10)
    kspace_db = kspace_db.numpy()

    # Compute target magnitude if provided
    mag_target = None
    if target is not None:
        target_cpu = target.to('cpu')
        mag_target = torch.sqrt(target_cpu[:, 0, :, :]**2 + target_cpu[:, 1, :, :]**2)
        mag_target = torch.clamp(mag_target, 0, 1).numpy()

    # Compute SSIM if target is provided
    mean_ssim = None
    if target is not None:
        try:
            # Ensure inputs are in correct shape: [batch_size, 1, height, width]
            mag_torch = torch.from_numpy(mag)[:, None, :, :].to(device)
            mag_target_torch = torch.from_numpy(mag_target)[:, None, :, :].to(device)
            mean_ssim = structural_similarity_index_measure(mag_torch, mag_target_torch, data_range=1.0).item()
        except:
            mean_ssim = None

    return mag, kspace_db, mean_ssim, mag_target

def size2imgCoordinates(smask):
    """
    Convert image size to coordinate arrays for each dimension.
    INPUT:
        smask - 1-D array of image dimensions (e.g., [height, width])
    OUTPUT:
        coords - List of 1-D arrays with centered coordinates for each dimension
    """
    coords = []
    for size in smask:
        # Generate coordinates from -size/2 to size/2-1
        coord = np.arange(size) - size // 2
        coords.append(coord)
    return coords

def vdSampleMask(smask, sigmas, numSamps, maskType='laplace'):
    """
    Generates a variable density sample mask
    INPUTS:
        smask - 1-D array corresponding to number of samples in each dimension
        sigmas - 1-D array corresponding to the standard deviation of the distribution
                 in each dimension
        numSamps - number of (total) samples
        maskType - 'laplace' or 'gaussian'
    """
    maxIters = 500
    rng = np.random.default_rng(20230911)
    coords = size2imgCoordinates(smask)
    mask = np.zeros(smask, dtype=np.float32)
    nDims = len(smask)  # can't pass in just an integer
    if maskType == 'laplace':
        pdf = lambda x, sig: laplace.pdf(x, loc=0, scale=np.sqrt(0.5*sig*sig))
    elif maskType == 'gaussian':
        pdf = lambda x, sig: norm.pdf(x, loc=0, scale=sig)
    for idx in range(maxIters):
        sampsLeft = int(numSamps - mask.sum(dtype=int))
        dimSamps = np.zeros((nDims, sampsLeft))
        for dimIdx in range(nDims):
            c = coords[dimIdx]
            probs = pdf(c, sigmas[dimIdx])
            probs = probs / sum(probs)
            samps = rng.choice(c, sampsLeft, p=probs)
            dimSamps[dimIdx, :] = samps - min(c)

        mask[tuple(dimSamps.astype(int))] = 1
        if mask.sum(dtype=int) >= numSamps:
            return mask

    print('hit max iters vdSampleMask')
    return mask

def visualize_images(fig, mag, kspace_db, title, save_path=None, losses=None, mag_target=None):
    print(f"Visualizing {mag.shape[0]} images with title: {title}")
    num_images = mag.shape[0]
    fig.clear()

    # Calculate figure dimensions
    img_w = 1.2  # Keep as perfect size
    img_h = img_w * 2  # 2.4
    scale = img_h
    ncols = min(num_images, 4)
    nrows = 3 if mag_target is not None else 2  # Adjust rows based on mag_target
    gap = 0.15
    title_h = 0.4
    loss_h = 0.1
    cbar_w = 0.08
    cbar_gap = 0.03
    left_margin = 0.7  # Keep for row titles
    right_margin = 0.5
    fig_w = ncols * img_w + (ncols - 1) * gap + (2 if mag_target is None else 3) * cbar_w + (2 if mag_target is None else 3) * cbar_gap + left_margin + right_margin
    fig_h = nrows * img_h + (nrows - 1) * gap + title_h + loss_h

    fig.set_size_inches(fig_w, fig_h)  # Adjust figure size dynamically

    if ncols == 0:
        print("Warning: No images to visualize (ncols=0)")
        return fig

    for i in range(ncols):
        # Output magnitude (first row)
        ax = fig.add_axes([(i * (img_w + gap) + left_margin) / fig_w, (fig_h - img_h - title_h - loss_h) / fig_h, img_w / fig_w, img_h / fig_h])
        im = ax.imshow(mag[i], cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
        ax.set_aspect('equal')
        if losses is not None and i < len(losses):
            ax.text(0.5, 1.05, f'Loss: {losses[i]:.2e}', fontsize=10, ha='center', va='bottom', transform=ax.transAxes)
        if i == 0:
            ax.text(-0.3, 0.5, 'Output Magnitude', fontsize=12, ha='right', va='center', transform=ax.transAxes, rotation=90)
        if i == ncols - 1:
            cbar_ax_mag = fig.add_axes([(ncols * (img_w + gap) + left_margin + cbar_gap) / fig_w, (fig_h - img_h - title_h - loss_h) / fig_h, cbar_w / fig_w, img_h / fig_h])
            fig.colorbar(im, cax=cbar_ax_mag)

        # K-space (second row)
        ax = fig.add_axes([(i * (img_w + gap) + left_margin) / fig_w, (fig_h - 2 * img_h - gap - title_h - loss_h) / fig_h, img_w / fig_w, img_h / fig_h])
        im = ax.imshow(kspace_db[i], cmap='viridis')
        ax.axis('off')
        ax.set_aspect('equal')
        if i == 0:
            ax.text(-0.3, 0.5, 'K-space (dB)', fontsize=12, ha='right', va='center', transform=ax.transAxes, rotation=90)
        if i == ncols - 1:
            cbar_ax_kspace = fig.add_axes([(ncols * (img_w + gap) + left_margin + cbar_gap) / fig_w, (fig_h - 2 * img_h - gap - title_h - loss_h) / fig_h, cbar_w / fig_w, img_h / fig_h])
            fig.colorbar(im, cax=cbar_ax_kspace)

        # Target magnitude (third row, if provided)
        if mag_target is not None:
            ax = fig.add_axes([(i * (img_w + gap) + left_margin) / fig_w, (fig_h - 3 * img_h - 2 * gap - title_h - loss_h) / fig_h, img_w / fig_w, img_h / fig_h])
            im = ax.imshow(mag_target[i], cmap='gray', vmin=0, vmax=1)
            ax.axis('off')
            ax.set_aspect('equal')
            if i == 0:
                ax.text(-0.3, 0.5, 'Target Magnitude', fontsize=12, ha='right', va='center', transform=ax.transAxes, rotation=90)
            if i == ncols - 1:
                cbar_ax_target = fig.add_axes([(ncols * (img_w + gap) + left_margin + cbar_gap) / fig_w, (fig_h - 3 * img_h - 2 * gap - title_h - loss_h) / fig_h, cbar_w / fig_w, img_h / fig_h])
                fig.colorbar(im, cax=cbar_ax_target)

    fig.suptitle(title, fontsize=12, y=0.99)
    if save_path:
        print(f"Saving visualization to {save_path}")
        fig.savefig(save_path, bbox_inches='tight', dpi=150)

    return fig

# Created on 08:30 AM MDT, Monday, September 01, 2025, Artifact ID: b31aaba1-8045-44a3-8855-4093605cb211

