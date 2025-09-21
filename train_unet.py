
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from unet_model import UNet
from fastmri_dataloader import fastMRI_knee_dataset
from utils import visualize_images, vdSampleMask, compute_visualization_data
import os
import matplotlib.pyplot as plt
import numpy as np
import csv
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Train UNet for MRI reconstruction')
parser.add_argument('--apply_dc', action='store_true', default=False, help='Apply data consistency (default: False)')
args = parser.parse_args()

# Set device based on priority: mps > cuda > cpu
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Print device only in the main process
if __name__ == "__main__":
    print(f"Using device: {device}")

# Hyperparameters
channels = 2
batch_size = 8
learning_rate = 0.001
num_epochs = 1000
viz_interval = 10
save_interval = 10
print_interval = 2
sample_frac = 0.1
model_save_interval = 50
apply_dc = args.apply_dc
out_dir = "./out_noDC" if not apply_dc else "./out_yesDC"
data_dir_train = "/Volumes/T7_Shield/FastMRI/knee/singlecoil_train"
data_dir_test = "/Volumes/T7_Shield/FastMRI/knee/singlecoil_train"
verbose = False

# Create output directories
if __name__ == "__main__":
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "visualizations"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "train_visualizations"), exist_ok=True)
    mask = torch.from_numpy(vdSampleMask(smask=[640, 320], sigmas=[640/4, 320/4], numSamps=int(640 * 320 * sample_frac), maskType='laplace')).float()
    mask_path = os.path.join(out_dir, "sample_mask.png")
    plt.imsave(mask_path, mask.numpy(), cmap='gray')
    print(f"Saved mask to {mask_path}")
    print(f"Mask non-zero fraction: {mask.mean().item():.3f}, expected ~{sample_frac}")

# Initialize dataset
train_dataset = fastMRI_knee_dataset(data_dir_train, device=device, sample_frac=sample_frac, out_dir=out_dir, mask=mask, verbose=verbose)
test_dataset = fastMRI_knee_dataset(data_dir_test, device=device, middle_slice_only=True, sample_frac=sample_frac, out_dir=out_dir, mask=mask, verbose=verbose)

# Print test dataset size
if __name__ == "__main__":
    print(f"Test dataset size: {len(test_dataset)}")

# Determine image_size
with torch.no_grad():
    _, out_img = test_dataset[0]
    image_size = [out_img.size(1), out_img.size(2)]

# Initialize dataloaders
dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Initialize model, loss, and optimizer
model = UNet(image_size=image_size, channels=channels, mask=mask).to(device)
mse_criterion = nn.MSELoss(reduction='none').to(device)
mae_criterion = nn.L1Loss(reduction='none').to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Initialize figures for visualization
test_fig = plt.figure()
train_fig = plt.figure()

# Initialize CSV for loss logging
loss_log_path = os.path.join(out_dir, "loss_log.csv")
if __name__ == "__main__":
    with open(loss_log_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Epoch', 'Batch', 'Loss', 'Average_Epoch_Loss', 'Mean_SSIM'])

# Initial zero-filled visualization
if __name__ == "__main__":
    model.eval()
    with torch.no_grad():
        plt.ion()  # Enable interactive mode
        print(f"Total test dataset samples: {len(test_dataset)}")
        # Use four dataset items to visualize four images
        input_kspace_list = []
        target_img_list = []
        file_indices = list(range(min(4, len(test_dataset))))
        for i in file_indices:
            kspace, img = test_dataset[i]
            input_kspace_list.append(kspace.unsqueeze(0))  # [1, 2, 640, 320]
            target_img_list.append(img.unsqueeze(0))  # [1, 2, 640, 320]
        input_kspace = torch.cat(input_kspace_list, dim=0)  # [4, 2, 640, 320]
        target_img = torch.cat(target_img_list, dim=0)  # [4, 2, 640, 320]
        print(f"input_kspace shape: {input_kspace.shape}")
        print(f"target_img shape: {target_img.shape}")
        # Convert kspace to complex on CPU to avoid MPS complex tensor issue
        input_kspace = input_kspace.to('cpu')
        input_kspace_complex = input_kspace[:, 0, :, :] + 1j * input_kspace[:, 1, :, :]
        input_img_list = []
        for i in range(input_kspace_complex.size(0)):
            img_i = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(input_kspace_complex[i], dim=(-2, -1)), norm='ortho', dim=(-2, -1)), dim=(-2, -1))
            input_img_list.append(torch.stack((img_i.real, img_i.imag), dim=0))  # [2, 640, 320]
        input_img = torch.stack(input_img_list, dim=0).to(device)  # [4, 2, 640, 320]
        print(f"input_img shape: {input_img.shape}")
        mag, kspace_db, _, mag_target = compute_visualization_data(input_img, target_img, device)
        print(f"mag shape: {mag.shape}, kspace_db shape: {kspace_db.shape}, mag_target shape: {mag_target.shape}")
        save_path = os.path.join(out_dir, "zero_filled_initial.png")
        visualize_images(test_fig, mag, kspace_db, "Zero-filled Visualization — Epoch 0, Batch 0", save_path, losses=None, mag_target=mag_target)
        test_fig.show()
        plt.show(block=False)
        plt.pause(0.1)  # Allow rendering
    model.train()

# Training loop
model.train()
batch_counter = 0
viz_counter = 0
print(f"Starting training with apply_dc={apply_dc}")
for epoch in range(num_epochs):
    running_loss = 0.0
    for batch_idx, (input_kspace, target_img) in enumerate(dataloader):
        target_img = target_img.to(device)
        # Convert kspace to complex on CPU to avoid MPS complex tensor issue
        input_kspace = input_kspace.to('cpu')
        input_kspace_complex = input_kspace[:, 0, :, :] + 1j * input_kspace[:, 1, :, :]
        input_img_list = []
        for i in range(input_kspace_complex.size(0)):
            img_i = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(input_kspace_complex[i], dim=(-2, -1)), norm='ortho', dim=(-2, -1)), dim=(-2, -1))
            input_img_list.append(torch.stack((img_i.real, img_i.imag), dim=0))  # [2, 640, 320]
        input_img = torch.stack(input_img_list, dim=0).to(device)  # [batch_size, 2, 640, 320]
        optimizer.zero_grad()
        output = model(input_img, input_kspace, apply_dc=apply_dc)
        # Compute loss on real and imaginary channels
        mse_loss = mse_criterion(output, target_img).mean(dim=(1, 2, 3))
        diff = output - target_img
        mae_loss = torch.sqrt(diff[:, 0, :, :]**2 + diff[:, 1, :, :]**2).mean(dim=(1, 2))
        per_image_loss = mse_loss + mae_loss
        loss = per_image_loss.mean()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # Log loss
        with open(loss_log_path, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([epoch + 1, batch_idx, f"{loss.item():.6e}", "", ""])

        # Visualize
        if batch_counter >= viz_interval:
            viz_counter += 1
            # Test visualization
            save_path = os.path.join(out_dir, "visualizations", f"viz_epoch_{epoch+1}_batch_{batch_idx}.png") if viz_counter % save_interval == 0 else None
            with torch.no_grad():
                # Use four dataset items to visualize four images
                input_kspace_list = []
                target_img_list = []
                file_indices = list(range(min(4, len(test_dataset))))
                for i in file_indices:
                    kspace, img = test_dataset[i]
                    input_kspace_list.append(kspace.unsqueeze(0))
                    target_img_list.append(img.unsqueeze(0))
                input_kspace_test = torch.cat(input_kspace_list, dim=0)
                target_img_test = torch.cat(target_img_list, dim=0)
                if input_kspace_test.size(0) == 0:
                    print("Error: Empty batch in test visualization")
                else:
                    input_kspace_test = input_kspace_test.to('cpu')
                    input_kspace_test_complex = input_kspace_test[:, 0, :, :] + 1j * input_kspace_test[:, 1, :, :]
                    input_img_list = []
                    for i in range(input_kspace_test_complex.size(0)):
                        img_i = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(input_kspace_test_complex[i], dim=(-2, -1)), norm='ortho', dim=(-2, -1)), dim=(-2, -1))
                        input_img_list.append(torch.stack((img_i.real, img_i.imag), dim=0))  # [2, 640, 320]
                    input_img_test = torch.stack(input_img_list, dim=0).to(device)  # [4, 2, 640, 320]
                    output_test = model(input_img_test, input_kspace_test, apply_dc=apply_dc)
                    mse_loss_test = mse_criterion(output_test, target_img_test).mean(dim=(1, 2, 3))
                    diff_test = output_test - target_img_test
                    mae_loss_test = torch.sqrt(diff_test[:, 0, :, :]**2 + diff_test[:, 1, :, :]**2).mean(dim=(1, 2))
                    test_losses = (mse_loss_test + mae_loss_test).cpu().tolist()
                    mag, kspace_db, mean_ssim, mag_target = compute_visualization_data(output_test, target_img_test, device)
                    if mean_ssim is not None:
                        with open(loss_log_path, 'a', newline='') as csvfile:
                            csv_writer = csv.writer(csvfile)
                            csv_writer.writerow([epoch + 1, batch_idx, "", "", f"{mean_ssim:.6e}"])
                    visualize_images(test_fig, mag, kspace_db, f"UNet Visualization — Epoch {epoch+1}, Batch {batch_idx}", save_path, losses=test_losses, mag_target=mag_target)
                    test_fig.show()
                    plt.show(block=False)
                    plt.pause(0.1)
            # Train visualization
            train_save_path = os.path.join(out_dir, "train_visualizations", f"train_viz_epoch_{epoch+1}_batch_{batch_idx}.png") if viz_counter % save_interval == 0 else None
            with torch.no_grad():
                train_losses = per_image_loss[:4].cpu().tolist()
                mag, kspace_db, mean_ssim, mag_target = compute_visualization_data(output[:4], target_img[:4], device)
                if mean_ssim is not None:
                    with open(loss_log_path, 'a', newline='') as csvfile:
                        csv_writer = csv.writer(csvfile)
                        csv_writer.writerow([epoch + 1, batch_idx, "", "", f"{mean_ssim:.6e}"])
                visualize_images(train_fig, mag, kspace_db, f"Training Visualization — Epoch {epoch+1}, Batch {batch_idx}", train_save_path, losses=train_losses, mag_target=mag_target)
                train_fig.show()
                plt.show(block=False)
                plt.pause(0.1)
            batch_counter = 0

        if batch_idx % print_interval == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.6e}")

        batch_counter += 1

    # Log average epoch loss
    avg_loss = running_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.6e}")
    with open(loss_log_path, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([epoch + 1, "Average", "", f"{avg_loss:.6e}", ""])

    # Save model
    if (epoch + 1) % model_save_interval == 0:
        model_save_path = os.path.join(out_dir, "models", f"model_{epoch+1}.pth")
        torch.save(model.state_dict(), model_save_path)
        print(f"Saved model to {model_save_path}")

# Save final model
torch.save(model.state_dict(), os.path.join(out_dir, "unet_trained.pth"))
print("Model saved to unet_trained.pth")

# Created on 07:47 AM MDT, Tuesday, September 02, 2025, Artifact ID: 86bff4e4-e8fe-4901-87ec-42b11b9bbcea

