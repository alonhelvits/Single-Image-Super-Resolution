import torch
import random
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# --------------------- Functions --------------------- #
def show_random_image(dataloader):
    # Get a random batch of images from the DataLoader
    batch = next(iter(dataloader))
    
    # Unpack the batch into high-resolution and low-resolution images
    hr_batch, lr_batch = batch
    
    # Choose a random image from the batch
    random_idx = random.randint(0, hr_batch.size(0) - 1)
    hr_image = hr_batch[random_idx]
    lr_image = lr_batch[random_idx]
    
    # Convert the image tensors to NumPy arrays and transpose the axes to (H, W, C)
    hr_image = hr_image.permute(1, 2, 0).numpy()
    lr_image = lr_image.permute(1, 2, 0).numpy()
    
    # Display the images side by side using matplotlib
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(hr_image)
    axes[0].set_title('High Resolution')
    axes[0].axis('off')  # Turn off axis labels
    
    axes[1].imshow(lr_image)
    axes[1].set_title('Low Resolution')
    axes[1].axis('off')  # Turn off axis labels
    
    plt.show()

# Define a function to sample from trainloader
def sample_trainloader(trainloader, sample_size):
    indices = random.sample(range(len(trainloader.dataset)), sample_size)
    sampled_data = torch.utils.data.Subset(trainloader.dataset, indices)
    return torch.utils.data.DataLoader(sampled_data, batch_size=trainloader.batch_size, shuffle=False)

def calculate_ssim_psnr(model, dataloader, device):
    model.eval()  # Set the model to evaluation mode
    total_ssim_sr = 0.0
    total_psnr_sr = 0.0
    total_ssim_lr = 0.0
    total_psnr_lr = 0.0
    count = 0

    with torch.no_grad():
        for hr_images, lr_images in dataloader:
            # Move images to the correct device
            lr_images = lr_images.to(device)
            hr_images = hr_images.to(device)
            
            # Generate the super-resolved images
            sr_images = model(lr_images)
            
            # Move tensors back to the CPU for SSIM and PSNR calculation
            sr_images = sr_images.cpu().numpy().transpose(0, 2, 3, 1)  # Convert to (batch, H, W, C)
            hr_images = hr_images.cpu().numpy().transpose(0, 2, 3, 1)  # Convert to (batch, H, W, C)
            lr_images = lr_images.cpu().numpy().transpose(0, 2, 3, 1)  # Convert to (batch, H, W, C)
            
            for i in range(sr_images.shape[0]):  # Iterate over the batch
                sr_img = sr_images[i]
                hr_img = hr_images[i]
                lr_img = lr_images[i]
                # lr_img_resized = resize(lr_img, (lr_img.shape[0] * 4, lr_img.shape[1] * 4), 
                #         anti_aliasing=True)
                
                # Calculate SSIM and PSNR for SR images
                ssim_value_sr = ssim(hr_img, sr_img, data_range=sr_img.max() - sr_img.min(), channel_axis=-1)
                psnr_value_sr = psnr(hr_img, sr_img, data_range=sr_img.max() - sr_img.min())
                
                total_ssim_sr += ssim_value_sr
                total_psnr_sr += psnr_value_sr

                # Calculate SSIM and PSNR for LR images
                ssim_value_lr = ssim(hr_img, lr_img, data_range=lr_img.max() - lr_img.min(), channel_axis=-1)
                psnr_value_lr = psnr(hr_img, lr_img, data_range=lr_img.max() - lr_img.min())
                
                total_ssim_lr += ssim_value_lr
                total_psnr_lr += psnr_value_lr

                count += 1

    # Calculate average SSIM and PSNR
    avg_ssim_sr = total_ssim_sr / count
    avg_psnr_sr = total_psnr_sr / count
    avg_ssim_lr = total_ssim_lr / count
    avg_psnr_lr = total_psnr_lr / count

    return avg_ssim_sr, avg_psnr_sr, avg_ssim_lr, avg_psnr_lr

def calculate_ssim_psnr_for_single_image(hr_image, lr_image, sr_image):
    # Remove the batch dimension if batch_size is 1
    if hr_image.ndim == 4 and hr_image.shape[0] == 1:
        hr_image = hr_image.squeeze(0)  # Convert from (1, C, H, W) to (C, H, W)
    if lr_image.ndim == 4 and lr_image.shape[0] == 1:
        lr_image = lr_image.squeeze(0)
    if sr_image.ndim == 4 and sr_image.shape[0] == 1:
        sr_image = sr_image.squeeze(0)

    # Convert tensors to numpy arrays and rearrange to (H, W, C) if necessary
    if isinstance(hr_image, torch.Tensor):
        hr_image = hr_image.cpu().numpy()
    if isinstance(lr_image, torch.Tensor):
        lr_image = lr_image.cpu().numpy()
    if isinstance(sr_image, torch.Tensor):
        sr_image = sr_image.cpu().numpy()

    # Check if the images are in (C, H, W) format, if so, transpose to (H, W, C)
    if hr_image.shape[0] == 3:  # Assuming color images
        hr_image = hr_image.transpose(1, 2, 0)  # Convert to (H, W, C)
    if lr_image.shape[0] == 3:
        lr_image = lr_image.transpose(1, 2, 0)
    if sr_image.shape[0] == 3:
        sr_image = sr_image.transpose(1, 2, 0)
    
    # Calculate SSIM and PSNR for SR images
    ssim_sr = ssim(hr_image, sr_image, data_range=sr_image.max() - sr_image.min(), channel_axis=-1)
    psnr_sr = psnr(hr_image, sr_image, data_range=sr_image.max() - sr_image.min())
    
    # Calculate SSIM and PSNR for LR images
    ssim_lr = ssim(hr_image, lr_image, data_range=lr_image.max() - lr_image.min(), channel_axis=-1)
    psnr_lr = psnr(hr_image, lr_image, data_range=lr_image.max() - lr_image.min())
    
    return ssim_sr, psnr_sr, ssim_lr, psnr_lr

def calculate_val_loss(model, validloader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for hr_images, lr_images in validloader:
            # Move images to the correct device
            lr_images = lr_images.to(device)
            hr_images = hr_images.to(device)
            
            # Generate the super-resolved images
            sr_images = model(lr_images)
            
            # Calculate the loss
            loss = criterion(sr_images, hr_images)
            total_loss += loss.item() * lr_images.size(0)
            count += lr_images.size(0)

    # Calculate the average loss
    avg_loss = total_loss / count

    return avg_loss
