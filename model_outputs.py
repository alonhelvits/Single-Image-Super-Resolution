from models import SmallUNet_4X
from data_classes import DIV2KDataset512Test
from unet_utilts import calculate_ssim_psnr_for_single_image
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch
import numpy as np

if __name__ == "__main__":
    # Define the transformation: Resize the image to 512x512
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Instantiate the dataset and dataloader
    test_dir = 'dataset/images_for_report'
    testset = DIV2KDataset512Test(image_dir=test_dir, scale=4, transform=transform)
    testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=1)
    #print(f"Number of validation batches: {len(testloader)}")
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = SmallUNet_4X()
    model.load_state_dict(torch.load('models_weights/baseline_model_L1_4X_psnr_27.378690308017333_ssim_0.7839492559432983.pth', map_location=torch.device('cpu')))
    model = model.to(device)

    # Evaluation mode
    model.eval()

    # Lists to store the top 10 images
    psnr_gaps = []
    best_psnrs = []

    # Evaluate on test images
    with torch.no_grad():
        for hr_images, lr_images, _ in testloader:
            lr_images = lr_images.to(device)
            hr_images = hr_images.to(device)
            
            # Get the model's output
            outputs = model(lr_images)
            
            # Calculate SSIM and PSNR
            ssim_sr, psnr_sr, ssim_lr, psnr_lr = calculate_ssim_psnr_for_single_image(hr_images, lr_images, outputs)

            # Track the top 10 images with the maximum PSNR gap
            psnr_gap = psnr_sr - psnr_lr
            psnr_gaps.append((psnr_gap, lr_images.cpu().numpy(), outputs.cpu().numpy(), hr_images.cpu().numpy(), ssim_lr, psnr_lr, ssim_sr, psnr_sr))
            psnr_gaps.sort(reverse=True, key=lambda x: x[0])
            psnr_gaps = psnr_gaps[:10]

            # Track the top 10 images with the highest PSNR_SR
            best_psnrs.append((psnr_sr, lr_images.cpu().numpy(), outputs.cpu().numpy(), hr_images.cpu().numpy(), ssim_lr, psnr_lr, ssim_sr, psnr_sr))
            best_psnrs.sort(reverse=True, key=lambda x: x[0])
            best_psnrs = best_psnrs[:10]

    # Save and plot the images
    def save_and_plot_images(lr_image, sr_image, hr_image, ssim_lr, psnr_lr, ssim_sr, psnr_sr, filename):
        lr_image = lr_image.transpose(1, 2, 0)
        sr_image = sr_image.transpose(1, 2, 0)
        hr_image = hr_image.transpose(1, 2, 0)
        
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        
        axs[0].imshow(lr_image, cmap='gray')
        axs[0].set_title(f"LR Image\nSSIM: {ssim_lr:.4f}\nPSNR: {psnr_lr:.4f}")
        axs[0].axis('off')
        
        axs[1].imshow(sr_image, cmap='gray')
        axs[1].set_title(f"SR Image\nSSIM: {ssim_sr:.4f}\nPSNR: {psnr_sr:.4f}")
        axs[1].axis('off')
        
        axs[2].imshow(hr_image, cmap='gray')
        axs[2].set_title("HR Image")
        axs[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(filename)
        plt.show()

    # Save and plot the top 10 images with the maximum PSNR gap
    for i, (psnr_gap, lr_image, sr_image, hr_image, ssim_lr, psnr_lr, ssim_sr, psnr_sr) in enumerate(psnr_gaps):
        save_and_plot_images(lr_image[0], sr_image[0], hr_image[0], ssim_lr, psnr_lr, ssim_sr, psnr_sr, f"max_psnr_gap_{i+1}.png")

    # Save and plot the top 10 images with the highest PSNR_SR
    for i, (psnr_sr, lr_image, sr_image, hr_image, ssim_lr, psnr_lr, ssim_sr, psnr_sr) in enumerate(best_psnrs):
        save_and_plot_images(lr_image[0], sr_image[0], hr_image[0], ssim_lr, psnr_lr, ssim_sr, psnr_sr, f"best_psnr_sr_{i+1}.png")
