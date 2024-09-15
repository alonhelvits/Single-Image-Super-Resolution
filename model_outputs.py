from models import SmallUNet_4X
from data_classes import DIV2KDataset512Test
from unet_utilts import calculate_ssim_psnr_for_single_image
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch
import numpy as np
import os

IMAGES_TO_PLOT = 5

def main(images_dir = '', model_path = ''):
    # Define the transformation: Resize the image to 512x512
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Instantiate the dataset and dataloader
    test_dir = os.path.join(images_dir, 'test')
    testset = DIV2KDataset512Test(image_dir=test_dir, scale=4, transform=transform)
    testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=1)
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = SmallUNet_4X()
    #model_2 = SmallUNet_4X()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    #model_2.load_state_dict(torch.load('models_weights/baseline_model_L2_4X_psnr_27.48844551738096_ssim_0.7799029941360156.pth', map_location=torch.device('cpu')))
    model = model.to(device)
    #model_2 = model_2.to(device)

    # Evaluation mode
    images_counter = 0
    model.eval()
    #model_2.eval()

    # Evaluate on test images and plot the results
    with torch.no_grad():
        print("Plotting 5 images...")
        for hr_images, lr_images, _ in testloader:
            lr_images = lr_images.to(device)
            hr_images = hr_images.to(device)
            
            # Get the model's output
            outputs = model(lr_images)
            #outputs_2 = model_2(lr_images)
            # Calculate SSIM and PSNR
            ssim_sr, psnr_sr, ssim_lr, psnr_lr = calculate_ssim_psnr_for_single_image(hr_images, lr_images, outputs)
            #ssim_sr_2, psnr_sr_2, ssim_lr_2, psnr_lr_2 = calculate_ssim_psnr_for_single_image(hr_images, lr_images, outputs_2)

            # Plot the images side by side (LR, L1-SR, L2-SR, HR)
            for i in range(len(lr_images)):
                fig, axs = plt.subplots(1, 3, figsize=(20, 5))
                
                # Plot low-resolution image
                axs[0].imshow(lr_images[i].permute(1, 2, 0).cpu().numpy())
                axs[0].set_title(f"Low-Resolution\nSSIM: {ssim_lr:.4f}\nPSNR: {psnr_lr:.4f}")
                axs[0].axis('off')
                
                # Plot L1 SR image (outputs from model 1)
                axs[1].imshow(outputs[i].permute(1, 2, 0).cpu().numpy())
                axs[1].set_title(f"Baseline SR Image\nSSIM: {ssim_sr:.4f}\nPSNR: {psnr_sr:.4f}")
                axs[1].axis('off')
                
                # Plot L2 SR image (outputs from model 2)
                # axs[2].imshow(outputs_2[i].permute(1, 2, 0).cpu().numpy())
                # axs[2].set_title(f"L2 SR Image\nSSIM: {ssim_sr_2:.4f}\nPSNR: {psnr_sr_2:.4f}")
                # axs[2].axis('off')
                
                # Plot high-resolution ground truth image
                axs[2].imshow(hr_images[i].permute(1, 2, 0).cpu().numpy())
                axs[2].set_title("Ground Truth (HR)")
                axs[2].axis('off')
                
                plt.tight_layout()
                plt.show()

                # Now plot the zoomed-in sections (same area for all images)
                zoom_start_x, zoom_start_y = 224, 224  # Arbitrary coordinates for the zoomed area

                # Extract 64x64 zoomed-in sections from the same coordinates
                lr_zoom = lr_images[i, :, zoom_start_x:zoom_start_x+64, zoom_start_y:zoom_start_y+64]
                sr_zoom_L1 = outputs[i, :, zoom_start_x:zoom_start_x+64, zoom_start_y:zoom_start_y+64]
                #sr_zoom_L2 = outputs_2[i, :, zoom_start_x:zoom_start_x+64, zoom_start_y:zoom_start_y+64]
                hr_zoom = hr_images[i, :, zoom_start_x:zoom_start_x+64, zoom_start_y:zoom_start_y+64]

                # Plot the zoomed-in sections
                fig, axs = plt.subplots(1, 3, figsize=(20, 5))
                
                # Plot zoomed low-resolution image
                axs[0].imshow(lr_zoom.permute(1, 2, 0).cpu().numpy())
                axs[0].set_title(f"Zoomed Low-Resolution\nSSIM: {ssim_lr:.4f}\nPSNR: {psnr_lr:.4f}")
                axs[0].axis('off')
                
                # Plot zoomed L1 SR image
                axs[1].imshow(sr_zoom_L1.permute(1, 2, 0).cpu().numpy())
                axs[1].set_title(f"Zoomed L1 SR Image\nSSIM: {ssim_sr:.4f}\nPSNR: {psnr_sr:.4f}")
                axs[1].axis('off')
                
                # Plot zoomed L2 SR image
                # axs[2].imshow(sr_zoom_L2.permute(1, 2, 0).cpu().numpy())
                # axs[2].set_title(f"Zoomed L2 SR Image\nSSIM: {ssim_sr_2:.4f}\nPSNR: {psnr_sr_2:.4f}")
                # axs[2].axis('off')
                
                # Plot zoomed high-resolution ground truth image
                axs[2].imshow(hr_zoom.permute(1, 2, 0).cpu().numpy())
                axs[2].set_title("Zoomed Ground Truth (HR)")
                axs[2].axis('off')
                
                plt.tight_layout()
                plt.show()
            if images_counter == IMAGES_TO_PLOT:
                print("Finished plotting images.")
                break
            images_counter += 1

if __name__ == "__main__":
    main()
