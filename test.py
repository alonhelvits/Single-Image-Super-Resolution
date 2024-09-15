# Linoy Ketashvili - 316220235
# Alon Helvits - 315531087
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from unet_utilts import calculate_ssim_psnr_for_single_image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from data_classes import DIV2KDataset512Test
from models import CustomResNet18, SmallUNet_4X
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# ---------------------- General definitions ----------------------
SR_test_transform = transforms.Compose([
    transforms.ToTensor(),
])

classifier_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.4518,0.4355,0.4003], std=[0.0506,0.0454,0.0471])
])

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


# Define the fixed class names and their labels
CLASSES = ['man_made', 'nature']
CLASS_TO_LABEL = {cls: i for i, cls in enumerate(CLASSES)}
LABEL_TO_CLASS = {i: cls for cls, i in CLASS_TO_LABEL.items()}

# ---------------------- General functions ----------------------
def calculate_ssim_psnr(model, dataloader, device):
    model.eval()  # Set the model to evaluation mode
    total_ssim_sr = 0.0
    total_psnr_sr = 0.0
    total_ssim_lr = 0.0
    total_psnr_lr = 0.0
    count = 0

    with torch.no_grad():
        for hr_images, lr_images, _ in dataloader:
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



# ---------------------- Pipeline functions ----------------------
def new_pipeline(classifier_model, nature_model, manmande_model, testloader, device):
    # Evaluation models
    classifier_model.eval()
    nature_model.eval()
    manmande_model.eval()
    ssim_sr, psnr_sr, ssim_lr, psnr_lr = 0.0, 0.0, 0.0, 0.0
    ssim_sr_acc, psnr_sr_acc, ssim_lr_acc, psnr_lr_acc = 0.0, 0.0, 0.0, 0.0
    ds_length = len(testloader.dataset)
    with torch.no_grad():
        for hr_images, lr_images, cl_image in testloader:
            # Move images to the correct device
            lr_images = lr_images.to(device)
            hr_images = hr_images.to(device)
            cl_image = cl_image.to(device)

            # Classify the image
            output = classifier_model(cl_image)
            _, predicted = torch.max(output, 1)

            # Select the correct model based on the classification
            sr_model = nature_model if predicted.item() == CLASS_TO_LABEL['nature'] else manmande_model
            
            # Generate the super-resolved images
            sr_images = sr_model(lr_images)

            # Evaluate the super-resolved images
            ssim_sr, psnr_sr, ssim_lr, psnr_lr = calculate_ssim_psnr_for_single_image(hr_images, lr_images, sr_images)
            ssim_sr_acc += ssim_sr
            psnr_sr_acc += psnr_sr
            ssim_lr_acc += ssim_lr
            psnr_lr_acc += psnr_lr
            
    avg_ssim_sr = ssim_sr_acc / ds_length
    avg_psnr_sr = psnr_sr_acc / ds_length
    avg_ssim_lr = ssim_lr_acc / ds_length
    avg_psnr_lr = psnr_lr_acc / ds_length
    
    print(f"New Model L1 - Average SSIM (SR): {avg_ssim_sr:.4f}")
    print(f"New Model L1 - Average PSNR (SR): {avg_psnr_sr:.4f}")
    print(f"New Model L1 - Average SSIM (LR): {avg_ssim_lr:.4f}")
    print(f"New Model L1 - Average PSNR (LR): {avg_psnr_lr:.4f}")
    
    return avg_ssim_sr, avg_psnr_sr, avg_ssim_lr, avg_psnr_lr

def baseline_pipeline(baseline_model, testloader, device):
    # Evaluation data

    baseline_model.eval()
    avg_ssim_sr, avg_psnr_sr, avg_ssim_lr, avg_psnr_lr = calculate_ssim_psnr(baseline_model, testloader, device)

    print(f"Baseline Model - Average SSIM (SR): {avg_ssim_sr:.4f}")
    print(f"Baseline Model - Average PSNR (SR): {avg_psnr_sr:.4f}")
    print(f"Baseline Model - Average SSIM (LR): {avg_ssim_lr:.4f}")
    print(f"Baseline Model - Average PSNR (LR): {avg_psnr_lr:.4f}")
    
    return avg_ssim_sr, avg_psnr_sr, avg_ssim_lr, avg_psnr_lr

def main(test_path='', model_path='', model_type='baseline'):
    # Define the transformation: Resize the image to 512x512

    # Create Test Loader
    test_dir = os.path.join(test_path, 'test')
    testset = DIV2KDataset512Test(image_dir=test_dir, scale=4, transform=SR_test_transform, classifier_transform=classifier_transform)
    testloader = DataLoader(testset, batch_size=1, shuffle=True, num_workers=1)
    if model_type == 'baseline':
        baseline_SR_model = SmallUNet_4X()
        baseline_SR_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        baseline_SR_model.to(device)
        baseline_pipeline(baseline_SR_model, testloader, device)
    elif model_type == 'CBSR':
        classifier_model = CustomResNet18(num_classes=2)
        nature_SR_model = SmallUNet_4X()
        man_made_SR_model = SmallUNet_4X()
        nature_SR_model.load_state_dict(torch.load('nature_800_model.pth', map_location=torch.device('cpu')))
        man_made_SR_model.load_state_dict(torch.load('man_made_800_model.pth', map_location=torch.device('cpu')))
        classifier_model = torch.load('classifier_100_epochs.pth', map_location=torch.device('cpu'))

        nature_SR_model.to(device)
        man_made_SR_model.to(device)
        classifier_model.to(device)

        new_pipeline(classifier_model, nature_SR_model, man_made_SR_model, testloader, device)


if __name__ == '__main__':
    main()