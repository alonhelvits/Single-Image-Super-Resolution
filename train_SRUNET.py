# Linoy Ketashvili - 316220235
# Alon Helvits - 315531087
import os
import random
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torchvision.transforms.functional as Func  # Correct import for rotate
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from models import SmallUNet_4X
from unet_utilts import *
from data_classes import DIV2KDataset512

LEARNING_RATE = 1e-3
EPOCHS = 15
BATCH_SIZE = 8
WEIGHT_DECAY = 1e-6
SCALE = 4



def train(model, trainloader, validloader, criterion, optimizer, num_epochs, device='cpu', use_wandb=False):
    model.train()
    
    # Learning rate scheduler
    scheduler = StepLR(optimizer, step_size=60, gamma=0.1)
    
    # Lists to store metrics
    loss_list = []
    val_loss_list = []
    ssim_sr_list = []
    psnr_sr_list = []
    ssim_lr_list = []
    psnr_lr_list = []
    train_ssim_sr_list = []
    train_psnr_sr_list = []
    train_ssim_lr_list = []
    train_psnr_lr_list = []
    max_valid_psnr = 0.0
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        for hr_images, lr_images in tqdm(trainloader):
            # Move the data to the appropriate device (GPU or CPU)
            lr_images = lr_images.to(device)
            hr_images = hr_images.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass: Get the model output
            outputs = model(lr_images)
            
            # Calculate the loss
            loss = criterion(outputs, hr_images)
            
            # Backward pass: Compute the gradients
            loss.backward()
            
            # Update the weights
            optimizer.step()
            
            # Accumulate the loss
            running_loss += loss.item() * lr_images.size(0)
        
        # Step the scheduler after each epoch
        scheduler.step()
        

        # Compute the average loss over the epoch
        epoch_loss = running_loss / len(trainloader.dataset)
        epoch_val_loss = calculate_val_loss(model, validloader, criterion, device)
        val_loss_list.append(epoch_val_loss)
        loss_list.append(epoch_loss)
        
        # Evaluate the model on the validation set
        avg_ssim_sr, avg_psnr_sr, avg_ssim_lr, avg_psnr_lr = calculate_ssim_psnr(model, validloader, device)
        if avg_psnr_sr > max_valid_psnr:
            max_valid_psnr = avg_psnr_sr
            torch.save(model.state_dict(), f'Best_SRUNET_4X.pth')
        # Evalute the model on the training set
        # Get the size of validloader
        validloader_size = len(validloader.dataset)

        # Sample the trainloader
        sampled_trainloader = sample_trainloader(trainloader, validloader_size)

        # Evaluate the model on the training set using the sampled trainloader
        avg_ssim_sr_train, avg_psnr_sr_train, avg_ssim_lr_train, avg_psnr_lr_train = calculate_ssim_psnr(model, sampled_trainloader, device)
        
        # Append validation metrics to their respective lists
        ssim_sr_list.append(avg_ssim_sr)
        psnr_sr_list.append(avg_psnr_sr)
        ssim_lr_list.append(avg_ssim_lr)
        psnr_lr_list.append(avg_psnr_lr)
        train_ssim_sr_list.append(avg_ssim_sr_train)
        train_psnr_sr_list.append(avg_psnr_sr_train)
        train_ssim_lr_list.append(avg_ssim_lr_train)
        train_psnr_lr_list.append(avg_psnr_lr_train)

        # Log metrics to wandb
        if use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'epoch_loss': epoch_loss,
                'epoch_val_loss': epoch_val_loss,
                'avg_ssim_sr': avg_ssim_sr,
                'avg_psnr_sr': avg_psnr_sr,
                'avg_ssim_lr': avg_ssim_lr,
                'avg_psnr_lr': avg_psnr_lr,
                'train_avg_ssim_sr': avg_ssim_sr_train,
                'train_avg_psnr_sr': avg_psnr_sr_train,
                'train_avg_ssim_lr': avg_ssim_lr_train,
                'train_avg_psnr_lr': avg_psnr_lr_train,
                'learning_rate': optimizer.param_groups[0]["lr"]
            })

        # Print metrics
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        print(f'Epoch {epoch+1}/{num_epochs}, Valid Loss: {epoch_val_loss:.4f}')
        print(f'Validation SSIM (SR): {avg_ssim_sr:.4f}, PSNR (SR): {avg_psnr_sr:.4f}')
        print(f'Validation SSIM (LR): {avg_ssim_lr:.4f}, PSNR (LR): {avg_psnr_lr:.4f}')
        
        # Save checkpoint every few epochs
        # if epoch % 9 == 0:
        #     torch.save(model.state_dict(), f"checkpoints_combined_500_on_man_made_val/unet_model_epoch_{epoch}_loss_{epoch_loss}_avg_sr_psnr_{avg_psnr_sr}_avg_sr_ssim_{avg_ssim_sr}.pth")
    
    # Log final results and finish the run
    if use_wandb: wandb.finish()

    # Create a figure with 3 subplots (1 row, 3 columns)
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # Plot Training Loss
    axs[0].plot(range(1, num_epochs + 1), loss_list, label='Training Loss', color='blue')
    axs[0].plot(range(1, num_epochs + 1), val_loss_list, label='Validation Loss', color='green')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Training Loss Over Epochs')
    axs[0].legend()

    # Plot Validation and train SSIM for SR and LR
    axs[1].plot(range(1, num_epochs + 1), ssim_sr_list, label='Validation SSIM (SR)', color='green')
    axs[1].plot(range(1, num_epochs + 1), ssim_lr_list, label='Validation SSIM (LR)', color='red')
    axs[1].plot(range(1, num_epochs + 1), train_ssim_sr_list, label='Train SSIM (SR)', color='blue')
    axs[1].plot(range(1, num_epochs + 1), train_ssim_lr_list, label='Train SSIM (LR)', color='orange')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('SSIM')
    axs[1].set_title('Validation and Train SSIM Over Epochs')
    axs[1].legend()

    # Plot Validation and Train PSNR for SR and LR
    axs[2].plot(range(1, num_epochs + 1), psnr_sr_list, label='Validation PSNR (SR)', color='purple')
    axs[2].plot(range(1, num_epochs + 1), psnr_lr_list, label='Validation PSNR (LR)', color='orange')
    axs[2].plot(range(1, num_epochs + 1), train_psnr_sr_list, label='Train PSNR (SR)', color='blue')
    axs[2].plot(range(1, num_epochs + 1), train_psnr_lr_list, label='Train PSNR (LR)', color='red')
    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('PSNR')
    axs[2].set_title('Validation and Train PSNR Over Epochs')
    axs[2].legend()

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the figure to the project folder
    plt.savefig('training_validation_metrics_4X_combined_500_cropped_on_man_made_val.png')
    plt.show()
    print("Finished Training")
    # Save lists to a file if needed
    # results = {
    #     'loss': loss_list,
    #     'ssim_sr': ssim_sr_list,
    #     'psnr_sr': psnr_sr_list,
    #     'ssim_lr': ssim_lr_list,
    #     'psnr_lr': psnr_lr_list,
    # }
    #torch.save(results, 'training_metrics_4X_man_made_updated.pth')


# Define a function for random rotation
def random_rotation(img):
    return Func.rotate(img, random.choice([0, 90, 270]))


def main(data_path='', use_wandb=False):
    config = {
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "optimizer": "Adam",
        "weight_decay": WEIGHT_DECAY,
        "scale": SCALE
    }
    if use_wandb:
        wandb.init(project="super_resolution_project", name=f'combined_500_{config["epochs"]}_epochs_man_made_validation',config=config)
    
    # Define the transformation: Resize the image to 512x512
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Updated transformation pipeline
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip horizontally with a probability of 0.5
        transforms.Lambda(random_rotation),  # Apply the random rotation function
        transforms.ToTensor(),
    ])

    train_dir = os.path.join(data_path, 'train')
    valid_dir = os.path.join(data_path, 'val')

    trainset = DIV2KDataset512(image_dir=train_dir, scale=4, transform=transform)
    validset = DIV2KDataset512(image_dir=valid_dir, scale=4, transform=transform)

    trainloader = DataLoader(trainset, batch_size=8, shuffle=True, num_workers=1)
    validloader = DataLoader(validset, batch_size=8, shuffle=False, num_workers=1)
    print(f"Number of training batches: {len(trainloader)}")
    print(f"Number of validation batches: {len(validloader)}")

    #show_random_image(trainloader)

    # Initialize the model
    model = SmallUNet_4X()
    #model.load_state_dict(torch.load('best_unet_model.pth'))
    # Move the model to GPU if available
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model.to(device)

    # Define the loss function (Mean Squared Error is common for super-resolution)
    criterion = nn.L1Loss()

    # Define the optimizer (Adam is commonly used)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)

    train(model, trainloader, validloader, criterion, optimizer, config['epochs'], device, use_wandb)

    #torch.save(model.state_dict(), 'best_unet_model_4X_man_made_updated.pth')

    # Evaluation mode
    model.eval()

    # Evaluate on a few test images from the dataloader (optional)
    with torch.no_grad():
        for hr_images, lr_images in trainloader:
            lr_images = lr_images.to(device)
            hr_images = hr_images.to(device)
            
            # Get the model's output
            outputs = model(lr_images)
            
            # Randomly select an image from the batch
            random_idx = random.randint(0, lr_images.size(0) - 1)
            
            # Convert tensors to numpy arrays for plotting
            lr_image = lr_images[random_idx].cpu().numpy().transpose(1, 2, 0)
            hr_image = hr_images[random_idx].cpu().numpy().transpose(1, 2, 0)
            output_image = outputs[random_idx].cpu().numpy().transpose(1, 2, 0)
            
            # Plot the images side by side
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].imshow(lr_image)
            axes[0].set_title('Low-Resolution (Input)')
            axes[0].axis('off')
            
            axes[1].imshow(hr_image)
            axes[1].set_title('High-Resolution (Ground Truth)')
            axes[1].axis('off')
            
            axes[2].imshow(output_image)
            axes[2].set_title('Model Output')
            axes[2].axis('off')
            
            plt.show()
            
            break  # Just one batch for quick evaluation

# --------------------- Main --------------------- #
if __name__ == '__main__':
    main()
