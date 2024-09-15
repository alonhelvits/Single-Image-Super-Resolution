# Linoy Ketashvili - 316220235
# Alon Helvits - 315531087
import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image


class DIV2KDataset512(Dataset):
    def __init__(self, image_dir, scale=2, transform=None):
        self.image_dir = image_dir
        self.scale = scale
        self.transform = transform
        self.image_files = [os.path.join(image_dir, file) for file in os.listdir(image_dir) if file.endswith(('png', 'jpg', 'jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        hr_image = Image.open(img_path).convert("RGB")
        
        # Create the low-resolution image by further downscaling the 512x512 HR image
        lr_image = hr_image.resize((512 // self.scale, 512 // self.scale), Image.BICUBIC)
        lr_image = lr_image.resize((512, 512), Image.BICUBIC)
        
        # Apply the same transformation to both HR and LR images
        if self.transform:
            seed = torch.seed()  # Get a random seed
            random.seed(seed)  # Apply the seed before transforming HR
            torch.manual_seed(seed)  # Apply the seed before transforming HR
            hr_image = self.transform(hr_image)
            
            random.seed(seed)  # Reset the seed before transforming LR
            torch.manual_seed(seed)  # Reset the seed before transforming LR
            lr_image = self.transform(lr_image)
        
        return hr_image, lr_image
    
class DIV2KDataset512Test(Dataset):
    def __init__(self, image_dir, scale=2, transform=None, classifier_transform=None):
        self.image_dir = image_dir
        self.scale = scale
        self.transform = transform
        self.classifier_transform = classifier_transform
        self.image_files = [os.path.join(image_dir, file) for file in os.listdir(image_dir) if file.endswith(('png', 'jpg', 'jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        hr_image = Image.open(img_path).convert("RGB")
        
        
        # Create the low-resolution image by further downscaling the 512x512 HR image
        lr_image = hr_image.resize((512 // self.scale, 512 // self.scale), Image.BICUBIC)
        lr_image = lr_image.resize((512, 512), Image.BICUBIC)
        cl_image = hr_image.resize((256,256), Image.BICUBIC)

        # Apply the classifier_transform to the resized HR image
        if self.classifier_transform:
            cl_image = self.classifier_transform(cl_image)
        else:
            cl_image = self.transform(cl_image)
        
        # Apply the same transformation to both HR and LR images
        if self.transform:
            seed = torch.seed()  # Get a random seed
            random.seed(seed)  # Apply the seed before transforming HR
            torch.manual_seed(seed)  # Apply the seed before transforming HR
            hr_image = self.transform(hr_image)
            
            random.seed(seed)  # Reset the seed before transforming LR
            torch.manual_seed(seed)  # Reset the seed before transforming LR
            lr_image = self.transform(lr_image)
        
        return hr_image, lr_image, cl_image