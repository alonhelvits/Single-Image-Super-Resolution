# Linoy Ketashvili - 316220235
# Alon Helvits - 315531087
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class CustomResNet18(nn.Module):
    def __init__(self, num_classes):
        super(CustomResNet18, self).__init__()
        # Load the pre-trained ResNet-18 model
        self.resnet = models.resnet18(pretrained=True)
        # Replace the final fully connected layer
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        # Forward pass through the ResNet-18 model
        return self.resnet(x)

class SmallUNet_4X(nn.Module):  # Updated to 4X
    def __init__(self):
        super(SmallUNet_4X, self).__init__()
        
        # Downsampling path (Encoder)
        self.encoder1 = self.conv_block(3, 32)
        self.encoder2 = self.conv_block(32, 64)
        self.encoder3 = self.conv_block(64, 128)
        self.encoder4 = self.conv_block(128, 256)
        self.encoder5 = self.conv_block(256, 512)

        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)

        # Upsampling path (Decoder)
        self.upconv5 = self.upconv(1024, 512)
        self.decoder5 = self.conv_block(1024, 512)
        self.upconv4 = self.upconv(512, 256)
        self.decoder4 = self.conv_block(512, 256)
        self.upconv3 = self.upconv(256, 128)
        self.decoder3 = self.conv_block(256, 128)
        self.upconv2 = self.upconv(128, 64)
        self.decoder2 = self.conv_block(128, 64)
        self.upconv1 = self.upconv(64, 32)
        self.decoder1 = self.conv_block(64, 32)
        
        # Final output layer of the U-Net (256x256 output)
        self.final_conv_unet = nn.Conv2d(32, 3, kernel_size=1)

        # Additional upsampling layers to achieve 512x512 output
        self.upconv_extra1 = self.upconv(32, 16)  # Upsample from 128x128 to 256x256
        self.decoder_extra1 = self.conv_block(16, 16)

        # Additional upsampling layer for 512x512 output
        self.upconv_extra2 = self.upconv(16, 8)  # Upsample from 256x256 to 512x512
        self.decoder_extra2 = self.conv_block(8, 8)

        # Final output layer to match the number of output channels
        self.final_conv = nn.Conv2d(8, 3, kernel_size=1)
    
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def upconv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    
    def forward(self, x):
        # Downsampling
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool2d(enc1, 2))
        enc3 = self.encoder3(F.max_pool2d(enc2, 2))
        enc4 = self.encoder4(F.max_pool2d(enc3, 2))
        enc5 = self.encoder5(F.max_pool2d(enc4, 2))
        
        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc5, 2))
        
        # Upsampling
        dec5 = self.upconv5(bottleneck)
        dec5 = torch.cat((dec5, enc5), dim=1)
        dec5 = self.decoder5(dec5)

        dec4 = self.upconv4(dec5)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        # Final U-Net output (256x256)
        out_unet = self.final_conv_unet(dec1)
        
        # # Additional upsampling to 256x256
        # up1 = self.upconv_extra1(out_unet)
        # up1 = self.decoder_extra1(up1)

        # # Additional upsampling to 512x512
        # up2 = self.upconv_extra2(up1)
        # up2 = self.decoder_extra2(up2)
        
        # # Final output layer
        # out = self.final_conv(up2)
        
        # Apply ReLU to ensure non-negative pixel values
        #out = F.relu(out_unet)
        
        return out_unet