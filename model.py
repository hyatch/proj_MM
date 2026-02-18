import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_dims, out_dims):
        super().__init__()
        self.doubleconv = nn.Sequential(
            nn.Conv2d(in_dims, out_dims, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(out_dims),
            nn.ReLU(True),
            nn.Conv2d(out_dims, out_dims, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(out_dims),
            nn.ReLU(True)
        )
    def forward(self, x):
        return self.doubleconv(x)

class UNet(nn.Module):
    def __init__(self, in_dims):
        super().__init__()
        self.receive = DoubleConv(in_dims, 64) #(2048, 2048, 1) -> (2048, 2048, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128)) # (1024, 1024, 128)
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256)) # (512, 512, 256)
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512)) # (256, 256, 512)
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024)) # (128, 128, 1024)
        self.up1 = nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = True) # (256, 256, 1024)
        # skip connect + 512
        self.conv1 = DoubleConv(1024+512, 512)
        self.up2 = nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = True) #(512,512,512)
        # skip connect to +256
        self.conv2 = DoubleConv(512+256, 256)
        self.up3 = nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = True) # (1024,1024,256)
        # skip connect + 128
        self.conv3 = DoubleConv(256+128, 128)
        self.up4 = nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = True) #(2056, 2056, 128)
        # skip connect +64
        self.conv4 = DoubleConv(128+64, 64) # (2056, 2056, 64)
        self.out = nn.Conv2d(64, 1, 1) # (2056, 2056, 1)

    def forward(self, x):
        x1 = self.receive(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5)
        x = torch.cat([x4, x], dim = 1)
        x = self.conv1(x)
        x = self.up2(x)
        x = torch.cat([x3, x], dim = 1)
        x= self.conv2(x)
        x = self.up3(x)
        x = torch.cat([x2, x], dim = 1)
        x = self.conv3(x)
        x = self.up4(x)
        x = torch.cat([x1, x], dim = 1)
        x = self.conv4(x)
        x = self.out(x)
        return x
    
    
