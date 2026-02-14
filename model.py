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
        self.receive = DoubleConv(in_dims, 64) #(2056, 2056, 1) -> (2056, 2056, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128)) # (1024, 1024, 128)
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256)) # (512, 512, 256)
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512)) # (256, 256, 512)
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024)) # (128, 128, 1024)
        self.up1 = nn.Sequential(nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = True), nn.Conv2d(1024+512, 512, 3, padding = 1)) # (256, 256, 512)
        self.up2 = nn.Sequential(nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = True), nn.Conv2d(512+256, 256, 3, padding = 1)) # (512, 512, 256)
        self.up3 = nn.Sequential(nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = True), nn.Conv2d(256+128, 128, 3, padding = 1)) # (1024, 1024, 128)
        self.up4 = nn.Sequential(nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = True), nn.Conv2d(128+64, 64, 3, padding = 1)) # (2056, 2056, 64)
        self.out = nn.Conv2d(64, 1, 1) # (2056, 2056, 1)

    def forward(self, x):
        x1 = self.receive(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5)
        x = torch.cat([x4, x], dim = 1)
        x = self.up2(x)
        x = torch.cat([x3, x], dim = 1)
        x = self.up3(x)
        x = torch.cat([x2, x], dim = 1)
        x = self.up4(x)
        x = torch.cat([x1, x], dim = 1)
        x = self.out(x)
