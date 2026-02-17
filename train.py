from re import M
from model import UNet
import wandb
import preprocess
import torch
import torch.nn as nn
import inspect

#hyperparameters and setup
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 0.02
epochs = 6
run = wandb.init(
    entity = "eyh002-uc-san-diego",
    project = "proj_MM",
    config={
        "learning_rate": 0.02,
        "architecture": "U-Net",
        "dataset": "MM",
        "epochs": epochs,
    },
)

class ComboLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.binary_entropy = nn.BCEWithLogitsLoss()
        self.alpha = 0.5
        self.beta = 1-self.alpha
    def forward(self, input, target):
        bce = self.binary_entropy(input, target)
        input = torch.sigmoid(input)
        smooth = 1.0
        
        inputs = input.view(-1)
        targets = target.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        dice_loss = 1 - dice 
        return (self.alpha * bce) + (self.beta * dice_loss), dice_loss


model = UNet(in_channels = 1)
model.to(device)
print(f"using {device}")
fused_avail = "fused" in inspect.signature(torch.optim.AdamW).parameters
use_fused = fused_avail and "cuda" in device
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)

for iter in range(epochs):
    if (iter % 5 == 0):
        model.val()
        run.log({"combo loss": ..., "dice score": ...})
    model.train()
    
run.finish()