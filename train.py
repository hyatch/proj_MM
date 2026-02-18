from model import UNet
import wandb
from preprocess import SliceDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import inspect
import pandas as pd

#hyperparameters and setup
df = pd.read_csv("image_mask.csv")
train_df = df.iloc[0:4]
val_df = df.iloc[-1:]
train_dataset = SliceDataset(train_df)
val_dataset = SliceDataset(val_df)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle = True)
val_dataloader = DataLoader(val_dataset, batch_size = 1)
device = "cpu" if torch.cuda.is_available() else "cpu" #CHANGE TO CUDA IF HAVE GPU
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

# training loop
model = UNet(1)
criterion = ComboLoss()
model.to(device)
print(f"using {device}")
fused_avail = "fused" in inspect.signature(torch.optim.AdamW).parameters
use_fused = fused_avail and "cuda" in device
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)

for iters in range(epochs):
    if iters % 5 == 0:
        model.eval()
        val_dice_total = 0.0
        
        with torch.no_grad():
            for batch_idx, (image, mask) in enumerate(val_dataloader):
                image = image.to(device)
                mask = mask.to(device).float()
                
                logits = model(image)
                val_loss, val_dice = criterion(logits, mask)
                
                val_dice_total += val_dice.item()
                
                predictions = (torch.sigmoid(logits) > 0.5).float()
                wandb.log({"val_img": [wandb.Image(image[0], caption="Input"), wandb.Image(predictions[0], caption="Pred")]})

        avg_val_dice = val_dice_total / len(val_dataloader)
        wandb.log({"val_dice_score": avg_val_dice, "epoch": iters})
        
        
    model.train()
    running = 0.0
    for batch_idx, (images, masks) in enumerate(train_dataloader):
        
        images = images.to(device) 
        masks = masks.to(device).float()
        optimizer.zero_grad()
        predictions = model(images)
        combo_loss, dice_loss = criterion(predictions, masks)
        combo_loss.backward() 
        optimizer.step()
        running += combo_loss.item()
        avg_train_loss = running / len(train_dataloader)
        wandb.log({"train_loss": avg_train_loss, "epoch": iters})
    
run.finish()