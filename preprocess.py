import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from pathlib import Path
import tifffile
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("image_mask.csv")
train_df = df.iloc[0:4]
val_df = df.iloc[-1:]

class SliceDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __getitem__(self, idx):
        image_path = Path(self.dataframe.iloc[idx]["image_path"])
        mask_path = Path(self.dataframe.iloc[idx]["mask_path"])
        img = tifffile.imread(str(image_path))
        mask = tifffile.imread(str(mask_path))

        return {
            "img": torch.from_numpy(img.astype(np.float32)), 
            "mask": torch.from_numpy(mask.astype(np.int64))
        }
    def __len__(self):
        return len(self.dataframe)

train_dataset = SliceDataset(train_df)
val_dataset = SliceDataset(val_df)
dataloader = DataLoader(train_dataset, batch_size=4)

# visualize image and mask overlay from one batch
batch = next(iter(dataloader))
imgs, masks = batch["img"].numpy(), batch["mask"].numpy()
n = min(4, len(imgs))
fig, axs = plt.subplots(1, n, figsize=(4 * n, 4))
for i in range(n):
    ax = axs[i] if n > 1 else axs
    ax.imshow(imgs[i], cmap="gray")
    ax.imshow(masks[i], alpha=0.4, cmap="Reds")
    ax.set_title(f"Sample {i}")
    ax.axis("off")
plt.tight_layout()
plt.savefig("overlay.png")
plt.show()

