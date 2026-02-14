import wandb
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from pathlib import Path
import tifffile

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

        if len(image.shape) == 2:
            image=np.expand_dims(img, axis=0)

        img_tensor = torch.from_numpy(image).float()

        return {
            "img": img_tensor,
            "mask": mask_tensor
        }
    def __len__():
        return len(self.dataframe)

train_dataset = SliceDataset(train_df)
val_dataset = SliceDataset(val_df)
dataloader = DataLoader(train_dataset, batch_size = 4)
for batch in dataloader:
    print

