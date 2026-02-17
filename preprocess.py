import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from pathlib import Path
import tifffile
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

        img = np.expand_dims(img, axis=0)
        mask = np.expand_dims(mask, axis=0)
        
        img /= 255

        return {
            "img": torch.from_numpy(img.astype(np.float32)), 
            "mask": torch.from_numpy(mask.astype(np.int64))
        }
    def __len__(self):
        return len(self.dataframe)

train_dataset = SliceDataset(train_df)
val_dataset = SliceDataset(val_df)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle = True)
val_dataloader = DataLoader(val_dataset, batch_size = 1)