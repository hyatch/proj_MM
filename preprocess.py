import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from pathlib import Path
import tifffile
import numpy as np
import torchvision.transforms.functional as TF

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
        img = torch.from_numpy(img.astype(np.float32))/255.0
        mask = torch.from_numpy(mask.astype(np.float32))
        
        img = TF.resize(img, [2048,2048])
        mask = TF.resize(mask, [2048,2048], interpolation=TF.InterpolationMode.NEAREST)

        return (
            img, mask
        )
    def __len__(self):
        return len(self.dataframe)

