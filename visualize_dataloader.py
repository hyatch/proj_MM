"""
Visualize dataloader: TIFF image with ORSObject mask overlay.
Run: python visualize_dataloader.py
"""
from pathlib import Path
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch
import re


def load_ors_object(mask_path, image_shape=None):
    """
    Try to load an ORSObject file (Dragonfly format: binary + XML tail).
    If image_shape is provided and binary size matches, interpret as raw mask array.
    Returns numpy array (H, W) or (H, W, 1) with values 0/1, or None on failure.
    """
    path = Path(mask_path)
    if not path.exists():
        return None
    data = path.read_bytes()
    if len(data) == 0:
        return None

    # ORSObject = binary blob + XML at end. Find where XML starts.
    # Common patterns: <?xml, or <Dragonfly, or <ORS
    for start_tag in (b"<?xml", b"<Dragonfly", b"<ORS", b"<"):
        idx = data.rfind(start_tag)
        if idx > 0:
            binary_part = data[:idx]
            break
    else:
        binary_part = data

    # Try to get shape from XML (optional)
    xml_part = data[len(binary_part):].decode("utf-8", errors="ignore")
    shape = None
    # Look for dimension-like numbers in XML
    for match in re.finditer(r"(\d+)\s*[,\s]\s*(\d+)", xml_part):
        w, h = int(match.group(1)), int(match.group(2))
        if 10 < w < 20000 and 10 < h < 20000 and w * h == len(binary_part):
            shape = (h, w)
            break
    if shape is None and image_shape is not None:
        h, w = image_shape[:2]
        if h * w == len(binary_part):
            shape = (h, w)
        elif h * w * 4 == len(binary_part):
            shape = (h, w)
    if shape is None:
        n = int(len(binary_part) ** 0.5)
        if n * n == len(binary_part):
            shape = (n, n)

    if shape is None:
        return None
    n_el = int(np.prod(shape))
    if len(binary_part) == n_el:
        dtype = np.uint8
    elif len(binary_part) == n_el * 4:
        dtype = np.uint32
    else:
        return None
    try:
        arr = np.frombuffer(binary_part, dtype=dtype)
        arr = arr.reshape(shape)
        # Binarize for overlay
        mask = (arr > 0).astype(np.uint8)
        return mask
    except Exception:
        return None


class SliceDataset(Dataset):
    def __init__(self, dataframe, base_dir=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.base_dir = Path(base_dir or ".")

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image_path = self.base_dir / self.dataframe.iloc[idx]["image_path"]
        mask_path = self.base_dir / self.dataframe.iloc[idx]["mask_path"]
        img = tifffile.imread(str(image_path))
        if img.ndim == 3:
            img = img.squeeze()
        mask = load_ors_object(str(mask_path), img.shape)
        # Return tensors so DataLoader can collate; use zeros if mask load failed
        img_t = torch.from_numpy(np.asarray(img, dtype=np.float32))
        if mask is not None and mask.shape == img.shape:
            mask_t = torch.from_numpy(mask)
        else:
            mask_t = torch.zeros_like(img_t, dtype=torch.uint8)
        return {
            "image": img_t,
            "mask": mask_t,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
        }


def visualize_batch(batch, figsize=(12, 5), overlay_color=(1, 0, 0), overlay_alpha=0.4, save_path=None):
    """Plot images with optional mask overlay. batch is list of dicts with 'image' and 'mask'."""
    n = len(batch)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    for ax, sample in zip(axes, batch):
        img = sample["image"]
        mask = sample.get("mask")
        if img.ndim == 3:
            img = img.squeeze()
        ax.imshow(img, cmap="gray")
        if mask is not None and mask.shape == img.shape and np.any(mask > 0):
            overlay = np.zeros((*mask.shape, 4))
            overlay[mask > 0] = (*overlay_color, overlay_alpha)
            ax.imshow(overlay)
        elif mask is not None and not np.any(mask > 0):
            ax.set_title("(mask empty / load failed)")
        ax.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()


def main():
    csv_path = Path("image_mask.csv")
    df = pd.read_csv(csv_path)
    base_dir = csv_path.resolve().parent

    dataset = SliceDataset(df, base_dir=base_dir)
    dataloader = DataLoader(dataset, batch_size=5, shuffle=False)

    # Visualize first batch
    batch = next(iter(dataloader))
    samples = []
    for i in range(batch["image"].shape[0]):
        img = batch["image"][i].numpy()
        mask = batch["mask"][i].numpy()
        samples.append({"image": img, "mask": mask})
    visualize_batch(samples, save_path="dataloader_overlay.png")


if __name__ == "__main__":
    main()
