"""
Save a grid of example images from Medium ImageNet to disk.

Run from the repo root:
    python visualize.py
Outputs: medium_imagenet_examples.png
"""

import h5py
import torch
from torchvision.utils import make_grid,save_image

from data.datasets import MediumImagenetHDF5Dataset

N = 16 # number of images to display
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
OUT_PATH = "medium_imagenet_examples.png"

def denormalize(imgs):
    """Undo transform.Normalize (which dataset returns) so colors look right: x * std + mean."""
    mean = torch.tensor(MEAN).view(1, 3, 1, 1)
    std = torch.tensor(STD).view(1, 3, 1, 1)
    return (imgs * std + mean).clamp(0, 1)

if __name__ == "__main__":
    # Peak at contents of H5PY file
    with h5py.File("/data/nmep/cv/medium-imagenet-96.hdf5", "r") as f:
        print("H5PY keys:", {k: f[k].shape for k in f.keys()})
    
    # H5PY keys: {'images-test': (75929, 3, 96, 96), 
    #'images-train': (607425, 3, 96, 96), 
    # 'images-val': (75928, 3, 96, 96), 
    # 'labels-train': (607425,), 
    # 'labels-val': (75928,)}

    # augment=False so we can see the real images, not jittered/flipped ones
    dataset = MediumImagenetHDF5Dataset(img_size=96, split="train", augment=False)

    # Spread indices across the dataset instead of taking the first N
    idxs = torch.linspace(0, len(dataset) - 1, N).long().tolist()
    samples = [dataset[i] for i in idxs]
    imgs = torch.stack([img for img, _ in samples])
    labels = [int(lbl) for _, lbl in samples]

    grid = make_grid(denormalize(imgs), nrow=4, padding=2)
    save_image(grid, OUT_PATH)
    print(f"Saved {OUT_PATH} | image shape {tuple(imgs.shape[1:])} | labels {labels}")
