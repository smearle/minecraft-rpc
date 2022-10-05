import json
import os
from pdb import set_trace as TT
import numpy as np
import pandas as pd
from PIL import Image
import torch as th
from torch.utils.data import Dataset
from torchvision.io import read_image

from utils import square_spiral

def sort_data(cfg):
    load_i = 0
    for name in ["train", "val", "test"]:
        base_load_i = load_i
        # Make data directory:
        # data_dir = os.path.join(data_subdir, "screenshots")
        # labels_dir = os.path.join(data_subdir, "voxels")
        # if not os.path.exists(data_dir):
            # os.makedirs(data_dir)
        # if not os.path.exists(labels_dir):
            # os.makedirs(labels_dir)

        data_idxs = {}
        coords = {}
        n_sample = 0
        pct = cfg.data[f"pct_{name}"]
        num_samples = int(pct * cfg.data.num_samples)
        while n_sample < num_samples:
            print(load_i)
            im_path = os.path.join(cfg.data.data_dir, "screenshots", f"{load_i}.png")
            if os.path.isfile(im_path):
                assert os.path.isfile(os.path.join(cfg.data.data_dir, "voxels", f"{load_i}.npy"))
                data_idxs[n_sample] = load_i
                coords[n_sample] = square_spiral(load_i)
                n_sample += 1
            load_i += 1
        # Dump data_idxs
        with open(os.path.join(cfg.data.data_dir, f"{name}_data_idxs.json"), "w") as f:
            json.dump(data_idxs, f)
        # Dump coords
        with open(os.path.join(cfg.data.data_dir, f"{name}_coords.json"), "w") as f:
            json.dump(coords, f)

class ImsVoxelsDataset(Dataset):
    def __init__(self, cfg, name='train'):
        self.img_dir = os.path.join(cfg.data.data_dir, "screenshots")
        self.vox_dir = os.path.join(cfg.data.data_dir, "voxels")
        with open(os.path.join(cfg.data.data_dir, f"{name}_data_idxs.json"), "r") as f:
            self.data_idxs = json.load(f)

    def __len__(self):
        return len(self.data_idxs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, f"{self.data_idxs[str(idx)]}.png")
        image = read_image(img_path)
        vox_path = os.path.join(self.vox_dir, f"{self.data_idxs[str(idx)]}.npy")
        voxels = th.Tensor(np.load(vox_path))
        # Changing to float32
        return image.float(), voxels.float()

class CoordsVoxelsDataset(Dataset):
    def __init__(self, cfg, name="train") -> None:
        self.coords_path = os.path.join(cfg.data.data_dir, f"{name}_coords.json")
        # load coords
        with open(self.coords_path, "r") as f:
            self.coords = json.load(f)
        coords_arr = np.array(list(self.coords.values()))
        max_coords = np.max(coords_arr, axis=0)
        self.coords = {k: v / max_coords for k, v in self.coords.items()}
        self.vox_dir = os.path.join(cfg.data.data_dir, "voxels")
        with open(os.path.join(cfg.data.data_dir, f"{name}_data_idxs.json"), "r") as f:
            self.data_idxs = json.load(f)

    def __len__(self):
        return len(self.data_idxs)

    def __getitem__(self, idx):
        c = th.Tensor(self.coords[str(idx)])
        vox_path = os.path.join(self.vox_dir, f"{self.data_idxs[str(idx)]}.npy")
        voxels = th.Tensor(np.load(vox_path))
        return c.float(), voxels.float()
