import json
import os
from pdb import set_trace as TT

from einops import rearrange
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch as th
from torch.utils.data import Dataset
from torchvision.io import read_image

from clients.python.utils import get_vox_xz_from_view
from utils import idx_to_x_z_rot


def sort_data(cfg):
    """Divide data into train/val/test sets."""
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
        idx_to_coords = {}
        n_sample = 0
        pct = cfg.data[f"pct_{name}"]
        num_samples = int(pct * cfg.data.num_samples)
        while n_sample < num_samples:
            im_path = os.path.join(cfg.data.data_dir, "screenshots", f"{load_i}.png")
            if os.path.isfile(im_path):
                # assert os.path.isfile(os.path.join(cfg.data.data_dir, "voxels", f"{load_i}.npy"))
                data_idxs[n_sample] = load_i
                x, z, rot = idx_to_x_z_rot(load_i)
                idx_to_coords[n_sample] = x, z
                n_sample += 1
            load_i += 1
        # Dump data_idxs
        print(data_idxs)
        with open(os.path.join(cfg.data.data_dir, f"{name}_data_idxs.json"), "w") as f:
            json.dump(data_idxs, f)




# TODO: factor out some of the shared functionality between these two classes.
class ImsVoxelsDataset(Dataset):
    def __init__(self, cfg, name='train'):
        self.screenshot_coords = pd.read_csv(os.path.join(cfg.data.data_dir, "screenshot_coords.csv"))
        self.vox_coords = pd.read_csv(os.path.join(cfg.data.data_dir, "voxel_coords.csv"))
        self.vox_xz_to_vox_idx = {(x, z): i for i, (_, x, y, z) in self.vox_coords.iterrows()}
        self.img_dir = os.path.join(cfg.data.data_dir, "screenshots")
        self.vox_dir = os.path.join(cfg.data.data_dir, "voxels")
        with open(os.path.join(cfg.data.data_dir, f"{name}_data_idxs.json"), "r") as f:
            self.data_idxs = json.load(f)
            self.data_idxs = {int(k): v for k, v in self.data_idxs.items()}

    def __len__(self):
        return len(self.data_idxs)

    def __getitem__(self, idx):
        img_idx = self.data_idxs[idx]
        img_path = os.path.join(self.img_dir, f"{img_idx}.png")
        # _, x, y, z, rot = self.screenshot_coords.loc[img_idx]

        # vox_x, vox_z = get_vox_xz_from_view(x, z, rot)
        # vox_idx = self.vox_xz_to_vox_idx[(vox_x, vox_z)]
        vox_idx = img_idx

        image = read_image(img_path)
        vox_path = os.path.join(self.vox_dir, f"{str(vox_idx)}.npy")
        voxels = th.Tensor(np.load(vox_path)).long()
        voxels = th.eye(256)[voxels]
        voxels = rearrange(voxels, "x y z c -> c x y z")
        # voxels = th.rot90(voxels, k=rot // 90, dims=(0, 2))

        # Rotate the chunk of voxels to be consistent with the rotation of the screenshot
        # DEBUG show image and plot voxels
     
        # fig = plt.figure()
        # ax = fig.add_subplot(211)
        # im = image.numpy()
        # im = im.transpose(1, 2, 0)
        # ax.imshow(im)
        # ax = fig.add_subplot(212, projection='3d')
        # v = voxels.numpy()
        # v = v.transpose(2, 0, 1)
        # # Rotate 90 deg around height axis
        # v = np.rot90(v, 1, (0, 1))
        # ax.voxels(v, edgecolor="k")
        # print("rot", rot)
        # plt.show()

        # print(vox_path)
        # print(voxels.shape)
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
        self.coords_to_idx = {(v, k) for k, v in self.coords.items()}
        self.norm_coords = {k: v / max_coords for k, v in self.coords.items()}
        self.vox_dir = os.path.join(cfg.data.data_dir, "voxels")
        with open(os.path.join(cfg.data.data_dir, f"{name}_data_idxs.json"), "r") as f:
            self.data_idxs = json.load(f)
            self.data_idxs = {int(k): v for k, v in self.data_idxs.items()}

    def __len__(self):
        return len(self.data_idxs)

    def __getitem__(self, idx):
        c = th.Tensor(self.coords[idx])
        vox_path = os.path.join(self.vox_dir, f"{self.data_idxs[str(idx)]}.npy")
        voxels = th.Tensor(np.load(vox_path))
        return c.float(), voxels.float()
    
class VoxelsDataset(Dataset):
    """Voxels to themselves, for training an autoencoder."""
    def __init__(self, cfg, name="train") -> None:
        self.vox_dir = os.path.join(cfg.data.data_dir, "voxels")
        with open(os.path.join(cfg.data.data_dir, f"{name}_data_idxs.json"), "r") as f:
            self.data_idxs = json.load(f)
            self.data_idxs = {int(k): v for k, v in self.data_idxs.items()}

    def __len__(self):
        return len(self.data_idxs)

    def __getitem__(self, idx):
        vox_path = os.path.join(self.vox_dir, f"{self.data_idxs[str(idx)]}.npy")
        voxels = th.Tensor(np.load(vox_path), dtype=th.int64)
        voxels = th.eye(256)[voxels]
        voxels = rearrange(voxels, "x y z c -> c x y z")
        return voxels, voxels