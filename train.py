"""Train a model to map RGB screenshots to 3D binary encodings"""
from math import prod
from pdb import set_trace as TT
import os
from PIL import Image

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch as th


# Define the model, mapping RGB images to 3D binary encodings
class Model(th.nn.Module):
    def __init__(self, in_shape, out_shape, cfg: DictConfig):    
        super().__init__()
        self.out_shape = out_shape
        self.conv1 = th.nn.Conv2d(4, 6, 5)
        self.pool = th.nn.MaxPool2d(2, 2)
        self.conv2 = th.nn.Conv2d(6, 16, 5)
        self.conv3 = th.nn.Conv2d(16, 8, 5)
        fc_shape = prod(self.pool(self.conv3(self.pool(self.conv2(self.pool(self.conv1(th.zeros(1, *in_shape))))))).shape[1:])
        self.fc1 = th.nn.Linear(fc_shape, 120)
        self.fc2 = th.nn.Linear(120, 84)
        out_size_flat = prod(out_shape)
        self.fc3 = th.nn.Linear(84, out_size_flat)
    
    def forward(self, x):
        b = x.shape[0]
        x = self.pool(th.nn.functional.relu(self.conv1(x)))
        x = self.pool(th.nn.functional.relu(self.conv2(x)))
        x = self.pool(th.nn.functional.relu(self.conv3(x)))
        x = x.reshape(b, -1)
        x = th.nn.functional.relu(self.fc1(x))
        x = th.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.view(-1, *self.out_shape)
        return x


def mse_loss(preds, targets):
    return th.nn.functional.mse_loss(preds, targets)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entrypoint for the project"""
    print(OmegaConf.to_yaml(cfg))
    # Load data
    ims = []
    voxels = []
    load_i = 0
    while len(ims) < cfg.data.num_samples:
        print(load_i)
        im_path = os.path.join(cfg.data.data_dir, "screenshots", f"{load_i}.png")
        if os.path.isfile(im_path):
            im = Image.open(im_path)
            im = np.array(im)
            ims.append(im)
            voxel_path = os.path.join(cfg.data.data_dir, "voxels", f"{load_i}.npy")
            vxls = np.load(voxel_path)
            voxels.append(vxls)
        load_i += 1
    
    ims = th.Tensor(np.array(ims))
    ims = ims.permute(0, 3, 1, 2)
    voxels = th.Tensor(np.array(voxels))
    
    model = Model(ims[0].shape, voxels[0].shape, cfg)
    optimizer = th.optim.Adam(model.parameters(), lr=cfg.train.lr)

    # Train the model
    for update_i in range(cfg.train.num_updates):
        batch_idxs = np.random.choice(len(ims), cfg.train.batch_size)
        ims_batch = ims[batch_idxs]
        voxels_batch = voxels[batch_idxs]
        # Forward pass
        outputs = model(ims_batch)
        loss = mse_loss(outputs, voxels_batch)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update the model
        for param in model.parameters():
            param.data -= cfg.train.lr * param.grad.data
        
        if (update_i + 1) % cfg.train.log_interval == 0:
            print(f'Epoch [{update_i + 1}/{cfg.train.num_updates}], Loss: {loss.item():.4e}')

if __name__ == "__main__":
    main()