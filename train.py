"""Train a model to map RGB screenshots to 3D binary encodings"""
from math import prod
from pdb import set_trace as TT
import os
import shutil
from PIL import Image

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch as th
from tqdm import tqdm
import yaml
from render import plot_voxels

from data import ImsVoxelsDataset, sort_data
from utils import get_exp_name


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
        x = th.nn.functional.sigmoid(self.fc3(x))
        x = x.view(-1, *self.out_shape)
        return x


def mse_loss(preds, targets):
    return th.nn.functional.mse_loss(preds, targets)



# def load_data(cfg):
#     # Load data
#     if not os.path.exists("data/ims.pt"):
#         ims = []
#         voxels = []
#         load_i = 0
#         while len(ims) < cfg.data.num_samples:
#             print(load_i)
#             im_path = os.path.join(cfg.data.data_dir, "screenshots", f"{load_i}.png")
#             if os.path.isfile(im_path):
#                 im = Image.open(im_path)
#                 im = np.array(im)
#                 ims.append(im)
#                 voxel_path = os.path.join(cfg.data.data_dir, "voxels", f"{load_i}.npy")
#                 vxls = np.load(voxel_path)
#                 voxels.append(vxls)
#             load_i += 1
        
#         ims = th.Tensor(np.array(ims))
#         ims = ims.permute(0, 3, 1, 2)
#         voxels = th.Tensor(np.array(voxels))
#         # Save ims and voxels datasets
#         th.save(ims, os.path.join("data", "ims.pt"))
#         th.save(voxels, os.path.join("data", "voxels.pt"))
        
#         pct_train = cfg.data.pct_train
#         num_train = int(pct_train * len(ims))
#         pct_val = cfg.data.pct_val
#         num_val = int(pct_val * len(ims))
#         # Save train/validation/test splits
#         th.save(ims[:num_train], cfg.data.train_data)
#         th.save(voxels[:num_train], cfg.data.train_labels)
#         th.save(ims[num_train:num_train+num_val], cfg.data.val_data)
#         th.save(voxels[num_train:num_train+num_val], cfg.data.val_labels)
#         th.save(ims[num_train+num_val:], cfg.data.test_data)
#         th.save(voxels[num_train+num_val:], cfg.data.test_labels)

#     else:
#         ims = th.load(os.path.join("data", "ims.pt"))[:cfg.data.num_samples]
#         voxels = th.load(os.path.join("data", "voxels.pt"))[:cfg.data.num_samples]
#     train_data = th.load(cfg.data.train_data)
#     train_labels = th.load(cfg.data.train_labels)
#     val_data = th.load(cfg.data.val_data)
#     val_labels = th.load(cfg.data.val_labels)
#     test_data = th.load(cfg.data.test_data)
#     test_labels = th.load(cfg.data.test_labels)
#     return {"data": ims, "labels": voxels,
#             "train_data": train_data, "train_labels": train_labels,
#             "val_data": val_data, "val_labels": val_labels,
#             "test_data": test_data, "test_labels": test_labels,
#             }


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entrypoint for the project"""
    load = cfg.train.load or cfg.evaluate.val
    save_dir = cfg.save_dir = os.path.join("saves", get_exp_name(cfg))
    if not os.path.exists(os.path.join(cfg.data.data_dir, "test_data_idxs.json")):
        sort_data(cfg)
    # data_dir = load_data(cfg)        
    # ims, voxels = data_dir["data"], data_dir["labels"]
    train_data = ImsVoxelsDataset(cfg=cfg, name="train")
    train_dataloader = DataLoader(train_data, batch_size=cfg.train.batch_size, shuffle=True)
    train_features, train_labels = next(iter(train_dataloader))
    model = Model(train_features[0].shape, train_labels[0].shape, cfg)
    optimizer = th.optim.Adam(model.parameters(), lr=cfg.train.lr)

    if load is True:
        update_i = yaml.load(open(os.path.join(save_dir, "log.yaml"), "r"), Loader=yaml.FullLoader)["update_i"]
        model.load_state_dict(th.load(os.path.join(cfg.save_dir, "model.pt")))
        optimizer.load_state_dict(th.load(os.path.join(cfg.save_dir, "optimizer.pt")))
        log = yaml.load(open(os.path.join(save_dir, "log.yaml"), "r"), Loader=yaml.FullLoader)
        update_i = log["update_i"]
    
    else:
        update_i = 0

    if cfg.evaluate.val is True:
        evaluate(model, update_i, cfg)
        return


    if cfg.train.load is False:
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir, exist_ok=True)

    # Initialize tensorboard writer
    writer = SummaryWriter(os.path.join(save_dir))
    
    # Train the model w/ tqdm
    for update_i in tqdm(range(update_i, cfg.train.num_updates)):
        ims_batch, voxels_batch = next(iter(train_dataloader))

        # batch_idxs = np.random.choice(len(ims), cfg.train.batch_size)
        # ims_batch = ims[batch_idxs]
        # voxels_batch = voxels[batch_idxs]
        # Forward pass
        outputs = model(ims_batch)

        loss = mse_loss(outputs, voxels_batch)
        writer.add_scalar("train/loss", loss.item(), update_i)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        hydra.output_subdir = save_dir

        # Update the model
        for param in model.parameters():
            param.data -= cfg.train.lr * param.grad.data
        
        if (update_i + 1) % cfg.train.log_interval == 0:
            print(f'Epoch [{update_i + 1}/{cfg.train.num_updates}], Loss: {loss.item():.4e}')

        # Save model and optimizer
        if (update_i + 1) % cfg.train.save_interval == 0:
            th.save(model.state_dict(), os.path.join(save_dir, f"model_{update_i}.pt"))
            th.save(optimizer.state_dict(), os.path.join(save_dir, f"optimizer_{update_i}.pt"))
            # Copy the latest model and optimizer
            th.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
            th.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer.pt"))
            # Remove old checkpoints
            last_update_i = update_i - cfg.train.save_interval
            # Save log yaml
            log = {"update_i": update_i, "loss": loss.item()}
            # Write los to tensorboard
            with open(os.path.join(save_dir, "log.yaml"), "w") as f:
                yaml.dump(log, f)

            last_model_path = os.path.join(save_dir, f"model_{last_update_i}.pt")
            last_optimizer_path = os.path.join(save_dir, f"optimizer_{last_update_i}.pt")
            if os.path.exists(last_model_path):
                os.remove(last_model_path)
            if os.path.exists(last_optimizer_path):
                os.remove(last_optimizer_path)


def evaluate(model, update_i, cfg):

    # results_dir = os.path.join(cfg.save_dir, f"results_{update_i}")
    results_dir = os.path.join(cfg.save_dir, f"results")
    # Create results directory
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir, exist_ok=True)

    # data_dir = load_data(cfg)

    def eval_data(name, model):
        data = ImsVoxelsDataset(cfg=cfg, name=name)
        dataloader = DataLoader(data, batch_size=cfg.evaluate.batch_size, shuffle=True)
        # data = data_dir[f"{name}_data"]
        # if data.shape[0] == 0:
            # return
        # labels = data_dir[f"{name}_labels"]
        features, labels = next(iter(dataloader))

        # TODO: batch this properly

        model.eval()
        with th.no_grad():
            # Evaluate the model
            preds = model(features)
            loss = mse_loss(preds, labels)
            # Visualize the results
            for i in range(min(10, len(features))):
                img = features[i].cpu().numpy().transpose(1, 2, 0)
                pred = preds[i].cpu().numpy()
                pred = np.round(pred)
                img = img.astype(np.uint8)
                img = Image.fromarray(img)
                img.save(os.path.join(results_dir, f"{name}_im_{i}.png"))
                label = labels[i].cpu().numpy()
                plot_voxels(pred, label, save_path=os.path.join(results_dir, f"{name}_pred_trg_{i}.png"))
                pred = preds[i].cpu().numpy()

            print(f"{name} loss: {loss.item()}")

    for name in ["train", "val", "test"]:
        eval_data(name, model)

if __name__ == "__main__":
    main()