"""Train a model to map RGB screenshots to 3D binary encodings"""
from math import prod
from pdb import set_trace as TT
import os
import shutil
from PIL import Image

from einops import rearrange
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from torchinfo import summary
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch as th
from tqdm import tqdm
import yaml
from clients.python.mc_render import mc_render
from render import plot_pred_trg, plot_voxels

from clients.python.src.main.proto.minecraft_pb2 import *
from data import CoordsVoxelsDataset, ImsVoxelsDataset, sort_data
from models import ConvDense, ConvDenseDeconv
from utils import get_exp_name


MODELS = {
    "ConvDense": ConvDense,
    "ConvDenseDeconv": ConvDenseDeconv,
}

DATASETS = {
    "ImsVoxelsDataset": ImsVoxelsDataset,
    "CoordsVoxelsDataset": CoordsVoxelsDataset,
}


def mse_loss(preds, targets):
    return th.nn.functional.mse_loss(preds, targets)

def cross_entropy_loss(preds, targets):
    preds = preds.view(-1, preds.shape[1])
    targets = targets.view(-1, targets.shape[1])
    return th.nn.functional.cross_entropy(preds, targets)
    weight = targets.shape[0] / targets.sum(dim=0)
    return th.nn.functional.cross_entropy(preds, targets, weight=weight)


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
    if cfg.model.name not in MODELS:
        raise Exception(f"Model {cfg.model.name} not implemented")
    Model = MODELS[cfg.model.name] 
    if cfg.train.loss == "mse":
        loss_fn = mse_loss
    elif cfg.train.loss == "ce":
        loss_fn = cross_entropy_loss
    load = cfg.train.load or (cfg.evaluate.mode is not None)    
    save_dir = cfg.save_dir = os.path.join("saves", get_exp_name(cfg))
    # if not os.path.exists(os.path.join(cfg.data.data_dir, "test_data_idxs.json")):
    # Time this function:
    import time
    start = time.time()
    sort_data(cfg)
    print(f"Time to sort data: {time.time() - start}")
    # data_dir = load_data(cfg)        
    # ims, voxels = data_dir["data"], data_dir["labels"]
    Dataset = DATASETS[cfg.data.dataset]
    train_data = Dataset(cfg=cfg, name="train")
    train_dataloader = DataLoader(train_data, batch_size=cfg.train.batch_size, shuffle=True)
    train_features, train_labels = next(iter(train_dataloader))
    model = Model(train_features[0].shape, train_labels[0].shape, cfg)
    # Print model stats with torchinfo
    summary(model, input_data=th.zeros(train_features[0:1].shape))

    optimizer = th.optim.Adam(model.parameters(), lr=cfg.train.lr)

    if load is True and os.path.exists(os.path.join(save_dir, "model.pt")):
        update_i = yaml.load(open(os.path.join(save_dir, "log.yaml"), "r"), Loader=yaml.FullLoader)["update_i"]
        model.load_state_dict(th.load(os.path.join(cfg.save_dir, "model.pt"), map_location=cfg.device))
        optimizer.load_state_dict(th.load(os.path.join(cfg.save_dir, "optimizer.pt"), map_location=cfg.device))
        log = yaml.load(open(os.path.join(save_dir, "log.yaml"), "r"), Loader=yaml.FullLoader)
        update_i = log["update_i"]
    
    else:
        update_i = 0

    if cfg.evaluate.mode != "None":
        evaluate(model, loss_fn, update_i, cfg)
        return


    if cfg.train.load is False or not os.path.exists(os.path.join(save_dir, "model.pt")):
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir, ignore_errors=True)
        os.makedirs(save_dir, exist_ok=True)

    # Initialize tensorboard writer
    writer = SummaryWriter(os.path.join(save_dir))
    
    # Train the model w/ tqdm
    for update_i in tqdm(range(update_i, cfg.train.num_updates)):
        ims_batch, voxels_batch = next(iter(train_dataloader))
        ims_batch = ims_batch.to(cfg.device)
        voxels_batch = voxels_batch.to(cfg.device)

        # batch_idxs = np.random.choice(len(ims), cfg.train.batch_size)
        # ims_batch = ims[batch_idxs]
        # voxels_batch = voxels[batch_idxs]
        # Forward pass
        outputs = model(ims_batch)

        loss = loss_fn(outputs, voxels_batch)
        # loss = mse_loss(outputs, voxels_batch)
        # loss = cross_entropy_loss(outputs, voxels_batch)
        writer.add_scalar("train/loss", loss.item(), update_i)
        disc_preds = th.eye(256).to(cfg.device)[outputs.argmax(1)]
        disc_preds = rearrange(disc_preds, "b w h d c -> b c w h d")
        disc_loss = mse_loss(disc_preds, voxels_batch)
        writer.add_scalar("train/discrete_loss", disc_loss.item(), update_i)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        hydra.output_subdir = save_dir

        # Update the model
        for param in model.parameters():
            param.data -= cfg.train.lr * param.grad.data
        
        if (update_i + 1) % cfg.train.log_interval == 0:
            print(f'Update [{update_i + 1}/{cfg.train.num_updates}], Loss: {loss.item():.4e}, Discrete loss: {disc_loss.item():.4e}')

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

        # Evaluate
        if (update_i + 1) % cfg.train.eval_interval == 0:
            val_loss, val_disc_loss = eval_data("val", model, loss_fn, cfg, results_dir=None)
            # Log to tensorboard
            if val_loss is not None:
                writer.add_scalar("val/loss", val_loss, update_i)
                writer.add_scalar("val/discrete_loss", val_disc_loss.item(), update_i)


def evaluate(model, loss_fn, update_i, cfg):

    # results_dir = os.path.join(cfg.save_dir, f"results_{update_i}")
    results_dir = os.path.join(cfg.save_dir, f"results")
    # Create results directory
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir, exist_ok=True)

    # data_dir = load_data(cfg)

    # for name in ["train", "val", "test"]:
    for name in ["train"]:
        eval_data(name, model, loss_fn, cfg, results_dir=results_dir)


def eval_data(name, model, loss_fn, cfg, results_dir=None):
    Dataset = DATASETS[cfg.data.dataset]
    data = Dataset(cfg=cfg, name=name)
    if len(data) == 0:
        return None, None
    dataloader = DataLoader(data, batch_size=cfg.evaluate.batch_size, shuffle=True)
    # data = data_dir[f"{name}_data"]
    # if data.shape[0] == 0:
        # return
    # labels = data_dir[f"{name}_labels"]
    if len(dataloader) == 0:
        return
    features, labels = next(iter(dataloader))
    # Put on device
    features = features.to(cfg.device)
    labels = labels.to(cfg.device)

    # TODO: batch this properly

    model.eval()
    with th.no_grad():
        # Evaluate the model
        preds = model(features)
        loss = loss_fn(preds, labels)
        disc_preds = th.eye(256).to(cfg.device)[preds.argmax(1)]
        disc_preds = rearrange(disc_preds, "b w h d c -> b c w h d")
        disc_loss = mse_loss(disc_preds, labels)
        # loss = cross_entropy_loss(preds, labels)
        if results_dir is not None:

            render_preds, render_labels = [], []
            # Visualize the results
            for i in range(min(10, len(features))):
                if cfg.data.dataset == "ImsVoxelsDataset":
                    img = features[i].cpu().numpy().transpose(1, 2, 0)
                    img = img.astype(np.uint8)
                else:
                    img = None
                # img = Image.fromarray(img)
                pred = preds[i].cpu().numpy()
                pred = np.round(pred)
                # img.save(os.path.join(results_dir, f"{name}_im_{i}.png"))
                label = labels[i].cpu().numpy()
                # plot_voxels(pred, label, save_path=os.path.join(results_dir, f"{name}_pred_trg_{i}.png"))
                plot_pred_trg(pred=pred, trg=label, img=img, save_path=os.path.join(results_dir, f"{name}_pred_trg_{i}.png"))
                pred = preds[i].cpu().numpy()
                render_preds.append(pred)
                render_labels.append(label)

            if cfg.evaluate.mode == "mc":
                mc_render(render_preds, render_labels)

    print(f"{name} loss: {loss.item()}")
    return loss, disc_loss

if __name__ == "__main__":
    main()