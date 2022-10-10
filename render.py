import os
from pdb import set_trace as TT

import matplotlib.pyplot as plt
import numpy as np
from clients.python.src.main.proto.minecraft_pb2 import *

def plot_vox_to_ax(ax, voxels: np.ndarray):
    voxels = np.argmax(voxels, axis=0)
    voxels = voxels.transpose(2, 0, 1)
    # Rotate the voxels 90 degrees around the height axis
    voxels = np.rot90(voxels, 1, (0, 1))
    # Dirt blocks are brown
    blocks_to_cols = {
        DIRT: "brown",
        GRASS: "green",
        STONE: "gray",
        WATER: "blue",
    }
    # vox_air = voxels == AIR
    other_vox = voxels != AIR
    for block_id in blocks_to_cols:
        ax.voxels(voxels == block_id, facecolors=blocks_to_cols[block_id])
        other_vox = other_vox & (voxels != block_id)
    ax.voxels(other_vox, edgecolor="k", facecolor="red")
    return ax

def plot_voxels(voxels: np.ndarray, trg_voxels: np.ndarray, save_path: str = None):
    if trg_voxels is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax = plot_vox_to_ax(ax, voxels)
    else:
        fig = plt.figure()
        ax = fig.add_subplot(121, projection='3d')
        ax = plot_vox_to_ax(ax, voxels)
        ax = fig.add_subplot(122, projection='3d')
        ax = plot_vox_to_ax(ax, trg_voxels)
        # plt.show()
        # plt.close()
    # Save the figure
    return fig

def plot_pred_trg(pred: np.ndarray, trg: np.ndarray, img: np.ndarray = None, save_path: str = None):
    # The image is large and wide on top, the voxels are small beneath it:
    if img is not None:
        fig = plt.figure()
        ax = fig.add_subplot(211)
        ax.imshow(img)
        ax = fig.add_subplot(223, projection='3d')
        ax = plot_vox_to_ax(ax, pred)
        ax = fig.add_subplot(224, projection='3d')
        ax = plot_vox_to_ax(ax, trg)
        # Save the figure
    else:
        fig = plt.figure()
        ax = fig.add_subplot(121, projection='3d')
        ax = plot_vox_to_ax(ax, pred)
        ax = fig.add_subplot(122, projection='3d')
        ax = plot_vox_to_ax(ax, trg)

    if save_path is not None:
        fig.savefig(save_path)
    plt.close()
