import os
import matplotlib.pyplot as plt
import numpy as np

def plot_voxels(voxels: np.ndarray, trg_voxels: np.ndarray, save_path: str = None):
    if trg_voxels is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        voxels = voxels.transpose(2, 0, 1)
        ax.voxels(voxels, edgecolor="k")
    else:
        fig = plt.figure()
        ax = fig.add_subplot(121, projection='3d')
        voxels = voxels.transpose(2, 0, 1)
        ax.voxels(voxels, edgecolor="k")
        ax = fig.add_subplot(122, projection='3d')
        trg_voxels = trg_voxels.transpose(2, 0, 1)
        ax.voxels(trg_voxels, edgecolor="k")
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
        pred = pred.transpose(2, 0, 1)
        # Rotate 90 deg around height axis
        pred = np.rot90(pred, 1, (0, 1))
        ax.voxels(pred, edgecolor="k")
        ax = fig.add_subplot(224, projection='3d')
        trg = trg.transpose(2, 0, 1)
        trg = np.rot90(trg, 1, (0, 1))
        ax.voxels(trg, edgecolor="k")
        # Save the figure
    else:
        fig = plt.figure()
        ax = fig.add_subplot(121, projection='3d')
        pred = pred.transpose(2, 0, 1)
        # Rotate 90 deg around height axis
        pred = np.rot90(pred, 1, (0, 1))
        ax.voxels(pred, edgecolor="k")
        ax = fig.add_subplot(122, projection='3d')
        trg = trg.transpose(2, 0, 1)
        trg = np.rot90(trg, 1, (0, 1))
        ax.voxels(trg, edgecolor="k")

    if save_path is not None:
        fig.savefig(save_path)
    plt.close()
