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
    if save_path is not None:
        fig.savefig(save_path)
    plt.close()

