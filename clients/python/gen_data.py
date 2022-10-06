import json
import math
import os
from pdb import set_trace as TT

from fire import Fire
import grpc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyscreenshot as ImageGrab
import pygetwindow
from utils import square_spiral, get_vox_xz_from_view

import src.main.proto.minecraft_pb2_grpc
from src.main.proto.minecraft_pb2 import *
from utils import idx_to_x_z_rot


def top_left_corner_screenshot(name: str, bbox):
    im = ImageGrab.grab(bbox)
    # Save image to file
    im.save(os.path.join("data", "screenshots", f"{name}.png"))


def cube_to_voxels(cube: Cube, shape: tuple, min_xyz: tuple):
    voxels = np.zeros(shape)
    for block in cube.blocks:
        if block.type != AIR:
            voxels[block.position.x - min_xyz[0], block.position.y - min_xyz[1], block.position.z - min_xyz[2]] = 1
    return voxels


def read_cube(client, min_xyz: tuple, max_xyz: tuple):
    print(min_xyz, max_xyz)
    return client.readCube(Cube(min=Point(x=min_xyz[0], y=min_xyz[1], z=min_xyz[2]),
                                max=Point(x=max_xyz[0], y=max_xyz[1], z=max_xyz[2])))


def get_vox_chunk(loc, shp, client):
    x, y, z = loc
    min = (x - math.floor(shp[0] // 2), 
           y - math.floor(shp[1] // 2),
           z - math.floor(shp[2] // 2))
    max = (x + math.ceil(shp[0] // 2) - 1, 
        y + math.ceil(shp[1] // 2) - 1, 
        z + math.ceil(shp[2] // 2) - 1)
    cube = read_cube(client, min, max)
    return cube_to_voxels(cube, shp, min)



def get_voxel_chunks(client, num_samples, load: bool = True):
    chunk_shape = (20, 10, 20)
    if load:
        with open("data/voxels_prog.json", "r") as f:
            i = json.load(f)["curr_i"]
        vox_coords = pd.read_csv("data/voxel_coords.csv", index_col=0)
    else:
        i = 0
        # Initiate datafram. Will be useful for recording y values which we get from interacting with the server.
        vox_coords = pd.DataFrame(columns=["x", "y", "z"])
    while i < num_samples:
        x, z = square_spiral(i)
        x *= chunk_shape[0]
        z *= chunk_shape[2]
        point = client.getHighestYAt(Point(x=x, y=0, z=z))
        y = point.y
        voxels = get_vox_chunk(x, y, z, chunk_shape, client)
        # We are hard-coding this chunk to be looking in front of us given our fixed rotation.
        # plot_voxels(voxels)
        np.save(os.path.join("data", "voxels", f"{i}.npy"), voxels)
        # Map indices to coordinates in pandas dataframe, write to ith row
        vox_coords.loc[i] = [x, y, z]
        if (i + 1) % 100 == 0:
            # Save dataframe to file
            # TODO: Append to the file instead of overwriting
            vox_coords.to_csv(os.path.join("data", "voxel_coords.csv"))
            print(f"Saved {i + 1} voxel chunk samples")
        i += 1
    {"curr_i": i}
    with open("data/voxels_prog.json", "w") as f:
        json.dump({"curr_i": i}, f)


def get_screenies(client, num_samples, load: bool = True):
    """Boopchie does a tourism."""
    chunk_shape = (20, 10, 20)

    SCREENSHOT_SIZE = (512, 512)
    BBOX_OFFSET = (0, 66)  # For 16" M1 MacBook Pro
    BBOX = (BBOX_OFFSET[0], BBOX_OFFSET[1], BBOX_OFFSET[0] + SCREENSHOT_SIZE[0], BBOX_OFFSET[1] + SCREENSHOT_SIZE[1])
    # bbox = pyautogui.locateOnScreen('mc.png')

    active_window = pygetwindow.getActiveWindow()
    # Continue when user enters "y"
    input("Put the Minecraft window in the top corner of the screen (do not scale it!). Enter the server (localhost) and " +
    " enter camera mode (F1). Press enter to continue...")
    print("Now change focus to the Minecraft window. (User, e.g. alt + tab. Do not move the window!)")

    if load:
        with open("data/screenshots_prog.json", "r") as f:
            i = json.load(f)["curr_i"]
        scrn_coords = pd.read_csv("data/screenshot_coords.csv", index_col=0)
        vox_coords = pd.read_csv("data/vox_coords.csv", index_col=0)
    else:
        i = 0
        scrn_coords = pd.DataFrame(columns=["x", "y", "z", "rot"])
        vox_coords = pd.DataFrame(columns=["x", "y", "z", "rot"])

    # while "Minecraft" not in active_window:
    #     print(active_window)
    #     time.sleep(1)
    #     active_window = pygetwindow.getActiveWindow()

    # print("Minecraft is active window!")
    # User pyautogui to take a screenshot of the Minecraft window

    # def spoof_window_screenshot():
    #     pyautogui.keyDown("command")
    #     pyautogui.keyDown("shift")
    #     pyautogui.keyDown("4")
    #     pyautogui.keyDown("space")
    #     pyautogui.click()
    # TODO: Factor out voxel collection, then figure out after the fact which chunks screenshots should be referring to?
    # chunk_shape = (10, 10, 10)
    # With this rotation, the z axis moves us forward.
    y = 0  # dummy height variable
    while i < num_samples:
        if load and os.path.exists(os.path.join("data", "screenshots", f"{i}.png")):
            i += 1
            continue
        if i % 100 == 0:
            # FIXME: hack to keep weather clear and daylight on, since this function does not seem to have a lasting effect.
            client.initDataGen(Point(x=0, y=0, z=0))  # dummy point variable
        x, z, rot = idx_to_x_z_rot(i)
        x *= 20
        z *= 20
        loc = client.setLoc(Point(x=x, y=y, z=z))
        client.setRot(Point(x=0, y=rot, z=0))
        assert loc.x == x and loc.z == z
        y = loc.y
        scrn_coords.loc[i] = [x, y, z, rot]
        foothold_cube = read_cube(client, (x, y-1, z), (x, y-1, z))
        # Don't overwrite existing screenshots if loading (in case we need to fix a select few that were duds, in which
        # case we'll delete these manually and reset the index in proj.json to an earlier value).
        if foothold_cube.blocks[0].type == WATER:
            print("Water detected at", x, y, z)
            i += 1
            continue
        vox_x, vox_z = get_vox_xz_from_view(x, z, rot)
        voxels = get_vox_chunk(loc=(vox_x, y, vox_z), shp=chunk_shape, client=client)
        # We are hard-coding this chunk to be looking in front of us given our fixed rotation.
        # min = (x - math.floor(chunk_shape[0] // 2), 
        #     y - 2,
        #     # y - math.floor(chunk_shape[1] // 2), 
        #     z + forward_dist)
        # max = (x + math.ceil(chunk_shape[0] // 2) - 1, 
        #     # y + math.ceil(chunk_shape[1] // 2) - 1, 
        #     y + chunk_shape[1] - 1 - 2,
        #     # z + math.ceil(chunk_shape[2] // 2) - 1)
        #     z + forward_dist + chunk_shape[2] - 1)
        vox_coords.loc[i] = [vox_x, y, vox_z, rot]
        if (i + 1) % 100 == 0:
            # Save dataframe to file
            # TODO: Append to the file instead of overwriting
            scrn_coords.to_csv(os.path.join("data", "screenshot_coords.csv"))
            vox_coords.to_csv(os.path.join("data", "vox_coords.csv"))
            print(f"Saved {i + 1} samples")
        # plot_voxels(voxels)
        np.save(os.path.join("data", "voxels", f"{i}.npy"), voxels)
        top_left_corner_screenshot(f"{i}", bbox=BBOX)
        print(f"Saved sreenshot {i}, location {x}, {y}, {z}")
        i += 1
    {"curr_i": i}
    with open("data/screenshots_prog.json", "w") as f:
        json.dump({"curr_i": i}, f)


def main(mode: str = "screenies", num_samples: int = 1000, load: bool = True):
    channel = grpc.insecure_channel('localhost:5001')
    client = src.main.proto.minecraft_pb2_grpc.MinecraftServiceStub(channel)
    # NOTE: num_samples is an upper bound on number of screenshots (since we don't take screenshots from on top of water).
    #  We do take all voxel samples, though.
    if mode == "screenies":
        get_screenies(client=client, num_samples=num_samples, load=load)
    # elif mode == "voxels":
    #     get_voxel_chunks(client, num_samples, load)
    # else:
        raise ValueError("Invalid mode")


if __name__ == "__main__":
    Fire(main)