import math
import os
from pdb import set_trace as TT
import time
import grpc
import matplotlib.pyplot as plt
import numpy as np
import pyscreenshot as ImageGrab
import pygetwindow

# import sys
# sys.path.append('../..')

from utils import square_spiral
# from clients.python.utils import square_spiral

import src.main.proto.minecraft_pb2_grpc
from src.main.proto.minecraft_pb2 import *

BBOX = (0, 66, 853, 544)  # For 16"/14" MBP, M1.
# bbox = pyautogui.locateOnScreen('mc.png')

VOXELS_FOLDER = os.path.join("..", "..", "data", "voxels")
PIC_FOLDER = os.path.join("..", "..", "data", "screenshots")
active_window = pygetwindow.getActiveWindow()
# Continue when user enters "y"
input("Put the Minecraft window in the top corner of the screen (do not scale it!). Enter the server (localhost) and " +
    " enter camera mode (F1). Press enter to continue...")
print("Now change focus to the Minecraft window. (User, e.g. alt + tab. Do not move the window!)")

while "Minecraft" not in active_window:
    print("Please switch to the Minecraft window, now is {}".format(active_window))
    time.sleep(1)
    active_window = pygetwindow.getActiveWindow()

print("Minecraft is active window now!")
# User pyautogui to take a screenshot of the Minecraft window

# def spoof_window_screenshot():
#     pyautogui.keyDown("command")
#     pyautogui.keyDown("shift")
#     pyautogui.keyDown("4")
#     pyautogui.keyDown("space")
#     pyautogui.click()

def top_left_corner_screenshot(name: str):
    im = ImageGrab.grab(bbox=BBOX)
    # Save image to file
    im.save(os.path.join(PIC_FOLDER, f"{name}.png"))


channel = grpc.insecure_channel('localhost:5001')
client = src.main.proto.minecraft_pb2_grpc.MinecraftServiceStub(channel)



def spawn_something(i, j, k, border_size=(1, 1, 1), base_pos=80,\
                    boundary_size=3, backgroud_type=STONE):
    '''
    Spawn a cube of size (i, j, k) at (base_pos, base_pos, base_pos), just for debug purpose. Copy from previous mc_render.py
    '''
    # render the base
    client.fillCube(FillCubeRequest(
        cube=Cube(
            min=Point(x=-boundary_size-border_size[0] + 2, 
                      y=base_pos - 2, z=-boundary_size-border_size[1] + 2),
            max=Point(x=i + boundary_size + border_size[0] - 3,
                      y=base_pos - 1, z=j + boundary_size + border_size[1] - 3)
        ),
        type=backgroud_type
    ))
# spawn_something(10, 10, 10)
# TT()

def cube_to_voxels(cube: Cube, shape: tuple, min: tuple):
    voxels = np.zeros(shape)
    for block in cube.blocks:
        if block.type != AIR:
            voxels[block.position.x - min[0], block.position.y - min[1], block.position.z - min[2]] = 1
    return voxels


def read_cube(client, min: tuple, max: tuple):
    return client.readCube(Cube(min=Point(x=min[0], y=min[1], z=min[2]),
                                max=Point(x=max[0], y=max[1], z=max[2])))


chunk_shape = (10, 10, 10)
# With this rotation, the z axis moves us forward.
client.setRot(Point(x=0, y=0, z=0))
i = 0
y = 0  # dummy height variable
forward_dist = 5
while i <= 200:
    if i % 100 == 0:
        # FIXME: hack to keep weather clear and daylight on, since this function does not seem to have a lasting effect.
        client.initDataGen(Point(x=0, y=0, z=0))  # dummy point variable
    x, z = square_spiral(i)
    x *= 20
    z *= 20
    loc = client.setLoc(Point(x=x, y=y, z=z))
    assert loc.x == x and loc.z == z
    y = loc.y
    foothold_cube = read_cube(client, (x, y-1, z), (x, y-1, z))
    if foothold_cube.blocks[0].type == WATER:
        print("Water detected at", x, y, z)
        i += 1
        continue
    # We are hard-coding this chunk to be looking in front of us given our fixed rotation.
    min = (x - math.floor(chunk_shape[0] // 2), 
        y - 2,
        # y - math.floor(chunk_shape[1] // 2), 
        z + forward_dist)
    max = (x + math.ceil(chunk_shape[0] // 2) - 1, 
        # y + math.ceil(chunk_shape[1] // 2) - 1, 
        y + chunk_shape[1] - 1 - 2,
        # z + math.ceil(chunk_shape[2] // 2) - 1)
        z + forward_dist + chunk_shape[2] - 1)
    cube = read_cube(client, min, max)
    voxels = cube_to_voxels(cube, chunk_shape, min)
    # plot_voxels(voxels)
    # TT()
    np.save(os.path.join(VOXELS_FOLDER, f"{i}.npy"), voxels)
    top_left_corner_screenshot(f"{i}")
    i += 1