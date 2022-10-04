import math
import os
from pdb import set_trace as TT
import time
import grpc
import matplotlib.pyplot as plt
import numpy as np
import pyautogui
import pyscreenshot as ImageGrab
import pygetwindow

import clients.python.src.main.proto.minecraft_pb2_grpc
from clients.python.src.main.proto.minecraft_pb2 import *

BBOX = (0, 66, 853, 544)  # For 16" MBP, M1.
# bbox = pyautogui.locateOnScreen('mc.png')
# TT()

active_window = pygetwindow.getActiveWindow()
# Continue when user enters "y"
input("Put the Minecraft window in the top corner of the screen (do not scale it!). Enter the server (localhost) and " +
    " enter camera mode (F1). Press enter to continue...")
print("Now change focus to the Minecraft window. (User, e.g. alt + tab. Do not move the window!)")

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

def top_left_corner_screenshot(name: str):
    im = ImageGrab.grab(bbox=BBOX)
    # Save image to file
    im.save(os.path.join("data", "screenshots", f"{name}.png"))


channel = grpc.insecure_channel('localhost:5001')
client = clients.python.src.main.proto.minecraft_pb2_grpc.MinecraftServiceStub(channel)

def square_spiral(n: int):
    # Parameterize square spiral as on https://math.stackexchange.com/a/3158068
    flr_sqrt_n = math.floor(math.sqrt(n))
    n_hat = flr_sqrt_n if flr_sqrt_n % 2 == 0 else flr_sqrt_n - 1
    n_hat_sqr = n_hat ** 2
    if n_hat_sqr <= n <= n_hat_sqr + n_hat:
        x = - n_hat / 2 + n - n_hat_sqr
        y = n_hat / 2
    elif n_hat_sqr + n_hat < n <= n_hat_sqr + 2 * n_hat + 1:
        x = n_hat / 2
        y = n_hat / 2 - n + n_hat_sqr + n_hat
    elif n_hat_sqr + 2 * n_hat + 1 < n <= n_hat_sqr + 3 * n_hat + 2:
        x = n_hat / 2 - n + n_hat_sqr + 2 * n_hat + 1
        y = - n_hat / 2 - 1
    elif n_hat_sqr + 3 * n_hat + 2 < n <= n_hat_sqr + 4 * n_hat + 3:
        x = - n_hat / 2 - 1
        y = - n_hat / 2 - 1 + n - n_hat_sqr - 3 * n_hat - 2
    else: raise Exception
    return int(x), int(y)



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
while True:
    if i % 100 == 0:
        # FIXME: hack to keep weather clear and daylight on, since this functiondoes not seem to have a lasting effect.
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
    np.save(os.path.join("data", "voxels", f"{i}.npy"), voxels)
    top_left_corner_screenshot(f"{i}")
    i += 1