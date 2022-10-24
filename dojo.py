from einops import rearrange
from PIL import Image
from typing import List
import numpy as np
import minedojo

def setup_env():
    env = minedojo.make(
        "open-ended",
        image_size=512,
        start_position=dict(x=0, y=4, z=0, yaw=0, pitch=0),
        generate_world_type="flat"
    )
    _ = env.reset()

    for _ in range(20):
        # warm up env
        env.step(env.action_space.no_op())
    return env

class Structure:
    def __init__(self, arr: np.array, blocks: List[str]):
        self.arr = arr        
        self.blocks = blocks

    def rotate90(self, axis: int = 0):
        self.arr = np.rot90(self.arr, 1, axes=(0,2))

class Faces:
    def __init__(self, faces):
        self.faces = faces

    def images(self):
        return [
            Image.fromarray(rearrange(face[0]['rgb'], 'c w h -> w h c')) for face in self.faces
        ]

class MineDojoSpawner:
    def __init__(self, z_offset: int = 7):
        self.env = setup_env()
        self.z_offset = z_offset

    def spawn(self, structure: Structure):
        if self.env is None:
            self.env = setup_env()

        arr = structure.arr
        x_half = arr.shape[0] // 2

        for (x,y,z), value in np.ndenumerate(arr):
            coord = np.array([x,y,z])
            coord[0] -= x_half
            coord[-1] += self.z_offset
            block_type = structure.blocks[int(value)]
            self.env.set_block(block_type, coord)

        # capture output
        out = self.env.step(self.env.action_space.no_op())

        # remove blocks
        for (x,y,z), value in np.ndenumerate(arr):
            coord = np.array([x,y,z])
            coord[0] -= x_half
            coord[-1] += self.z_offset
            block_type = structure.blocks[int(value)]
            self.env.set_block("air", coord)
        return out

    def capture_rotations(self, structure: Structure) -> Faces:
        faces =[]
        for _ in range(4):
            faces.append(self.spawn(structure))
            structure.rotate90()
        return Faces(faces=faces)

    def close(self):
        self.env.close()
        self.env = None


# Example Usage
if __name__ == "__main__":
    spawner = MineDojoSpawner()

    z = np.zeros((5, 4, 6))
    z[:, :, -1] = 3
    z[0, :, :] = 2
    z[-1, :, :] = 1

    z = z.astype(int)

    structure = Structure(z, blocks = ["diamond_ore", "obsidian", "dirt", "stone"])

    # capture faces
    faces = spawner.capture_rotations(structure)

    # images
    # PIL Images, so we can just save with images[0].save("img.jpg")
    images = faces.images()

    # close env
    spawner.close()
