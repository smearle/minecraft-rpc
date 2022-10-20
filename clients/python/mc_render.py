import os
from pdb import set_trace as TT

import grpc


# HACK: auto-generated proto files do relative imports. Will this hack cause a conflict with src/main ??
# if "PATH" not in os.environ: 
#     os.environ["PATH"] = "clients/python"
# elif "clients/python" not in os.environ["PATH"]:
#     os.environ["PATH"] += ":clients/python"
# print(os.environ["PATH"])
# os.chdir("clients/python")

# HACK: Replace relative import
with open("clients/python/src/main/proto/minecraft_pb2_grpc.py", "r") as f:
    contents = f.read()
contents = contents.replace("from src.main.proto", "from clients.python.src.main.proto")
# Write
with open("clients/python/src/main/proto/minecraft_pb2_grpc.py", "w") as f:
    f.write(contents)

import clients
import clients.python.src.main.proto.minecraft_pb2_grpc
from clients.python.src.main.proto.minecraft_pb2 import *

def mc_render(preds, labels):
    # input("Make sure you are running the server from `server_test`, or you may ruin the world from which we collect " 
    # "data (in which case, delete the `server/world` folder and re-launch the data-gen server). Press enter to continue.")
    channel = grpc.insecure_channel('localhost:5001')
    client = clients.python.src.main.proto.minecraft_pb2_grpc.MinecraftServiceStub(channel)
    clear(client)

    client.initDataGen(Point(x=0, y=0, z=0))  # dummy point variable
    loc = client.setLoc(Point(x=-5, y=0, z=0))  # dummy point variable
    y0 = loc.y
    xi, yi, zi = preds[0].shape[1:]

    for i in range(len(preds)):
        pred, label = preds[i], labels[i]
        pred = pred.argmax(0)
        label = label.argmax(0)
        render_blocks(client, (i*2*xi, y0, 0), pred)
        render_blocks(client, ((i*2+1)*xi, y0, 0), label)

def render_blocks(client, orig, blocks):
    x0, y0, z0 = orig
    block_lst = []
    for k in range(len(blocks)):
        for j in range(len(blocks[k])):
            for i in range(len(blocks[k][j])):
                block = int((blocks[k][j][i]))
                block_lst.append(Block(position=Point(x=x0+i, y=y0+j,  z=z0+k),
                                    type=block, orientation=NORTH))
    client.spawnBlocks(Blocks(blocks=block_lst))


def clear(client):
    # clear(20,20)
    client.fillCube(FillCubeRequest(
        cube=Cube(
            min=Point(x=-50, y=3, z=-50),
            max=Point(x=500, y=3, z=50)
        ),
        type=GRASS
    ))
    client.fillCube(FillCubeRequest(
        cube=Cube(
            min=Point(x=-50, y=4, z=-50),
            max=Point(x=500, y=90, z=50)
        ),
        type=AIR
    ))
