import random
import grpc

import minecraft_pb2_grpc
from minecraft_pb2 import *

channel = grpc.insecure_channel('localhost:5001')
client = minecraft_pb2_grpc.MinecraftServiceStub(channel)

client.fillCube(FillCubeRequest(  # Clear a 20x10x20 working area
    cube=Cube(
        min=Point(x=-10, y=4, z=-10),
        max=Point(x=10, y=100, z=10)
    ),
    type=AIR
))
client.spawnBlocks(Blocks(blocks=[  # Spawn a flying machine
    # Lower layer
    Block(position=Point(x=0, y=4, z=0), type=PISTON, orientation=NORTH),
    # Block(position=Point(x=1, y=5, z=0), type=SLIME, orientation=NORTH),
    # Block(position=Point(x=1, y=5, z=-1), type=STICKY_PISTON, orientation=SOUTH),
    # Block(position=Point(x=1, y=5, z=-2), type=ENTITY_COW, orientation=NORTH),
    # Block(position=Point(x=1, y=5, z=-4), type=SLIME, orientation=NORTH),
    # # Upper layer
    # Block(position=Point(x=1, y=6, z=0), type=REDSTONE_BLOCK, orientation=NORTH),
    # Block(position=Point(x=1, y=6, z=-4), type=REDSTONE_BLOCK, orientation=NORTH),
    # # Activate
    # Block(position=Point(x=1, y=6, z=-1), type=QUARTZ_BLOCK, orientation=NORTH),
]))

while True:
    client.spawnEntities(SpawnEntities(spawnEntities=[
        SpawnEntity(type=ENTITY_CHICKEN, spawnPosition=Point(x=random.randint(0, 10), y=random.randint(4, 6), z=random.randint(0, 10)))]
        # [Entity(position=Point(x=random.randint(0, 100), y=random.randint(0, 100), z=random.randint(0, 100)), type=random.randint(0, 90))]
        # [SpawnEntity()]
    ))
    # client.spawnBlocks(Blocks(blocks=
    #     [Block(position=Point(x=random.randint(0, 100), y=random.randint(0, 100), z=random.randint(0, 100)), type=random.randint(0, 90), orientation=NORTH)]
    # ))
    # client.spawnBlocks(Blocks(blocks=
    #     [Block(position=Point(x=random.randint(0, 100), y=random.randint(0, 100), z=random.randint(0, 100)), type=random.randint(0, 90), orientation=NORTH)]
    # ))
# client.spawnEntities(SpawnEntities(spawnEntities=[
#         SpawnEntity(type=ENTITY_CHICKEN, spawnPosition=Point(x=0, y=6, z=0))]
#         # [Entity(position=Point(x=random.randint(0, 100), y=random.randint(0, 100), z=random.randint(0, 100)), type=random.randint(0, 90))]
#         # [SpawnEntity()]
# ))