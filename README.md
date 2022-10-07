
# Mine Fusion

## Data generation
Make sure you have java installed, then launch a modded Minecraft server with:
```
cd server
java -jar spongevanilla-1.12.2-7.3.0.jar 
``` 

Now open minecraft. Download and install version `1.12.2`. Launch the game, select Multiplayer, Direct Connect, and join the server `localhost`.

Now you can run:
```
python clients/python/data_gen.py
```
This will generate a dataset of [screenshot <--> block layout] pairs by teleporting your player around the map in a square spiral, recording a chunk of blocks roughly in front of you, then taking a screenshot.

The latter is done very hackishly via screen captures. You will need to tweak `BBOX` in this script to ensure the screenshots contain your Minecraft window, which you will want to stick in, e.g., the top-left corner of your screen. For Mac users, you can press `cmd`+`shift`+`4` to enter screenshot mode, then put the mouse at the bottom right of the Minecraft window to have the coordinates in pixels displayed.



Change `options.text`, (in your minecraft directory. Mac OS: `~/Library/Application Support/minecraft/options.txt`), settings `OverrideWidth:512` and `OverrideHeight:512` so that we collect screenshots of an approriate size. You'll probably also want to edit the line `pauseOnLostFocus:false` so that you can change focus from the Minecraft client while collecting data, without bringing up the pause screen.

You will also need to change the name of the player being teleported around in `MinecraftRPC.java` to match your player name (or change your player name to "boopchie"). See "Build the Minecraft Mod" below.

## Training
_Under construction_

```
python train.py
```

We use hydra. The config can be found at `conf/config.yaml`

**TODO: Checkpointing, validation, loading, etc.**

See below for details about how we have hacked `minecraft-rpc` to interact with a modded Minecraft server via python.

# Minecraft RPC (Havocraft)
A [gRPC](https://grpc.io) interface for Minecraft. 

We will be editing `./src/main/proto/minecraft.proto` and `./src/main/java/dk/itu/real/ooe/services/MinecraftService.java`, to add new python commands via which we can interact with our modded Minecraft server. (E.g., we'd like to have a command to remove all collectible items, and add certain block types.) We will test functionality in `clients/python/example.py`.


## Install Dependencies

### Java (for building the mod)
* Install Java 1.8 (You can check your version with `java -version`)
  * unix: `sudo apt-get install openjdk-8-jre`
  * osx: [mkyong.com/java/how-to-install-java-on-mac-osx](https://mkyong.com/java/how-to-install-java-on-mac-osx/)  

### Python (for compiling grpc)
After creating a new conda environment, install dependencies:
```
python -m pip install grpcio
python -m pip install grpcio-tools
```

## Build the Minecraft Mod
Modify the mod:
* Go to `./src/main/java/dk/itu/real/ooe/services/MinecraftService.java`
* Overwrite your favorite function :D
* Build the minecraft mod as a jar:
```
mvn install
```
The `minecraft-rpc-0.0.5.jar` will be available in `target/` after this. Put this `jar` file inside `./server/mods`
```
mv target/minecraft-rpc-0.0.5.jar server/mods/
```

## Overwrite and Compile the grpc script (python api)
* Add your favorite Minecraft item in `./minecraft-rpc/src/main/proto/minecraft.proto`
* Navigate to the `./` of this repo. Then generate the grpc python files as follow:
```
python -m grpc_tools.protoc -I./ --python_out=./clients/python/ --grpc_python_out=./clients/python/ ./src/main/proto/minecraft.proto
```
The `minecraft_pb2_grpc.py` and `minecraft_pb2.py` will be available in `./clients/python/src/main/proto` folder. You can copy them into the folder in which you place the `mc_render.py` (or any other python script you're working on).


## One-line compilation

Alternatively, here is a one-line command for rebuilding the mode after changes are made:
```
python -m grpc_tools.protoc -I./ --python_out=./clients/python/ --grpc_python_out=./clients/python/ ./src/main/proto/minecraft.proto && mvn install && mv target/minecraft-rpc-0.0.5.jar server/mods/
```


## Run 
* Go to `./server`
* Start the server with 
```
java -jar spongevanilla-1.12.2-7.3.0.jar
```
* The first time you start the server you must accept the Minecraft EULA by modifying eula.txt
* You can change the properties of the world by modifying `server.properties`
* You should see a bunch of output including `[... INFO] [minecraft_rpc]: Listening on 5001`. 
This means it's working and the server is ready for commands on port 5001.

Note that if you run `java -jar ./server/spongevanilla-1.12.2-7.3.0.jar` inside the `./` folder, new `eula.txt` and `server.properties` will be generated and applied. So we recommand you not being too lazy to go to `./server` folder :D


# Reference 
For more information, see [Evocraft](https://github.com/real-itu/Evocraft-py) and the original [Minecraft RPC](https://github.com/real-itu/minecraft-rpc)