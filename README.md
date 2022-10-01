



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

## Overwrite and Compile the grpc script
* Add your favorite Minecraft item in `./minecraft-rpc/src/main/proto/minecraft.proto`
* Navigate to the `./` of this repo. Then generate the grpc python files as follow:
```
python -m grpc_tools.protoc -I./ --python_out=./clients/python/ --grpc_python_out=./clients/python/ ./src/main/proto/minecraft.proto
```
The `minecraft_pb2_grpc.py` and `minecraft_pb2.py` will be available in `./clients/python/src/main/proto` folder. You can copy them into the folder in which you place the `mc_render.py` (or any other python script you're working on).

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

## Example script
To run the example script, run:
```
python clients/python/example.py
```
You will need to make sure the latest `minecraft_pb2_grpc.py` and `minecraft_pb2.py` generated above are in the same directory as the example script.

# Reference 
For more information, see [Evocraft](https://github.com/real-itu/Evocraft-py) and the original [Minecraft RPC](https://github.com/real-itu/minecraft-rpc)