mvn compile&&mvn package&&py -m grpc_tools.protoc -I=src/main/proto --python_out=clients/python --grpc_python_out=clients/python minecraft.proto&&cp clients/python/minecraft_pb2.py ../../../../Desktop/mcb/minecraft_pb2.py&&cp clients/python/minecraft_pb2_grpc.py ../../../../Desktop/mcb/minecraft_pb2_grpc.py&&cp target/minecraft-rpc-0.0.5.jar ../../../../Desktop/mcb/mods/minecraft-rpc-0.0.5.jar
