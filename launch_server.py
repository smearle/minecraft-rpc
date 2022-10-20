import os
import re

from fire import Fire

# HACK: For some reason unable to have distinct servers for data-collection and
# testing/rendering??? So we'll just edit the server properties and force-regenerate the world when changing modes lol.
SERVER_PROPERTIES_PATH = "server/server.properties"

def launch_server(level_type="default"):
    trg_level_type = level_type
    # Read contents of server.properties
    with open(SERVER_PROPERTIES_PATH, "r") as f:
        content = f.read()
    # User regular expression to check if level-type=default
    if not re.search(f"level-type={trg_level_type}", content):
        content = re.sub(r'level-type=\w+', f'level-type={trg_level_type}', content)
        # Write contents of server.properties
        with open(SERVER_PROPERTIES_PATH, "w") as f:
            f.writelines(content)
        # Remove server/world folder.
        os.system("rm -rf server/world")
    # Launch the server.
    os.system("cd server && java -jar spongevanilla-1.12.2-7.3.0.jar")


if __name__ == "__main__":
    # Launch the server
    Fire(launch_server)