#!/bin/bash

# Define a function to terminate a screen
terminate_screen () {
    screen -S "$1" -X quit
}

# Terminate the Python scripts running in screens
terminate_screen "carla_manager"
terminate_screen "pylot_manager"
terminate_screen "main"

docker exec -it pylot bash -c "ps -ef | grep configs/ | awk '{print \$2}' | xargs -r kill -9" > /dev/null 2>&1