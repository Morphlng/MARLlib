#!/bin/bash

# Define a function to terminate a screen, apply for single process
terminate_screen () {
    screen -S "$1" -X quit
}

# Define a function to terminate a screen, apply for multiple processes
terminate_screen_thoroughly() {
    # Send SIGTERM to gracefully terminate processes
    screen -S "$1" -X stuff "^C"

    # Give processes some time to terminate gracefully
    sleep 1

    # Finally, terminate the screen session
    terminate_screen "$1"
}

# Terminate the Python scripts running in screens
terminate_screen "carla_manager"
terminate_screen "pylot_manager"
terminate_screen_thoroughly "main"
terminate_screen_thoroughly "eval"

docker exec -it pylot bash -c "ps -ef | grep configs/ | awk '{print \$2}' | xargs -r kill -9" > /dev/null 2>&1