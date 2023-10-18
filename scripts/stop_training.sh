#!/bin/bash

# Define a function to terminate a screen, apply for single process
terminate_screen () {
    screen -S "$1" -X quit
}

# Define a function to terminate a screen, apply for multiple processes
terminate_screen_thoroughly() {
    # Send SIGTERM to gracefully terminate processes
    screen -S "$1" -X stuff "^C"
    sleep 2  # Give processes some time to terminate gracefully

    # If processes are still running, send SIGKILL
    for pid in $(pgrep -f "$1"); do
        # Kill the process and its child processes
        pkill -TERM -P $pid
        sleep 1  # Give processes some time to terminate gracefully
        pkill -KILL -P $pid
    done

    # Finally, terminate the screen session
    screen -S "$1" -X quit
}

# Terminate the Python scripts running in screens
terminate_screen "carla_manager"
terminate_screen "pylot_manager"
terminate_screen_thoroughly "main"

docker exec -it pylot bash -c "ps -ef | grep configs/ | awk '{print \$2}' | xargs -r kill -9" > /dev/null 2>&1