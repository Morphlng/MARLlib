#!/bin/bash

# Start docker containers
docker start pylot
docker start redis
echo "Docker containers started"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate marllib
echo "Conda environment activated"

# Define a function to run a command in a new screen
run_in_new_screen () {
    screen -dmS "$1" bash -c "$2"
}

# Run the Python scripts in separate screens
run_in_new_screen "carla_manager" "python ${HOME}/code/MARLlib/marllib/envs/base_env/carla_manager.py"
run_in_new_screen "pylot_manager" "python ${HOME}/code/MARLlib/marllib/envs/base_env/pylot_manager.py"
run_in_new_screen "main" "python ${HOME}/code/MARLlib/marllib/main.py"
