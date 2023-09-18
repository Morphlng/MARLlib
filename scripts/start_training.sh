#!/bin/bash

# Default values
START_DOCKER=true
CONDA_ENV_NAME="marllib"

# Parse command-line arguments
while getopts ":d:e:" opt; do
  case $opt in
    d) 
      START_DOCKER=$OPTARG
      ;;
    e)
      CONDA_ENV_NAME=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

# Start docker containers if required
if [ "$START_DOCKER" == "true" ]; then
    docker start pylot
    docker start redis
    echo "Docker containers started"
fi

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV_NAME
echo "Conda environment $CONDA_ENV_NAME activated"

# Define a function to run a command in a new screen
run_in_new_screen () {
    screen -dmS "$1" bash -c "$2"
}

current_time=$(date +"%Y-%m-%d_%H:%M:%S")

# Get the directory of the current script
SCRIPT_DIR=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")

# Run the Python scripts in separate screens
run_in_new_screen "carla_manager" "python ${SCRIPT_DIR}/../marllib/envs/base_env/carla_manager.py"
run_in_new_screen "pylot_manager" "python ${SCRIPT_DIR}/../marllib/envs/base_env/pylot_manager.py"
run_in_new_screen "main" "python ${SCRIPT_DIR}/../marllib/main.py > ~/${current_time}_train.log 2>&1"
