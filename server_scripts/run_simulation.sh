#!/bin/bash
#SBATCH --job-name=traffic_simulation
#SBATCH --qos=test
#SBATCH --mem=32G
#SBATCH --cpus-per-task=10
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1

PATH_PROGRAM="/home/$USER/Milestone-One"
PUT_PROGRAM_TO="/app"
PATH_SUMO_CONTAINER="/shared/sets/singularity/sumo_docker.sif"
CMD_PATH="/home/$USER/Milestone-One/server_scripts/cmd_container.sh"
PRINTS_SAVE_PATH="/home/$USER/Milestone-One/server_scripts/container_printouts/output_$SLURM_JOB_ID.txt"

# Run container by adding code by binding, run commands from cmd_container.sh, save printouts to a file
singularity exec --bind "$PATH_PROGRAM":"$PUT_PROGRAM_TO" "$PATH_SUMO_CONTAINER" /bin/bash "$CMD_PATH" > "$PRINTS_SAVE_PATH" 2>&1