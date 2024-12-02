#!/bin/bash
#SBATCH --job-name=srsd_test_all
#SBATCH --mail-user=zoudj@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --account=stats_dept1
#SBATCH --partition=gpu
#SBATCH --nodes=1                           # Number of nodes
#SBATCH --ntasks=1                          # Number of tasks
#SBATCH --gres=gpu:1                        # Request 1 GPU
#SBATCH --cpus-per-task=4                   # Number of CPU cores per task
#SBATCH --mem=2G                           # Total memory per node
#SBATCH --time=05:20:00                     # Time limit (adjust if needed)
#SBATCH --output=logs/batch_gpu_output_%j.log    # Standard output and error log file (%j will be replaced by job ID)

# Load necessary modules
module load cuda/11.2.2                       # Load CUDA module, adjust version to match cluster setup

# Activate the Conda environment
eval "$(conda shell.bash hook)"
conda activate srsd                    # Activate the conda environment (adjust name if needed)

# Define checkpoint file
CKPT_FILE=./resource/ckpt/model_original.pt

# Iterate over dataset groups: easy, medium, hard
for group_name in easy medium hard; do
    # Define output directory
    OUT_DIR=./e2e-no_constants/srsd-feynman_${group_name}
    mkdir -p ${OUT_DIR}                     # Create the output directory if it doesn't exist

    # Iterate over all training files in the dataset group
    for filepath in ../../resource/datasets/srsd/${group_name}_set/train/*; do
        echo ${filepath}
	FILE_NAME=$(basename ${filepath})
	TRAIN_FILE=${filepath}
	TEST_FILE=${filepath}
	echo ${filepath}
        # Run runner.py with a timeout of 5 minutes for each dataset file
        timeout 10m python runner.py --train ${TRAIN_FILE} --test ${TEST_FILE} --ckpt ${CKPT_FILE} --out ${OUT_DIR}/${FILE_NAME}
        echo ${filepath}   
    done
done

conda deactivate
