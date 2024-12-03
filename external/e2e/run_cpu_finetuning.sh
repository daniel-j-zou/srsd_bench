#!/bin/bash
#SBATCH --job-name=finetuning
#SBATCH --mail-user=zoudj@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --account=stats_dept1
#SBATCH --partition=standard
#SBATCH --nodes=1                           # Number of nodes
#SBATCH --ntasks=1                          # Number of tasks
#SBATCH --cpus-per-task=4                   # Number of CPU cores per task
#SBATCH --mem=8G                           # Total memory per node
#SBATCH --time=00:20:00                     # Time limit (adjust if needed)
#SBATCH --output=logs/finetuning_output_%j.log    # Standard output and error log file (%j will be replaced by job ID)

# Load necessary modules
module load cuda/11.2.2                  # Load CUDA module, adjust version to match cluster setup

# Activate the Conda environment
eval "$(conda shell.bash hook)"
conda activate srsd                    # Activate the conda environment (adjust name if needed)
python finetuning_cpu.py
conda deactivate
