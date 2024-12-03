#!/bin/bash
#SBATCH --job-name=finetuning_gpu
#SBATCH --mail-user=zoudj@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --account=stats_dept1
#SBATCH --partition=gpu
#SBATCH --nodes=1                           # Number of nodes
#SBATCH --ntasks=1                          # Number of tasks
#SBATCH --gres=gpu:1                        # Request 1 GPU
#SBATCH --cpus-per-task=2                   # Number of CPU cores per task
#SBATCH --mem=4G                           # Total memory per node
#SBATCH --time=04:00:00                     # Time limit (adjust if needed)
#SBATCH --output=logs/finetuning_gpu_output_%j.log    # Standard output and error log file (%j will be replaced by job ID)

echo "Starting job $SLURM_JOB_ID at $(date)"
# Load necessary modules
echo "Loading CUDA module..."
module load cuda/11.2.2                  # Load CUDA module, adjust version to match cluster setup

# Activate the Conda environment
echo "Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate srsd                    # Activate the conda environment (adjust name if needed)
conda list
echo "Printing the content of finetuning_easy.py:"
# cat finetuning_easy.py

echo "Python version and path being used:"
python --version
which python
conda info --envs


echo "Running Python script..."
python finetuning_easy.py
conda deactivate
