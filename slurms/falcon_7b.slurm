#!/bin/bash
#SBATCH --account=glucas_540
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:20:00

# Change to proper directory
cd ..

# setup
module purge
module load conda
eval "$(conda shell.bash hook)"
conda activate falcon

# run
echo "Running falcon_7b.py"

python3 falcon_7b.py

echo "DONE"