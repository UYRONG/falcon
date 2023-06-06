#!/bin/bash
#SBATCH --account=glucas_540
#SBATCH --partition=gpu
#SBATCH --gpus-per-task=2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100:1
#SBATCH --gpus-per-task=a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

# Change to proper directory
cd ..

# setup
module purge
module load conda
eval "$(conda shell.bash hook)"
conda activate falcon

# run
echo "running"
echo ""

python3 falcon_40b.py

echo "DONE"