#!/bin/bash
#SBATCH --account=glucas_540
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00

# Change to proper directory
cd ..

# setup
module purge
module load conda
eval "$(conda shell.bash hook)"
conda activate falcon

# run
echo ""
echo "Running test predictions with model size ${MODEL_SIZE}"
echo ""

python3 falcon_7b.py

echo "DONE"