#!/bin/bash
#SBATCH --job-name=falcon-40b-mn
#SBATCH --account=glucas_540
#SBATCH --partition=gpu

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1

#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:v100:2

#SBATCH --time=00:10:00

#SBATCH --output=TEST_multinode_falcon.%j.out

# Change to proper directory
cd ..

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO

# setup
module purge
module load conda
eval "$(conda shell.bash hook)"
conda activate falcon

# run
echo ""
echo "Running test predictions with falcon 40B on multiple slurm nodes"
echo ""

srun torchrun \
--nnodes 2 \
--nproc_per_node 2 \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint $head_node_ip:29500 \
falcon_40b_mn.py

echo "DONE"