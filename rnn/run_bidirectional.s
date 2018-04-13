#!/bin/bash
#
#SBATCH --job-name=rnn_timeseries
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=40GB
#SBATCH --output=slurm-%A.out
#SBATCH --error=slurm-%A.err

# All the load commands below may not be required by everyone.
# The commands below can be modified/changed/removed  based on the python version you have
module purge
module load python3/intel/3.5.3
module load pytorch/python3.5/0.2.0_3
module load torchvision/python3.5/0.1.9



python3 -u bidirectional_rnn_train.py

