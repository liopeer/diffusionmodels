#!/bin/bash
#SBATCH  --output=log/%j.out
#SBATCH  --gres=gpu:2
#SBATCH  --mem=10G
#SBATCH  --job-name=mnist_double
#SBATCH  --constraint='titan_xp'

source /scratch_net/biwidl311/peerli/conda/etc/profile.d/conda.sh
conda activate liotorch
mkdir log
python -u train_parallel.py "$@"
