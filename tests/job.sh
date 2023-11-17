#!/bin/bash
#SBATCH  --account=student
#SBATCH  --output=log/%j.out
#SBATCH  --error=log/%j.err
#SBATCH  --gres=gpu:1
#SBATCH  --mem=32G
#SBATCH  --job-name=mnist_double
#SBATCH  --constraint='titan_xp|geforce_gtx_titan_x'

source /scratch_net/biwidl311/peerli/conda/etc/profile.d/conda.sh
conda activate liotorch
mkdir log
python -u train_generative.py "$@"