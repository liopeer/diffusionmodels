#!/bin/bash
#SBATCH  --output=log/%j.out
#SBATCH  --error=log/%j.err
#SBATCH  --gres=gpu:a6000:1
#SBATCH  --job-name=infer

conda activate liotorch
mkdir log
python -u SDS_guidance.py "$@"