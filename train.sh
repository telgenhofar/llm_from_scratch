#!/bin/bash

#SBATCH --partition=dgx
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --time=24:00:00
#SBATCH --job-name=llm_train
#SBATCH --output=logs/train-%j.out
#SBATCH --chdir=/home/ad.msoe.edu/telgenhofar/PersonalProjects/LLM_from_scratch

# Train the LLM. Resumes from checkpoints/ckpt.pt if it exists.
#
# Usage:
#   sbatch train.sh configs/runs/tinystories.yaml
#   sbatch train.sh configs/runs/fineweb_small.yaml

CONFIG=${1:?usage: sbatch train.sh <path/to/run.yaml>}

bash --login -c "
  source /etc/profile
  conda activate /home/ad.msoe.edu/telgenhofar/.conda/envs/llm_from_scratch
  cd /home/ad.msoe.edu/telgenhofar/PersonalProjects/LLM_from_scratch
  echo 'host:' \$(hostname)
  echo 'config:' ${CONFIG}
  echo 'python:' \$(which python)
  nvidia-smi
  python LLM.py --config ${CONFIG} --out_dir checkpoints
"