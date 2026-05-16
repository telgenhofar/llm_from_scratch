#!/bin/bash

#SBATCH --partition=teaching
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --job-name=llm_train_tokenizer
#SBATCH --output=logs/tokenizer-%j.out
#SBATCH --chdir=/home/ad.msoe.edu/telgenhofar/PersonalProjects/LLM_from_scratch

# Train the BPE tokenizer. CPU-only, no GPU needed.

bash --login -c "
  source /etc/profile
  conda activate /home/ad.msoe.edu/telgenhofar/.conda/envs/llm_from_scratch
  cd /home/ad.msoe.edu/telgenhofar/PersonalProjects/LLM_from_scratch
  echo 'host:' \$(hostname)
  echo 'python:' \$(which python)
  python train_tokenizer.py
"