#!/bin/bash

#SBATCH --partition=teaching
#SBATCH --cpus-per-task=16
#SBATCH --time=8:00:00
#SBATCH --job-name=llm_prep_data
#SBATCH --output=logs/prep-%j.out
#SBATCH --chdir=/home/ad.msoe.edu/telgenhofar/PersonalProjects/LLM_from_scratch

# Download and tokenize a dataset. Runs on the CPU teaching partition because
# tokenization doesn't need a GPU — no point burning H100 hours on it.
#
# Usage:
#   sbatch prepare_data.sh configs/data/tinystories.yaml
#   sbatch prepare_data.sh configs/data/fineweb_edu.yaml

CONFIG=${1:?usage: sbatch prepare_data.sh <path/to/data.yaml>}

bash --login -c "
  source /etc/profile
  conda activate /home/ad.msoe.edu/telgenhofar/.conda/envs/llm_from_scratch
  cd /home/ad.msoe.edu/telgenhofar/PersonalProjects/LLM_from_scratch
  echo 'host:' \$(hostname)
  echo 'config:' ${CONFIG}
  python prepare_data.py --config ${CONFIG} --tokenizer tokenizer.json --num_proc 16
"