# LLM from scratch

## Setup
1. Train tokenizer: `python train_tokenizer.py`
2. Prepare data: `sbatch prepare_data.sh configs/data/tinystories.yaml`
3. Train: `sbatch train.sh configs/runs/tinystories.yaml`
4. Chat: `python chat.py --checkpoint checkpoints/ckpt_best.pt`

## Stages
- Stage 1: TinyStories (pipeline check, ~2h on H100)
- Stage 2: FineWeb-Edu 10B sample (~24-36h)
- Stage 3: FineWeb-Edu 100B sample (~3-5 days, multiple jobs)