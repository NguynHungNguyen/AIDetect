#!/bin/bash
#SBATCH --partition=skylake-gpu
#SBATCH --nodes=2
#SBATCH --gres=gpu:a100:4 
#SBATCH --job-name=aidetect
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --time=6-00:00:00


# ==== Setup ====
echo "Activating environment..."
source ~/.bashrc
source .venv/bin/activate
echo "Using GPU:"
nvidia-smi || echo "No GPU detected"

# davinci chatGPT cohere dolly  
# wikipedia reddit wikihow peerread arxiv
# ==== Launch ====
echo "Launching training..."
python train2.py \
  --train-models davinci chatGPT cohere dolly \
  --test-models davinci \
  --train-domains wikipedia wikihow peerread arxiv \
  --test-domains reddit

echo "Training finished."
