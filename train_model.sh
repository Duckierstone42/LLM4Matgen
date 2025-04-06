#!/bin/bash

#SBATCH --job-name=train_model_custom
#SBATCH --ntasks=1
#SBATCH --time=16:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=128GB


module load anaconda3

cd /home/hice1/athalanki3/scratch/LLM4StructGen

conda run -n llm4structgen tune run lora_finetune_single_device --config configs/train/llama3_1/8B_lora_single_device.yaml