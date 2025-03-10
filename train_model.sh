#!/bin/bash

#SBATCH --job-name=train_model
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:h200:1


module load anaconda3

cd /home/hice1/athalanki3/scratch/LLM4StructGen

conda run -n llm4structgen tune run lora_finetune_single_device --config configs/train/llama3_1/8B_lora_single_device.yaml