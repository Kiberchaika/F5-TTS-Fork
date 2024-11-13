#!/bin/bash

# Set CUDA device
export CUDA_VISIBLE_DEVICES=1

# Navigate to the correct directory
cd /home/k4/Python/F5-TTS-Fork/src

# Activate virtual environment
source /home/k4/Python/F5-TTS-Fork/.venv/bin/activate

# Remove old checkpoints
rm -rf /home/k4/Python/F5-TTS-Fork/ckpts/russian_dataset_ft

# Run the training script
/home/k4/Python/F5-TTS-Fork/.venv/bin/python finetune_russian_singing.py \
    --exp_name F5TTS_Base \
    --learning_rate 0.00005 \
    --batch_size_per_gpu 2 \
    --batch_size_type sample \
    --max_samples 64 \
    --grad_accumulation_steps 2 \
    --max_grad_norm 1 \
    --epochs 10000 \
    --num_warmup_updates 500 \
    --save_per_updates 500 \
    --samples_per_updates 500 \
    --last_per_steps 2000 \
    --dataset_name russian_dataset_ft \
    --tokenizer pinyin \
    --log_samples True \
    --logger tensorboard \
    --finetune True \
    --tokenizer_path ../data/Emilia_ZH_EN_pinyin/vocab.txt \
    --pretrain /home/k4/Python/F5-TTS-Fork/ckpts/russian_dataset_ft_translit_pinyin/model_last.pt  
 