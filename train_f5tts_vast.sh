#!/bin/bash

# Navigate to the correct directory
cd /root/F5-TTS-Fork/src

# Activate virtual environment
source /root/F5-TTS-Fork/.venv/bin/activate

while true; do
   # Run the training script
   accelerate launch finetune_russian_singing.py \
       --exp_name F5TTS_Base \
       --learning_rate 0.00005 \
       --batch_size_per_gpu 4 \
       --batch_size_type sample \
       --max_samples 64 \
       --grad_accumulation_steps 2 \
       --max_grad_norm 1 \
       --epochs 10000 \
       --num_warmup_updates 500 \
       --save_per_updates 500 \
       --samples_per_updates 500 \
       --last_per_steps 1000 \
       --dataset_name russian_dataset_ft_section \
       --tokenizer pinyin \
       --log_samples True \
       --logger tensorboard \
       --finetune True \
       --tokenizer_path ../data/Emilia_ZH_EN_pinyin/vocab.txt \
       --pretrain /root/F5-TTS-Fork/ckpts/russian_dataset_ft_section/model_last.pt

   echo "Training crashed or completed, restarting in 5 seconds..."
   sleep 5
done