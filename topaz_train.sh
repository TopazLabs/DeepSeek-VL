#! /bin/bash

CUDA_VISIBLE_DEVICES=2 swift sft \
    --model_type deepseek-vl-1_3b-chat \
    --dataset /datasets/auto_prompt/topaz_auto_prompt.jsonl \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --num_train_epochs 25 \
    --output_dir ./deepseek_vl_finetuned