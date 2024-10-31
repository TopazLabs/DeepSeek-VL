#! /bin/bash

# python train.py \
#     --model_path "deepseek-ai/deepseek-vl-1.3b-chat" \
#     --train_data_path "/datasets/auto_prompt/topaz_auto_prompt.jsonl" \
#     --output_dir "./deepseek_vl_finetuned" \
#     --num_train_epochs 3 \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 8 \
#     --learning_rate 5e-5 \
#     --fp16 \
#     --overwrite_output_dir


# python train_deepseek_vl.py \
#     --model_path "deepseek-ai/deepseek-vl-7b-chat" \
#     --train_data_path "./data/train_dataset.jsonl" \
#     --eval_data_path "./data/eval_dataset.jsonl" \
#     --output_dir "./deepseek_vl_finetuned" \
#     --num_train_epochs 3 \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 4 \
#     --learning_rate 5e-5 \
#     --fp16 \
#     --use_lora \
#     --lora_r 8 \
#     --lora_alpha 16 \
#     --lora_dropout 0.05 \
#     --lora_target_modules q_proj v_proj \
#     --do_eval \
#     --overwrite_output_dir

CUDA_VISIBLE_DEVICES=2 swift sft \
    --model_type deepseek-vl-1_3b-chat \
    --dataset /datasets/auto_prompt/topaz_auto_prompt.jsonl \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --num_train_epochs 25 \
    --output_dir ./deepseek_vl_finetuned