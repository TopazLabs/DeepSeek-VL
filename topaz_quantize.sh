#!/bin/bash
CHECKPOINT_PATH="/home/topaz_koch/dev/imageunderstanding/SemanticAnalysis/DeepSeek-VL/deepseek_vl_finetuned/deepseek-vl-1_3b-chat/v18-20241031-002822/checkpoint-4250"
OUTPUT_PATH="/home/topaz_koch/dev/imageunderstanding/SemanticAnalysis/DeepSeek-VL/deepseek_vl_quantized"

swift export \
    --ckpt_dir $CHECKPOINT_PATH \
    --merge_lora true \
    --dataset /datasets/auto_prompt/topaz_auto_prompt.jsonl

# swift export \
#     --ckpt_dir $CHECKPOINT_PATH \
#     --merge_lora true --quant_bits 8 \
#     --dataset /datasets/auto_prompt/topaz_auto_prompt.jsonl \
#     --quant_method hqq