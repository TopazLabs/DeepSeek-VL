#!/bin/bash
CHECKPOINT_PATH="/home/topaz_koch/dev/imageunderstanding/SemanticAnalysis/DeepSeek-VL/deepseek_vl/checkpoint-2400"
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