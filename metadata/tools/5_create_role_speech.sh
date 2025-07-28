#!/bin/bash

TOTAL_PART=32
NUM_GPUS=8
SCRIPT=tools/5_create_role_speech.py
MODE=test  # 默认test，可传train

mkdir -p create_multiturn_data/logs

for ((i=1; i<=TOTAL_PART; i++)); do
    gpu_id=$(( (i-1) % NUM_GPUS ))
    echo "启动: part $i on GPU $gpu_id"
    CUDA_VISIBLE_DEVICES=$gpu_id nohup python $SCRIPT --total_part $TOTAL_PART --current_part $i --mode $MODE > create_multiturn_data/logs/log_part_${i}.txt 2>&1 &
done

wait
