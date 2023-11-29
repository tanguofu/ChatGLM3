#! /usr/bin/env bash

fmt_info(){
  printf '%s info: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" 
}

set -e

# https://github.com/tanguofu/ChatGLM3/tree/main/finetune_chatmodel_demo

# 1. support multi-turn

NUM_GPUS=$(nvidia-smi -L |wc -l)

LR=1e-4

MAX_SEQ_LEN=2048
DEV_BATCH_SIZE=4
GRAD_ACCUMULARION_STEPS=1
MAX_STEP=200
SAVE_INTERVAL=100

DATESTR=`date +%Y%m%d-%H%M%S`
RUN_NAME=tool_alpaca_ft

# in cos
DATASET_PATH=/model/chatglm3/ChatGLM3/finetune_demo/formatted_data/tool_alpaca.jsonl
BASE_MODEL_PATH=/model/chatglm3/chatglm3-6b

OUTPUT_DIR=/data/output/${RUN_NAME}-${DATESTR}-${LR}

mkdir -p $OUTPUT_DIR

fmt_info "Start finetune.py at pwd:$(pwd)"

pip3 install -r requirements.txt

set -x
torchrun --nnodes=$WORLD_SIZE  --nproc_per_node=$NUM_GPUS --max-restarts=1  --rdzv-id=$MASTER_ADDR --rdzv-backend=c10d --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
    finetune.py \
    --train_format multi-turn \
    --train_file $DATASET_PATH \
    --max_seq_length $MAX_SEQ_LEN \
    --preprocessing_num_workers 1 \
    --model_name_or_path $BASE_MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --per_device_train_batch_size $DEV_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUMULARION_STEPS \
    --max_steps $MAX_STEP \
    --logging_steps 1 \
    --save_steps $SAVE_INTERVAL \
    --fp16 \
    --deepspeed configs/deepspeed-stage2.json 2>&1 | tee ${OUTPUT_DIR}/train.log