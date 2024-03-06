#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
DIR=`pwd`

# Guide:
# This script supports distributed training on multi-gpu workers (as well as single-worker training).
# Please set the options below according to the comments.
# For multi-gpu workers training, these options should be manually set for each worker.
# After setting the options, please run the script on each worker.

# Number of GPUs per GPU worker
GPUS_PER_NODE=8

# Number of GPU workers, for single-worker training, please set to 1
NNODES=1

nj_wfs_home=/mnt/wfs/mmnanjingwfssh/project_security-others-common/doodleliang
ceph_home=/mnt/cephfs
MODEL="${ceph_home}/LLM_MODEL_HUB/qwen/Qwen-72B-Chat" # Set the path if you do not want to load from huggingface directly
# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
# See the section for finetuning in README for more information.
DATA="${ceph_home}/doodleliang/Qwen/data/zjp_train.json"
DS_CONFIG_PATH="${ceph_home}/doodleliang/Qwen/finetune/ds_config_zero3.json"

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

torchrun $DISTRIBUTED_ARGS finetune.py \
    --model_name_or_path $MODEL \
    --data_path $DATA \
    --fp16 True \
    --output_dir ${ceph_home}/doodleliang/model/zjp_qwen_72b \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 2 \
    --learning_rate 3e-4 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "none" \
    --model_max_length 3000 \
    --lazy_preprocess True \
    --use_lora \
    --gradient_checkpointing \
    --deepspeed ${DS_CONFIG_PATH}
