#!/bin/bash

export WANDB_MODE=offline
export NUM_TRAIN_EPOCHS=1.1
export WANDB_PROJECT=llama-8b-8k
export MODEL_PATH=./init_model/Llama-3-mini
export OUTPUT_DIR=./Llama-3-mini-8k
export SAVE_STEPS=1000
export LOGGING_STEPS=100
export DATASET_PATH=./processed_data

deepspeed ./scripts/run_clm.py \
--deepspeed ./configs/ds_config_zero3.json \
--model_name_or_path $MODEL_PATH \
--torch_dtype bfloat16 \
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 1 \
--output_dir $OUTPUT_DIR \
--bf16 \
--do_train \
--num_train_epochs $NUM_TRAIN_EPOCHS \
--dataset_path $DATASET_PATH \
--logging_steps $LOGGING_STEPS \
--learning_rate 2e-5 \
--adam_beta1 0.9 \
--adam_beta2 0.99 \
--dataloader_num_workers 64 \
--save_steps $SAVE_STEPS \
--save_total_limit 5 \
--gradient_checkpointing true \
--include_num_input_tokens_seen true
