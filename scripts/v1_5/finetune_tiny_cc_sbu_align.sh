#!/bin/bash

# TODO:
# pip install -e ".[train]"
# download checkpoint model
# cd /TinyLLaVA
# python download_checkpoints.py
# cd /TinyLLaVA/playground
# python download_datasets.py
# To train on fewer GPUs, you can reduce the per_device_train_batch_size and increase the gradient_accumulation_steps accordingly. Always keep the global batch size the same: per_device_train_batch_size x gradient_accumulation_steps x num_gpus.

deepspeed llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 5e-3 \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --version v1 \
    --data_path ./playground/data/cc_sbu_align_llava/filter_cap_llava.json \
    --image_folder ./playground/data/cc_sbu_align_llava/images \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter ./checkpoints/TinyLLaVA-1.1B-pretrained-projector/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter False \
    --freeze_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/tiny-llava-align \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 2 \
    --learning_rate 5e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
