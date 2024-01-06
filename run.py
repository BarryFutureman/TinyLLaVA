import subprocess

command = [
    "deepspeed", "llava/train/train.py",
    "--deepspeed", "./scripts/zero2.json",
    "--model_name_or_path", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # <= Tiny LLaMA
    "--version", "plain",
    "--data_path", "./playground/data/LLaVA-Pretrain/chat.json",
    "--image_folder", "./playground/data/LLaVA-Pretrain/images",
    "--image_aspect_ratio", "pad",
    "--vision_tower", "openai/clip-vit-large-patch14-336",
    "--mm_projector_type", "mlp2x_gelu",
    "--tune_mm_mlp_adapter", "True",
    "--mm_vision_select_layer", "-2",
    "--mm_use_im_start_end", "False",
    "--mm_use_im_patch_token", "False",
    "--bf16", "True",
    "--output_dir", "./checkpoints/tiny-llava-pretrain",
    "--num_train_epochs", "1",
    "--per_device_train_batch_size", "32",
    "--per_device_eval_batch_size", "4",
    "--gradient_accumulation_steps", "1",
    "--evaluation_strategy", "no",
    "--save_strategy", "steps",
    "--save_steps", "24000",
    "--save_total_limit", "1",
    "--learning_rate", "5e-3",
    "--weight_decay", "0.",
    "--warmup_ratio", "0.03",
    "--lr_scheduler_type", "cosine",
    "--logging_steps", "1",
    "--tf32", "True",
    "--model_max_length", "2048",
    "--gradient_checkpointing", "True",
    "--dataloader_num_workers", "4",
    "--lazy_preprocess", "True",
    "--report_to", "wandb"
]

subprocess.run(command)
