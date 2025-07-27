NPROC_PER_NODE=8
NNODES=1

export DISABLE_VERSION_CHECK=1
MODEL=/opt/LLaMA-Factory-main/ckpt/Qwen2.5-VL-3B-Instruct
OUTPUT=/opt/LLaMA-Factory-main/ckpt/Qwen2.5-VL-3B-Instruct-tune

torchrun --nproc-per-node $NPROC_PER_NODE \
    --nnodes $NNODES \
    src/train.py \
    --stage sft \
    --do_train \
    --finetuning_type full \
    --deepspeed examples/deepspeed/ds_z2_config.json \
    --dataset mllm_demo \
    --template qwen2_vl \
    --model_name_or_path $MODEL \
    --output_dir $OUTPUT \
    --overwrite_cache \
    --overwrite_output_dir \
    --warmup_steps 100 \
    --weight_decay 0.1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --ddp_timeout 9000 \
    --learning_rate 5e-6 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --cutoff_len 4096 \
    --save_steps 1000 \
    --plot_loss \
    --num_train_epochs 3 \
    --bf16 

