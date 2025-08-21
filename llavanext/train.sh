NPROC_PER_NODE=8
NNODES=1

export DISABLE_VERSION_CHECK=1
export OMP_NUM_THREADS=1
export NCCL_BLOCKING_WAIT=1  # 强制等待超时返回
export NCCL_ASYNC_ERROR_HANDLING=1  # 启用异步错误处理
export NCCL_P2P_DISABLED=1  # 禁用直连通信（如问题依然存在）
export NCCL_DEBUG=WARN  # 输出 NCCL 的核心调试日志

MODEL=/opt/models/llava-hf/llava-hf/llava-v1.6-vicuna-7b-hf
OUTPUT=/opt/models/llava-hf/llava-hf/llava-v1.6-vicuna-7b-hf-instruct

torchrun --nproc-per-node $NPROC_PER_NODE \
    --nnodes $NNODES \
    src/train.py \
    --stage sft \
    --do_train \
    --finetuning_type full \
    --deepspeed examples/deepspeed/ds_z2_config.json \
    --dataset tgdoc \
    --template llava_next \
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
    --save_steps 800 \
    --plot_loss \
    --num_train_epochs 1 \
    --preprocessing_num_workers 128 \
    --bf16  2>&1 | tee error.txt