NPROC_PER_NODE=8
NNODES=1

MODEL=Qwen/Qwen2-VL-7B-Instruct
OUTPUT=Qwen2-VL-7B-FULL-1452-cot-v1

torchrun --nproc-per-node $NPROC_PER_NODE \
    --nnodes $NNODES \
    src/train.py \
    --stage sft \
    --do_train \
    --finetuning_type full \
    --deepspeed /llm-cfs-nj/person/harryyhwang/LLaMA-Factory/examples/deepspeed/ds_z3_config.json \
    --dataset qwen2vl_1452_cot_v1 \
    --template qwen2_vl \
    --model_name_or_path /llm-cfs-nj/person/harryyhwang/Qwen2-VL/ckpt/$MODEL \
    --output_dir /llm-cfs-nj/person/harryyhwang/LLaMA-Factory/ckpt/$OUTPUT \
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
    --bf16  2>&1 | tee error.txt
