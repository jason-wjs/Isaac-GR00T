# set -x -e
export WANDB_MODE=offline
export NUM_GPUS=4
export RANK=0 WORLD_SIZE=1 MASTER_ADDR=localhost MASTER_PORT=29500

# 1. 禁用 P2P，解决 4090 通信不稳定问题
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

uv run torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT \
    gr00t/experiment/launch_finetune.py \
    --base_model_path nvidia/GR00T-N1.6-3B \
    --dataset_path  demo_data/single_goods_orders \
    --embodiment_tag LEJU_KUAVO \
    --num_gpus $NUM_GPUS \
    --output_dir ckpt/kuavo_finetune \
    --save_steps 2000 \
    --save_total_limit 5 \
    --max_steps 20000 \
    --warmup_ratio 0.05 \
    --weight_decay 1e-5 \
    --learning_rate 1e-4 \
    --use_wandb \
    --global_batch_size 128 \
    --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --dataloader_num_workers 2
