# set -x -e
export WANDB_API_KEY=7c5f305e84248e078c1e77d32d2592d31d2bc834
export WANDB_PROJECT=GR00T_leju
export NUM_GPUS=2
export CUDA_VISIBLE_DEVICES=0,1
export RANK=0 WORLD_SIZE=2 MASTER_ADDR=localhost MASTER_PORT=29500
# 设置 Hugging Face 缓存目录到大容量存储路径
export HF_HOME="/mnt/pfs/scalelab2/Humanoid/hf_cache"
# 增加 NCCL 超时时间到 1 小时 (默认 10 分钟)，防止因数据加载慢导致训练中断
export NCCL_TIMEOUT=3600
export TORCH_NCCL_BLOCKING_WAIT=1


uv run torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT \
    gr00t/experiment/launch_finetune.py \
    --base_model_path ckpt/kuavo_finetune/checkpoint-4000 \
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
