#!/bin/bash

# 1. 安全设置：任何命令失败立即退出
set -e

# 2. 环境变量设置
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 3. 准备工作
mkdir -p logs
mkdir -p output

echo "========================================================="
echo "   Start Training Pipeline: $(date)"
echo "========================================================="

# 4. 数据预处理 (Step 1)
if [ ! -f "data/train_formatted.jsonl" ]; then
    echo "[Step 1] Data preprocessing..."
    python trainer/dataset/preprocess.py
else
    echo "[Step 1] Found existing data, skipping preprocessing."
fi

# 5. 启动训练 (Step 2)
echo "[Step 2] Starting LoRA Finetuning..."
# 使用 tee 命令：既能在屏幕看进度，又能存入 logs/train.log
python train.py 2>&1 | tee logs/train.log

# 6. 模型合并 (Step 3) - 关键修复：必须在训练完成后执行
echo "[Step 3] Merge LoRA Weights (Creating output/mineru_merged_model)..."
python trainer/merge/merge_lora.py

echo "========================================================="
echo "   Training and Merging Finished Successfully!"
echo "   Check logs/train.log for details."
echo "========================================================="
