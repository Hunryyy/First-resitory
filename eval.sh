#!/bin/bash

set -e
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 定义关键路径变量 (需与 python 脚本中的配置保持一致)
ADAPTER_PATH="output/mineru_style_finetune"
SUBMIT_FILE="submit.jsonl"

echo "========================================================="
echo "   Start Inference Pipeline: $(date)"
echo "========================================================="

# 1. 前置检查
if [ ! -d "$ADAPTER_PATH" ]; then
    echo "Error: Adapter path '$ADAPTER_PATH' not found!"
    echo "Please run train.sh first to generate model weights."
    exit 1
fi

# 2. 清理旧的提交文件
if [ -f "$SUBMIT_FILE" ]; then
    echo "Removing old submission file..."
    rm "$SUBMIT_FILE"
fi

# 3. 执行推理
echo "[Step 3] Running Inference (eval.py)..."
python eval.py

# 4. 结果验证
if [ -f "$SUBMIT_FILE" ]; then
    COUNT=$(wc -l < "$SUBMIT_FILE")
    echo "Success! Generated $SUBMIT_FILE with $COUNT lines."
else
    echo "Error: Failed to generate $SUBMIT_FILE."
    exit 1
fi
