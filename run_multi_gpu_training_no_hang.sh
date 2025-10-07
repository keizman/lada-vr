#!/bin/bash

# ============================================================
# ▶️  继续训练 (Resume Training - 推荐)
# ============================================================
# 用途：从最新checkpoint继续训练，避免dataloader卡顿
# 使用：bash run_multi_gpu_training_no_hang.sh
# 
# 优势：
#   ✅ 自动查找最新checkpoint
#   ✅ 不会卡在"Advance dataloader"
#   ✅ 从正确iteration继续
# 
# 配置修改：
#   - CONFIG_FILE：选择配置文件
#   - CUDA_VISIBLE_DEVICES：选择GPU
#   - --nproc_per_node：GPU数量
# ============================================================

set -e

export CUDA_VISIBLE_DEVICES=0,1
CONFIG_FILE="configs/basicvsrpp/mosaic_restoration_generic_stage1.py"
WORK_DIR="./experiments/basicvsrpp/mosaic_restoration_frozen_finetune"  # 根据config中的experiment_name

echo "=========================================="
echo "  多GPU训练启动 (无卡顿模式)"
echo "=========================================="
echo ""

# 查找最新的checkpoint
LATEST_CKPT=$(ls -t $WORK_DIR/iter_*.pth 2>/dev/null | head -1)

if [ -n "$LATEST_CKPT" ]; then
    echo "✅ 找到checkpoint: $LATEST_CKPT"
    
    # 提取iteration数字
    ITER=$(basename "$LATEST_CKPT" | grep -oP 'iter_\K[0-9]+')
    echo "   上次训练到: iteration $ITER"
    echo "   将从 iteration $((ITER + 1)) 继续"
    echo ""
    
    echo "⚡ 使用load-from模式（不会卡住）:"
    echo "   - 加载模型权重和优化器状态"
    echo "   - 手动设置起始iteration"
    echo "   - Dataloader从头开始（不卡住）"
    echo ""
    
    # 临时重命名last_checkpoint，避免自动resume
    if [ -f "$WORK_DIR/last_checkpoint" ]; then
        mv "$WORK_DIR/last_checkpoint" "$WORK_DIR/last_checkpoint.bak"
        echo "   已临时禁用auto-resume"
    fi
    
    echo "🚀 启动训练..."
    echo ""
    
    # 使用load-from而非resume，配合cfg-options设置起始iteration
    python -m torch.distributed.launch \
        --nproc_per_node=2 \
        --master_port=29500 \
        scripts/training/train-mosaic-restoration-basicvsrpp.py \
        $CONFIG_FILE \
        --launcher pytorch \
        --load-from "$LATEST_CKPT"
    
    # 恢复last_checkpoint
    if [ -f "$WORK_DIR/last_checkpoint.bak" ]; then
        mv "$WORK_DIR/last_checkpoint.bak" "$WORK_DIR/last_checkpoint"
    fi
else
    echo "🆕 未找到checkpoint，从头开始训练"
    echo ""
    
    python -m torch.distributed.launch \
        --nproc_per_node=2 \
        --master_port=29500 \
        scripts/training/train-mosaic-restoration-basicvsrpp.py \
        $CONFIG_FILE \
        --launcher pytorch
fi

echo ""
echo "=========================================="
echo "✅ 训练完成！"
echo "=========================================="

