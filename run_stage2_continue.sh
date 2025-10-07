#!/bin/bash

# ============================================================
# ▶️  Stage2 继续训练脚本
# ============================================================
# 用途：从最新的 Stage2 checkpoint 继续训练
# 使用：bash run_stage2_continue.sh
# ============================================================

set -e

export CUDA_VISIBLE_DEVICES=0,1
CONFIG_FILE="configs/basicvsrpp/mosaic_restoration_generic_stage2.py"
WORK_DIR="./experiments/basicvsrpp/mosaic_restoration_generic_stage2"

echo "=========================================="
echo "  Stage2 继续训练"
echo "=========================================="
echo ""

# 查找最新的checkpoint
LATEST_CKPT=$(ls -t $WORK_DIR/iter_*.pth 2>/dev/null | head -1)

if [ -n "$LATEST_CKPT" ]; then
    echo "✅ 找到checkpoint: $LATEST_CKPT"
    
    # 提取iteration数字
    ITER=$(basename "$LATEST_CKPT" | grep -oP 'iter_\K[0-9]+')
    echo "   上次训练到: iteration $ITER"
    echo "   将继续训练"
    echo ""
    
    # 临时禁用auto-resume
    if [ -f "$WORK_DIR/last_checkpoint" ]; then
        mv "$WORK_DIR/last_checkpoint" "$WORK_DIR/last_checkpoint.bak"
    fi
    
    echo "🚀 启动训练..."
    echo ""
    
    python -m torch.distributed.launch \
        --nproc_per_node=2 \
        --master_port=29501 \
        scripts/training/train-mosaic-restoration-basicvsrpp.py \
        $CONFIG_FILE \
        --launcher pytorch \
        --load-from "$LATEST_CKPT"
    
    # 恢复last_checkpoint
    if [ -f "$WORK_DIR/last_checkpoint.bak" ]; then
        mv "$WORK_DIR/last_checkpoint.bak" "$WORK_DIR/last_checkpoint"
    fi
else
    echo "🆕 未找到 Stage2 checkpoint"
    echo "   请先运行: bash run_stage2_from_stage1.sh"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ 训练完成！"
echo "=========================================="

