#!/bin/bash

# ============================================================
# 🎯 Stage2 训练脚本 - 从 Stage1 checkpoint 开始
# ============================================================
# 用途：加载 Stage1 训练好的权重，开始 Stage2 GAN 训练
# 使用：bash run_stage2_from_stage1.sh
# 
# Stage1 → Stage2 转换：
#   需要先运行 convert-weights-basicvsrpp-stage1-to-stage2.py
#   将 Stage1 的权重转换为 Stage2 格式（添加 discriminator 等）
# ============================================================

set -e

export CUDA_VISIBLE_DEVICES=0,1

# 配置文件和checkpoint路径
CONFIG_FILE="configs/basicvsrpp/mosaic_restoration_generic_stage2.py"
STAGE1_CKPT="experiments/basicvsrpp/mosaic_restoration_frozen_finetune/iter_converted.pth"

echo "=========================================="
echo "  Stage2 GAN 训练启动"
echo "=========================================="
echo ""

# 检查转换后的checkpoint是否存在
if [ ! -f "$STAGE1_CKPT" ]; then
    echo "❌ 错误：未找到转换后的 Stage1 checkpoint"
    echo "   路径: $STAGE1_CKPT"
    echo ""
    echo "请先运行转换脚本："
    echo "   python scripts/training/convert-weights-basicvsrpp-stage1-to-stage2.py \\"
    echo "       experiments/basicvsrpp/mosaic_restoration_frozen_finetune/iter_3000.pth \\"
    echo "       experiments/basicvsrpp/mosaic_restoration_frozen_finetune/iter_converted.pth"
    exit 1
fi

echo "✅ 找到 Stage1 checkpoint: $STAGE1_CKPT"
echo ""

# 显示配置
echo "📋 训练配置:"
echo "   - 配置文件: $CONFIG_FILE"
echo "   - GPU数量: 2 (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "   - 加载权重: $STAGE1_CKPT"
echo "   - 训练模式: Stage2 GAN (Generator + Discriminator + Perceptual Loss)"
echo ""

# 启动训练
echo "🚀 启动 Stage2 训练..."
echo ""

python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port=29501 \
    scripts/training/train-mosaic-restoration-basicvsrpp.py \
    $CONFIG_FILE \
    --launcher pytorch \
    --load-from "$STAGE1_CKPT"

echo ""
echo "=========================================="
echo "✅ Stage2 训练完成！"
echo "=========================================="

