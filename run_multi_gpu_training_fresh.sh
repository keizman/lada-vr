#!/bin/bash

# ============================================================
# 🚀 从头开始训练 (Fresh Training)
# ============================================================
# 用途：忽略已有checkpoint，强制从头开始新训练
# 使用：bash run_multi_gpu_training_fresh.sh
# 
# 配置修改：
#   - CONFIG_FILE：选择配置文件
#   - CUDA_VISIBLE_DEVICES：选择GPU
#   - --nproc_per_node：GPU数量
# ============================================================

set -e

# 设置使用的GPU
export CUDA_VISIBLE_DEVICES=0,1

# 配置文件路径
CONFIG_FILE="configs/basicvsrpp/mosaic_restoration_generic_stage1.py"
WORK_DIR="./experiments/basicvsrpp/mosaic_restoration_generic_stage1"

echo "=========================================="
echo "  多GPU训练启动 (强制从头开始)"
echo "=========================================="
echo ""

# 检查是否存在旧的训练数据
if [ -d "$WORK_DIR" ]; then
    echo "⚠️  警告: 检测到已有训练目录"
    echo "   位置: $WORK_DIR"
    echo ""
    echo "   此脚本会忽略已有checkpoint从头开始训练"
    echo "   建议先备份重要的checkpoint文件"
    echo ""
    read -p "是否继续？[y/N] " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "操作已取消"
        exit 0
    fi
    echo ""
fi

# 显示配置
echo "📋 训练配置:"
echo "   - 配置文件: $CONFIG_FILE"
echo "   - GPU数量: 2 (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "   - 工作目录: $WORK_DIR"
echo "   - 模式: 从头开始（忽略checkpoint）"
echo ""

# 启动训练
echo "🚀 启动训练..."
echo ""

# 使用torch.distributed.launch启动多GPU训练
# 使用--load-from参数会阻止auto-resume
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port=29500 \
    scripts/training/train-mosaic-restoration-basicvsrpp.py \
    $CONFIG_FILE \
    --launcher pytorch \
    --load-from ""

echo ""
echo "=========================================="
echo "✅ 训练完成！"
echo "=========================================="

