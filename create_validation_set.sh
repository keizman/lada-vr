#!/bin/bash

# 验证集划分快速执行脚本
# 自动从训练集中划分出验证集

set -e

echo "=========================================="
echo "  验证集自动划分工具"
echo "=========================================="
echo ""

# 配置参数
SRC_ROOT="/root/autodl-tmp/train_with_mosaic"
VAL_RATIO=0.1  # 10%作为验证集
VAL_SIZE=""    # 留空使用比例，或设置固定数量如 200

# 检查源目录是否存在
if [ ! -d "$SRC_ROOT" ]; then
    echo "❌ 错误: 训练集目录不存在: $SRC_ROOT"
    echo "   请修改脚本中的 SRC_ROOT 变量"
    exit 1
fi

# 统计训练集样本数
TOTAL_SAMPLES=$(find "$SRC_ROOT/crop_unscaled_meta" -name "*.json" 2>/dev/null | wc -l)
echo "📊 当前训练集样本数: $TOTAL_SAMPLES"

if [ $TOTAL_SAMPLES -eq 0 ]; then
    echo "❌ 错误: 在 $SRC_ROOT/crop_unscaled_meta 中没有找到JSON文件"
    exit 1
fi

echo ""
echo "📋 划分配置:"
if [ -n "$VAL_SIZE" ]; then
    echo "   - 验证集大小: $VAL_SIZE 个样本（固定数量）"
else
    VAL_COUNT=$(python3 -c "print(int($TOTAL_SAMPLES * $VAL_RATIO))")
    echo "   - 验证集比例: ${VAL_RATIO} (${VAL_RATIO}0% = $VAL_COUNT 个样本)"
fi
echo "   - 源训练集: $SRC_ROOT"
echo "   - 目标验证集: ${SRC_ROOT}_val"
echo ""

# 询问用户确认
read -p "是否继续？[y/N] " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "操作已取消"
    exit 0
fi

echo ""
echo "🚀 开始划分..."
echo ""

# 构建命令
CMD="python3 scripts/dataset_creation/split-train-val-dataset.py --src-root $SRC_ROOT"

if [ -n "$VAL_SIZE" ]; then
    CMD="$CMD --val-size $VAL_SIZE"
else
    CMD="$CMD --val-ratio $VAL_RATIO"
fi

# 执行划分
cd /root/lada
$CMD

# 检查结果
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 验证集划分完成！"
    echo ""
    echo "📁 验证集位置: ${SRC_ROOT}_val"
    echo ""
    echo "🔧 配置文件已更新:"
    echo "   configs/basicvsrpp/mosaic_restoration_generic_stage1.py"
    echo ""
    echo "🎯 下一步: 重启训练以使用新的验证集"
    echo "   ./run_multi_gpu_training.sh"
else
    echo ""
    echo "❌ 划分过程出现错误"
    exit 1
fi

