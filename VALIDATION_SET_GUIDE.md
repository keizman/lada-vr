# 验证集划分指南

## 📌 问题背景

在训练马赛克修复模型时，如果没有独立的验证集会导致：
- ❌ 训练过程中看不到 PSNR / SSIM 等评估指标
- ❌ 只能盲目观察 loss，无法判断模型真实性能
- ❌ 模型可能过拟合，上线后才发现问题

## ✅ 解决方案

我们提供了自动化脚本，可以从训练集中划分出验证集。

### 当前数据集状态

- **训练集**: `/root/autodl-tmp/train_with_mosaic` (2019个样本)
- **验证集**: 尚未创建

## 🚀 快速开始

### 方法1：使用一键脚本（推荐）

```bash
cd /root/lada
./create_validation_set.sh
```

这个脚本会：
1. 自动检测训练集样本数
2. 划分10%作为验证集（约200个样本）
3. 从训练集中移除这些样本
4. 配置文件已自动更新

### 方法2：手动执行（更灵活）

#### 基础用法

```bash
# 按比例划分（默认10%）
python scripts/dataset_creation/split-train-val-dataset.py \
    --src-root /root/autodl-tmp/train_with_mosaic

# 按固定数量划分（推荐200-500个样本）
python scripts/dataset_creation/split-train-val-dataset.py \
    --src-root /root/autodl-tmp/train_with_mosaic \
    --val-size 200
```

#### 高级选项

```bash
# 预览模式（不实际执行）
python scripts/dataset_creation/split-train-val-dataset.py \
    --src-root /root/autodl-tmp/train_with_mosaic \
    --val-size 200 \
    --dry-run

# 仅复制不删除（保留训练集中的验证样本）
python scripts/dataset_creation/split-train-val-dataset.py \
    --src-root /root/autodl-tmp/train_with_mosaic \
    --val-size 200 \
    --no-remove

# 自定义验证集路径
python scripts/dataset_creation/split-train-val-dataset.py \
    --src-root /root/autodl-tmp/train_with_mosaic \
    --dst-root /root/autodl-tmp/my_custom_val_set \
    --val-size 200

# 设置随机种子确保可复现
python scripts/dataset_creation/split-train-val-dataset.py \
    --src-root /root/autodl-tmp/train_with_mosaic \
    --seed 42
```

## 📊 推荐配置

| 训练集大小 | 验证集建议 | 说明 |
|-----------|----------|------|
| < 1000 | 10-15% | 确保有足够的验证样本 |
| 1000-5000 | 5-10% | 平衡训练和验证 |
| > 5000 | 3-5% 或固定500个 | 避免浪费过多数据 |

**当前推荐**（2019个样本）：
- 比例方式：10% ≈ 200个样本 ✅
- 固定方式：200-300个样本 ✅

## 📁 目录结构

划分后的目录结构：

```
/root/autodl-tmp/
├── train_with_mosaic/           # 训练集 (~1819个样本)
│   ├── crop_unscaled_img/       # NSFW视频片段
│   ├── crop_unscaled_mask/      # 分割mask
│   └── crop_unscaled_meta/      # 元数据JSON
│
└── train_with_mosaic_val/       # 验证集 (~200个样本)
    ├── crop_unscaled_img/
    ├── crop_unscaled_mask/
    └── crop_unscaled_meta/
```

## 🔧 配置文件说明

配置文件 `configs/basicvsrpp/mosaic_restoration_generic_stage1.py` 已自动更新：

```python
# 数据路径配置
data_root = '/root/autodl-tmp/train_with_mosaic'
val_root = '/root/autodl-tmp/train_with_mosaic_val'  # 验证集路径

# 训练集配置
train_dataloader = dict(
    dataset=dict(
        metadata_root_dir=data_root + "/crop_unscaled_meta",
        ...
    )
)

# 验证集配置
val_dataloader = dict(
    dataset=dict(
        metadata_root_dir=val_root + "/crop_unscaled_meta",  # 使用独立验证集
        ...
    )
)
```

## 📈 训练时的验证指标

划分验证集后，训练时每3000个iteration会自动进行验证并输出：

- **PSNR** (Peak Signal-to-Noise Ratio): 峰值信噪比，越高越好
  - 典型范围：25-35 dB
  - > 30 dB 通常认为是好的结果
  
- **SSIM** (Structural Similarity Index): 结构相似性指数，越高越好
  - 范围：0-1
  - > 0.9 通常认为是好的结果

示例训练日志：
```
10/07 15:00:00 - mmengine - INFO - Iter [3000/30000] ... loss: 0.0123
10/07 15:05:00 - mmengine - INFO - Validation Results:
10/07 15:05:00 - mmengine - INFO -   PSNR: 28.5
10/07 15:05:00 - mmengine - INFO -   SSIM: 0.87
```

## 🎯 工作流程

### 完整流程

```bash
# 1. 停止当前训练（如果正在运行）
pkill -f train-mosaic-restoration-basicvsrpp

# 2. 划分验证集
cd /root/lada
./create_validation_set.sh

# 3. 验证配置（可选）
cat configs/basicvsrpp/mosaic_restoration_generic_stage1.py | grep -A 2 "val_root"

# 4. 重新启动训练
./run_multi_gpu_training.sh

# 5. 监控训练日志
tail -f experiments/basicvsrpp/mosaic_restoration_generic_stage1/*/vis_data/scalars.json
```

### 注意事项

⚠️ **重要提示**：
1. 验证集样本会从训练集中移除（使用 `--no-remove` 可保留）
2. 划分操作不可逆，建议先用 `--dry-run` 预览
3. 如果训练已经进行了一段时间，重新划分验证集会影响loss曲线的连续性
4. 验证集应该尽量覆盖不同场景、不同质量的样本

## 🔍 验证结果分析

### 如何判断模型训练效果

1. **正常训练**:
   - Training loss 稳定下降
   - Validation PSNR/SSIM 稳定上升
   - 两者趋势基本一致

2. **过拟合迹象**:
   - Training loss 继续下降
   - Validation PSNR/SSIM 不再上升或开始下降
   - 应考虑提前停止训练

3. **欠拟合**:
   - Training loss 和 validation metrics 都还在明显改善
   - 可以继续训练或增加模型复杂度

### 早停策略

建议在以下情况提前停止训练：
- 验证集 PSNR 连续3次验证不再上升
- 验证集 SSIM 开始下降
- 达到满意的指标值

## 📝 脚本参数完整列表

```
split-train-val-dataset.py 参数说明:

必需参数:
  --src-root PATH          源训练集根目录

可选参数:
  --dst-root PATH          目标验证集根目录 (默认: {src-root}_val)
  --val-ratio FLOAT        验证集比例 (默认: 0.1, 即10%)
  --val-size INT           固定验证集大小 (覆盖 --val-ratio)
  --seed INT               随机种子 (默认: 42)
  --no-remove              仅复制不删除训练集样本
  --dry-run                预览模式，不实际执行
  -h, --help               显示帮助信息
```

## 🐛 常见问题

### Q1: 验证集应该划分多少？
**A**: 对于2019个样本，推荐200-300个（约10-15%）。足够评估模型性能，又不会过度减少训练数据。

### Q2: 已经开始训练了，还能划分验证集吗？
**A**: 可以，但需要重启训练。如果想继续训练，需要用 `--resume` 参数，但验证集的变化会影响指标的连续性。

### Q3: 验证集样本是随机选择的吗？
**A**: 是的，使用随机抽样。可以通过 `--seed` 参数设置随机种子确保可复现。

### Q4: 能否使用不同的验证集路径？
**A**: 可以，使用 `--dst-root` 参数指定。记得同步修改配置文件中的 `val_root`。

### Q5: 如果想保留训练集中的验证样本怎么办？
**A**: 使用 `--no-remove` 参数。但这样会导致验证集和训练集有重叠，验证指标会偏高。

## 📖 相关文档

- [训练参数对比](TRAINING_PARAMS_COMPARISON.md) - 微调参数优化
- [更新日志](CHANGELOG.md) - 所有修改记录
- [训练文档](docs/training_and_dataset_creation.md) - 完整训练指南

## 💡 最佳实践

1. **验证集选择**: 尽量选择多样化的样本，覆盖不同场景
2. **验证频率**: 当前配置每3000 iter验证一次，共10次验证
3. **指标监控**: 重点关注PSNR和SSIM，配合可视化结果
4. **早停时机**: PSNR不再上升时考虑停止，避免过拟合
5. **数据备份**: 划分前建议备份原始数据集

---

创建时间: 2025-10-07
脚本位置: `scripts/dataset_creation/split-train-val-dataset.py`
配置文件: `configs/basicvsrpp/mosaic_restoration_generic_stage1.py`

