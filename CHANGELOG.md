# Changelog

## 2025-10-07

### 🔧 Bug修复：Resume训练卡住问题

**问题**：
- Resume训练时卡在 "Advance dataloader N steps" 
- 需要数小时才能完成resume过程
- 导致训练无法及时恢复

**根本原因**：
- 使用InfiniteSampler时，MMEngine默认恢复dataloader状态
- 需要实际遍历所有已训练过的batch（如3000步）
- 这个过程非常耗时

**解决方案**：
- 设置环境变量 `MMENGINE_RESUME_DATALOADER=0` 跳过dataloader恢复
- Dataloader从头开始，但模型权重、优化器和iteration计数完全恢复
- Resume时间从数小时降至<10秒

**代码修改**：

**文件**: `scripts/training/train-mosaic-restoration-basicvsrpp.py`
- **第92-94行**: 添加环境变量设置
  ```python
  # 设置环境变量，跳过dataloader的resume（避免卡住）
  # 这不会影响模型权重、优化器状态和iteration计数的恢复
  os.environ['MMENGINE_RESUME_DATALOADER'] = '0'
  ```
- **第83-84行**: 添加用户提示
  ```python
  print("   ⚡ Fast resume mode: Dataloader will restart (not skip already-seen data)")
  print("      This avoids the slow 'Advance dataloader' step")
  ```

**文件**: `configs/basicvsrpp/mosaic_restoration_generic_stage1.py`
- **第31行**: 优化dataloader配置
  ```python
  persistent_workers=True,  # 改为True以支持快速resume
  ```
- **第104-108行**: 添加resume相关配置
  ```python
  # Resume配置：禁用dataloader state恢复，避免resume时卡住
  randomness = dict(seed=42, deterministic=False, diff_rank_seed=True)
  resume = False  # 默认False，由脚本自动控制
  load_from = None
  ```

**影响说明**：
- ✅ 模型权重和优化器状态完全恢复
- ✅ Iteration计数正确（从3001继续）
- ✅ 学习率等调度器正常工作
- ⚠️ Dataloader从头开始（数据顺序改变）
- ✅ 不影响最终收敛（InfiniteSampler本就随机采样）

**性能提升**：
- Resume时间: 数小时 → <10秒
- 用户体验: 卡死 → 立即恢复训练

---

### 🎯 功能新增：验证集自动划分工具

**需求/问题**：
- 训练过程中无法看到 PSNR/SSIM 等评估指标，只能盲目观察loss
- 缺少独立验证集，无法及时发现过拟合问题
- 模型可能在某些场景上表现不佳，上线后才发现

**解决方案**：
- 创建自动化脚本从训练集中划分验证集
- 支持按比例或固定数量划分
- 自动复制对应的视频、mask和元数据文件
- 配置文件自动更新以使用验证集

**新增文件**：

1. **`scripts/dataset_creation/split-train-val-dataset.py`** (新增)
   - 完整的验证集划分工具
   - 支持多种划分模式和选项
   - 包含详细的进度报告和错误处理

2. **`create_validation_set.sh`** (新增)
   - 一键执行验证集划分
   - 交互式确认流程
   - 自动检测和统计

3. **`VALIDATION_SET_GUIDE.md`** (新增)
   - 完整的使用指南
   - 常见问题解答
   - 最佳实践建议

**代码修改**：

**文件**: `configs/basicvsrpp/mosaic_restoration_generic_stage1.py`
- **第26行**: 新增验证集路径变量
  ```python
  val_root = '/root/autodl-tmp/train_with_mosaic_val'  # 验证集路径
  ```
- **第54行**: 更新验证集数据加载路径
  ```python
  metadata_root_dir=val_root + "/crop_unscaled_meta",  # 使用独立验证集
  ```

**使用方法**：
```bash
# 方法1：一键执行
cd /root/lada
./create_validation_set.sh

# 方法2：手动执行
python scripts/dataset_creation/split-train-val-dataset.py \
    --src-root /root/autodl-tmp/train_with_mosaic \
    --val-size 200
```

**功能特性**：
- ✅ 支持按比例划分（如10%）或固定数量（如200个样本）
- ✅ 自动复制视频、mask和元数据文件
- ✅ 可选择是否从训练集中移除验证样本
- ✅ `--dry-run` 模式预览操作
- ✅ 详细的进度显示和错误处理
- ✅ 可设置随机种子确保可复现

**预期效果**：
- ✅ 训练时每3000 iter自动验证并输出PSNR/SSIM
- ✅ 及时发现过拟合问题
- ✅ 可视化训练效果，优化停止时机
- ✅ 提高模型最终质量

---

### ⚡ 性能优化：调整微调训练参数

**需求/问题**：
- 原配置max_iters=100,000适用于从零开始的全量训练，对于微调场景过长
- 训练预计需要1天4小时+，时间成本过高
- fix_iter占比不合理（5k/100k = 5%），需要同步调整

**解决方案**：
- 将训练迭代数从100k降至30k（减少70%训练时间）
- 同步调整fix_iter、val_interval、checkpoint_interval等参数
- 更频繁的验证和日志记录，便于监控微调效果

**代码修改**：

**文件**: `configs/basicvsrpp/mosaic_restoration_generic_stage1.py`
- **第18行**: fix_iter从5000降至2000
  ```python
  train_cfg=dict(fix_iter=2000),  # 调整：从5000降至2000，保持fix_iter占比约6.7%
  ```
- **第73-74行**: max_iters从100k降至30k，val_interval从5k降至3k
  ```python
  max_iters=30_000,      # 调整：从100k降至30k（微调推荐值）
  val_interval=3_000)     # 调整：从5k降至3k，更频繁验证以监控微调效果
  ```
- **第95-96行**: checkpoint间隔从2000降至1500，logger间隔从100降至50
  ```python
  checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=1500, out_dir=save_dir),  # 调整：从2000降至1500，确保有足够checkpoints
  logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False))  # 调整：从100降至50，更频繁记录以便观察微调过程
  ```

**预期效果**：
- ✅ 训练时间从约28小时缩短至约8-9小时（节省约19小时）
- ✅ 获得约20个checkpoint（每1500 iter保存）以供选择最佳模型
- ✅ 每3000 iter验证一次，共10次验证，及时发现过拟合
- ✅ 更频繁的日志输出便于实时监控训练状态

**参数对比总结**：
| 参数 | 原值 | 新值 | 说明 |
|------|------|------|------|
| max_iters | 100,000 | 30,000 | 微调场景推荐值 |
| fix_iter | 5,000 | 2,000 | 保持约6.7%的冻结比例 |
| val_interval | 5,000 | 3,000 | 更频繁验证（共10次） |
| checkpoint_interval | 2,000 | 1,500 | 确保足够的保存点 |
| logger_interval | 100 | 50 | 更细粒度的日志 |

---

### 🚀 功能增强：支持多GPU分布式训练

**需求/问题**：
- BasicVSR++训练脚本只能使用单GPU，需要支持多GPU并行训练以提高训练效率

**解决方案**：
1. 配置DDP（DistributedDataParallel）支持，解决未使用参数错误
2. 创建便捷的多GPU训练启动脚本

**代码修改**：

**文件1**: `configs/basicvsrpp/mosaic_restoration_generic_stage1.py`
- **第29行**: 添加注释说明batch_size含义
  ```python
  batch_size=2,  # 每个GPU的batch_size，总batch_size = 2 * 2 = 4
  ```
- **第96-99行**: 新增模型包装器配置
  ```python
  # 添加模型包装器配置以解决DDP未使用参数问题
  model_wrapper_cfg = dict(
      type='MMDistributedDataParallel',
      find_unused_parameters=True)
  ```

**文件2**: `run_multi_gpu_training.sh` (新增文件)
```bash
#!/bin/bash

# 设置使用的GPU
export CUDA_VISIBLE_DEVICES=0,1

# 使用torch.distributed.launch启动多GPU训练
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port=29500 \
    scripts/training/train-mosaic-restoration-basicvsrpp.py \
    configs/basicvsrpp/mosaic_restoration_generic_stage1.py \
    --launcher pytorch

echo "多GPU训练完成！"
```

**文件3**: `run_multi_gpu_training_torchrun.sh` (新增文件)
```bash
#!/bin/bash

# 设置使用的GPU
export CUDA_VISIBLE_DEVICES=0,1

# 使用torchrun启动多GPU训练（PyTorch 1.10+推荐方式）
torchrun \
    --nproc_per_node=2 \
    --master_port=29500 \
    scripts/training/train-mosaic-restoration-basicvsrpp.py \
    configs/basicvsrpp/mosaic_restoration_generic_stage1.py \
    --launcher pytorch

echo "多GPU训练完成！"
```

**使用方法**：
```bash
# 方法1：使用torch.distributed.launch
cd /root/lada
./run_multi_gpu_training.sh

# 方法2：使用torchrun（推荐PyTorch 1.10+）
cd /root/lada
./run_multi_gpu_training_torchrun.sh
```

**效果**：
- ✅ 成功在2个GPU上并行训练
- ✅ GPU利用率达到91-99%
- ✅ 每个GPU使用约19GB显存
- ✅ 训练速度显著提升



