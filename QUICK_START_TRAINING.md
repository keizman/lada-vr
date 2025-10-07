# 快速开始训练

## 🚀 训练脚本

### 从头开始
```bash
bash run_multi_gpu_training_fresh.sh
```

### 继续训练（推荐）
```bash
bash run_multi_gpu_training_no_hang.sh
```

## ⚙️ 配置调整

直接修改 `configs/basicvsrpp/mosaic_restoration_generic_stage1.py`：

```python
# 快速试跑
num_frame = 15          # 减少帧数
batch_size = 1          # 减小batch
max_iters = 3_000       # 3k步试跑

# 层冻结微调（在 optim_wrapper.paramwise_cfg.custom_keys 中添加）
'generator.feat_extract': dict(lr_mult=0.0),   # 冻结
'generator.deform_align': dict(lr_mult=0.0),   # 冻结
'generator.backbone': dict(lr_mult=0.0),       # 冻结
# 只训练重建层：reconstruction, upsample*, conv_hr, conv_last
```

## 📊 监控

```bash
# TensorBoard
tensorboard --logdir=./experiments/basicvsrpp --port=6006

# 查看日志
tail -f ./experiments/basicvsrpp/*/$(ls -t ./experiments/basicvsrpp/*/2* | head -1)/*.log
```

## 📝 配置说明详见文件开头注释

