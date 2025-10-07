from mmengine.config import read_base

with read_base():
    from ._base_.default_runtime import *


experiment_name = 'mosaic_restoration_frozen_finetune'  # 🔥 新实验：冻结层微调
work_dir = f'./experiments/basicvsrpp/{experiment_name}'
save_dir = './experiments/basicvsrpp'

model = dict(
    type='BasicVSR',
    generator=dict(
        type='BasicVSRPlusPlusNet',
        mid_channels=64,
        num_blocks=15,
        spynet_pretrained='model_weights/3rd_party/spynet_20210409-c6c1bd09.pth'),
    pixel_loss=dict(type='CharbonnierLoss', loss_weight=1.0, reduction='mean'),
    train_cfg=dict(fix_iter=500),  # 🔥 快速试跑：500步后开始训练（前期冻结SpyNet）
    data_preprocessor=dict(
        type='DataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
    ))

data_root = '/root/autodl-tmp/train_with_mosaic'
val_root = '/root/autodl-tmp/train_with_mosaic_val'  # 验证集路径

train_dataloader = dict(
    num_workers=4,
    batch_size=1,  # 🔥 快速试跑：每GPU 1个样本
    persistent_workers=False,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='MosaicVideoDataset',
        metadata_root_dir=data_root + "/crop_unscaled_meta",
        num_frame=15,  # 🔥 快速试跑：减少到15帧
        degrade=False,  # 不额外退化（不加压缩/模糊/噪声）
        use_hflip=True,
        repeatable_random=False,
        random_mosaic_params=True,  # ✅ 实时生成马赛克（数据集没有预生成）
        filter_watermark=False,
        filter_nudenet_nsfw=False,
        filter_video_quality=False,
        lq_size=256),
    collate_fn=dict(type='default_collate'))

val_dataloader = dict(
    num_workers=1,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='MosaicVideoDataset',
        metadata_root_dir=val_root + "/crop_unscaled_meta",  # 使用独立验证集
        num_frame=15,  # 🔥 快速试跑：减少到15帧
        degrade=False,  # 不额外退化
        use_hflip=False,
        repeatable_random=True,
        random_mosaic_params=True,  # ✅ 实时生成，但repeatable保证可重复
        filter_watermark=False,
        filter_nudenet_nsfw=False,
        filter_video_quality=False,
        lq_size=256),
    collate_fn=dict(type='default_collate'))

val_evaluator = dict(
    type='Evaluator', metrics=[
        dict(type='PSNR'),
        dict(type='SSIM'),
    ])

train_cfg = dict(
    type='IterBasedTrainLoop', 
    max_iters=3_000,        # 🔥 快速试跑：3k步观察效果
    val_interval=500)       # 🔥 每500步验证一次
val_cfg = dict(type='MultiValLoop')

# ============================================================
# 🎯 微调策略：冻结前半段，只训练后半段重建层
# ============================================================
# 原理：
#   - 前半段（光流+对齐+特征提取）已学会理解模糊视频
#   - 后半段（重建层）直接映射到像素，最容易调整颜色/纹理
#   - 集中学习能力到最后几层，适配茄子目标
#
# 冻结层 (lr_mult=0.0):
#   ├── spynet            # 光流估计
#   ├── feat_extract      # 特征提取（2个stride conv + residual blocks）
#   ├── deform_align      # 可变形对齐
#   └── backbone          # 多分支传播
#
# 训练层 (较高LR):
#   ├── reconstruction    # 特征聚合
#   ├── upsample1         # 第一次上采样
#   ├── upsample2         # 第二次上采样
#   ├── conv_hr           # 高分辨率卷积（决定颜色/纹理）
#   └── conv_last         # 最终输出层（直接输出RGB像素）
# ============================================================
optim_wrapper = dict(
    constructor='DefaultOptimWrapperConstructor',
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=2e-4, betas=(0.9, 0.99)),  # 提高基础LR
    paramwise_cfg=dict(
        custom_keys={
            # 🔒 冻结前半段
            'spynet': dict(lr_mult=0.0),
            'generator.feat_extract': dict(lr_mult=0.0),
            'generator.deform_align': dict(lr_mult=0.0),
            'generator.backbone': dict(lr_mult=0.0),
            
            # 🔥 训练后半段重建层（使用较高学习率）
            'generator.reconstruction': dict(lr_mult=1.0),
            'generator.upsample1': dict(lr_mult=1.0),
            'generator.upsample2': dict(lr_mult=1.0),
            'generator.conv_hr': dict(lr_mult=1.0),
            'generator.conv_last': dict(lr_mult=1.0),
        }
    )
)


vis_backends = [dict(type='TensorboardVisBackend')]
visualizer = dict(
    name='visualizer',
    type='ConcatImageVisualizer',
    vis_backends=vis_backends,
    fn_key='gt_path',
    img_keys=['gt_img', 'input', 'pred_img'],
    bgr2rgb=True)
custom_hooks = [dict(type='BasicVisualizationHook', interval=5)]

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=1000, out_dir=save_dir),  # 🔥 快速试跑：每500步保存checkpoint
    logger=dict(type='LoggerHook', interval=40, log_metric_by_epoch=False))  # 🔥 快速试跑：每20步记录日志

# 添加模型包装器配置以解决DDP未使用参数问题
model_wrapper_cfg = dict(
    type='MMDistributedDataParallel',
    find_unused_parameters=True)
