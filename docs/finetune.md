
快速微调 Lada 输出“茄子”——最小成本实操

目标
- 让模型把马赛克区域稳定替换为“茄子”。
- 仅做 Stage1 小步微调即可达标；Stage2 为可选抛光，不是必须。

Stage1 vs Stage2（做什么）
- Stage1（BasicVSR/Charbonnier）
  - 学像素级重建与时序对齐，稳定、轻量、对小数据友好。
  - 适合“固定替换为茄子”的功能性微调（成本最低）。
- Stage2（BasicVSR++ GAN + 感知）
  - 在 Stage1 基础上做外观抛光（质感/自然度），更重更敏感。
  - 仅当追求更自然的质感时才需要。

前置：数据要求
- 你的数据集已按官方流程生成，但要确保：
  - 输入（LQ/马赛克）保持原样；
  - GT（高质量/相对路径 `relative_nsfw_video_path`）已经是你“合成好茄子”的版本。
- 在 `configs/basicvsrpp/mosaic_restoration_generic_stage1.py` 里通过 `--cfg-options data_root=...` 覆盖到你的数据根目录（其下存在 `crop_unscaled_meta/`、`val/crop_unscaled_meta/`）。

一、Stage1 小步微调（推荐方案，最省）
- 命令（把尖括号换成你自己的路径）：
```
python scripts/training/train-mosaic-restoration-basicvsrpp.py \
  configs/basicvsrpp/mosaic_restoration_generic_stage1.py \
  --load-from <可选：你的Stage1起点ckpt，如 experiments/basicvsrpp/mosaic_restoration_generic_stage1/iter_5000.pth> \
  --cfg-options \
    data_root=/path/to/eggplant_dataset \
    train_dataloader.dataset.degrade=False \
    train_dataloader.dataset.random_mosaic_params=False \
    train_dataloader.dataset.num_frame=12 \
    train_dataloader.batch_size=1 \
    train_cfg.max_iters=3000 \
    default_hooks.logger.interval=10 \
    default_hooks.checkpoint.interval=500 \
    optim_wrapper.optimizer.lr=5e-5 \
    optim_wrapper.paramwise_cfg.custom_keys.spynet.lr_mult=0 \
    optim_wrapper.paramwise_cfg.custom_keys.feat_extract.lr_mult=0 \
    optim_wrapper.paramwise_cfg.custom_keys.deform_align.lr_mult=0 \
    optim_wrapper.paramwise_cfg.custom_keys.backbone.lr_mult=0
```
- 说明：
  - 关闭随机马赛克与退化，直接读取你预生成的 LQ/GT，首批就很快；
  - 短序列、batch=1、几千步就能把输出“锁定”为茄子；
  - 冻结 `spynet/feat_extract/deform_align/backbone`，只训重建头，收敛更快、更稳。

二、离线验证（不走检测/拼接，全看模型本身）
- 准备一个目录放待测马赛克小片段（任意尺寸，脚本会缩放/Pad）。
- 命令：
```
python scripts/evaluation/validate-basicvsrpp.py \
  --config-path configs/basicvsrpp/mosaic_restoration_generic_stage1.py \
  --model-path experiments/basicvsrpp/mosaic_restoration_generic_stage1/iter_3000.pth \
  --in-dir /path/to/test_mosaic_clips \
  --out-dir /path/to/test_restored
```
- 打开输出视频，若已稳定是“茄子”，说明模型本身 OK；否则回到“一”再多训 1–2k 步或适当增大学习率。

三、用 CLI 跑整段视频（仍用 Stage1 配置与权重）
- 命令：
```
lada-cli --input /path/to/your_video.mp4 \
  --device cuda:0 --codec h264_nvenc \
  --mosaic-restoration-model basicvsrpp \
  --mosaic-restoration-config-path configs/basicvsrpp/mosaic_restoration_generic_stage1.py \
  --mosaic-restoration-model-path experiments/basicvsrpp/mosaic_restoration_generic_stage1/iter_3000.pth \
  --max-clip-length 240 \
  --hw-decode \
  --mosaic-detection-batch-size 16 \
  --detector-queue-size 32 \
  --mosaic-detection-model-path model_weights/lada_mosaic_detection_model_v3.1_accurate.pt
```
- 提示：
  - 如果离线验证“是茄子”，但 CLI 输出不是，多数是“检测没打上/阈值偏高/未形成 clip”；先换更敏感检测权重（如上），必要时再排查阈值与场景切分。

四、可选：Stage2 轻对抗微调（仅为抛光，不必做）
- 想在 Stage1 成功基础上增加质感时，再做少量步数的 Stage2 微调：
```
python scripts/training/train-mosaic-restoration-basicvsrpp.py \
  configs/basicvsrpp/mosaic_restoration_generic_stage2.py \
  --load-from <你的Stage1转换权重iter_xxxx_converted.pth> \
  --cfg-options \
    data_root=/path/to/eggplant_dataset \
    train_dataloader.dataset.degrade=False \
    train_dataloader.dataset.random_mosaic_params=False \
    train_dataloader.dataset.num_frame=12 \
    train_dataloader.batch_size=1 \
    train_cfg.max_iters=5000 \
    default_hooks.logger.interval=10 \
    default_hooks.checkpoint.interval=500 \
    optim_wrapper.generator.optimizer.lr=2e-5 \
    optim_wrapper.discriminator.optimizer.lr=5e-5 \
    model.perceptual_loss.perceptual_weight=0.05 \
    model.gan_loss.loss_weight=0.02 \
    optim_wrapper.generator.paramwise_cfg.custom_keys.spynet.lr_mult=0 \
    optim_wrapper.generator.paramwise_cfg.custom_keys.feat_extract.lr_mult=0 \
    optim_wrapper.generator.paramwise_cfg.custom_keys.deform_align.lr_mult=0 \
    optim_wrapper.generator.paramwise_cfg.custom_keys.backbone.lr_mult=0
```
- 若只想先锁定功能，可把 `model.gan_loss.loss_weight=0`、`model.perceptual_loss.perceptual_weight=0` 跑 2–3k 步确认输出，再逐步加回小权重。

常见问题与排查
- CLI 输出不是茄子，但离线验证是：
  - 多半是检测命中率问题。先换 `v3.1_accurate`，必要时再降低阈值（如需可在 CLI 中暴露该参数）。
- 训练“卡住”或首个 batch 很慢：
  - 是随机马赛克/退化在 CPU 侧做视频重编码，建议先关 `degrade/random_mosaic_params` 验流程；
  - 把 `default_hooks.logger.interval` 调小可快速看到迭代日志。
- 权重体积异常小（几百 KB）：
  - 说明文件损坏或是 LFS 指针；正常推理权重要几十 MB。请重新导出/同步。

建议的最短路径（复盘）
1) 仅用 Stage1 + 冻结前半 + 小步微调 → 2–3k 步。
2) 用评估脚本离线验证小片段 → 确认“是茄子”。
3) 用 CLI + Stage1 配置和权重跑整段 → 如有偏差，优先调检测侧。

