#!/usr/bin/env python3
"""
自动划分训练集和验证集的脚本

功能：
1. 从训练集中随机抽取指定比例或数量的样本作为验证集
2. 自动复制对应的视频、mask和元数据文件
3. 从原训练集中移除验证集样本（可选）
4. 生成详细的划分报告

使用方法：
    # 按比例划分（默认10%）
    python split-train-val-dataset.py --src-root /root/autodl-tmp/train_with_mosaic
    
    # 按固定数量划分
    python split-train-val-dataset.py --src-root /root/autodl-tmp/train_with_mosaic --val-size 500
    
    # 不从训练集删除（仅复制）
    python split-train-val-dataset.py --src-root /root/autodl-tmp/train_with_mosaic --no-remove
"""

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import List, Dict, Tuple
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description='Split mosaic restoration dataset into train and validation sets'
    )
    parser.add_argument(
        '--src-root',
        type=str,
        required=True,
        help='Source dataset root directory (e.g., /root/autodl-tmp/train_with_mosaic)'
    )
    parser.add_argument(
        '--dst-root',
        type=str,
        default=None,
        help='Destination validation set root directory (default: {src-root}_val)'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.1,
        help='Validation set ratio (default: 0.1, i.e., 10%%)'
    )
    parser.add_argument(
        '--val-size',
        type=int,
        default=None,
        help='Fixed validation set size (overrides --val-ratio if specified)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--no-remove',
        action='store_true',
        help='Do not remove validation samples from training set (only copy)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview what would be done without actually copying/moving files'
    )
    return parser.parse_args()


def collect_metadata_files(src_root: Path) -> List[Path]:
    """收集所有元数据JSON文件"""
    meta_dir = src_root / 'crop_unscaled_meta'
    if not meta_dir.exists():
        raise FileNotFoundError(f"Metadata directory not found: {meta_dir}")
    
    meta_files = list(meta_dir.glob('*.json'))
    if not meta_files:
        raise FileNotFoundError(f"No JSON files found in {meta_dir}")
    
    return meta_files


def parse_metadata(meta_path: Path) -> Dict:
    """解析元数据文件"""
    try:
        with open(meta_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to parse {meta_path}: {e}")
        return None


def resolve_file_paths(meta_path: Path, metadata: Dict) -> Tuple[Path, Path]:
    """解析视频和mask文件的绝对路径"""
    base_dir = meta_path.parent
    
    # 解析相对路径
    video_rel = metadata.get('relative_nsfw_video_path', '')
    mask_rel = metadata.get('relative_mask_video_path', '')
    
    if not video_rel or not mask_rel:
        raise ValueError(f"Missing video/mask paths in {meta_path.name}")
    
    # 构建绝对路径
    video_path = (base_dir / video_rel).resolve()
    mask_path = (base_dir / mask_rel).resolve()
    
    return video_path, mask_path


def create_val_directory_structure(dst_root: Path):
    """创建验证集目录结构"""
    for subdir in ['crop_unscaled_img', 'crop_unscaled_mask', 'crop_unscaled_meta']:
        (dst_root / subdir).mkdir(parents=True, exist_ok=True)


def copy_sample(meta_path: Path, video_path: Path, mask_path: Path, 
                dst_root: Path, dry_run: bool = False) -> bool:
    """复制一个样本到验证集"""
    try:
        dst_meta = dst_root / 'crop_unscaled_meta' / meta_path.name
        dst_video = dst_root / 'crop_unscaled_img' / video_path.name
        dst_mask = dst_root / 'crop_unscaled_mask' / mask_path.name
        
        if dry_run:
            print(f"  [DRY-RUN] Would copy:")
            print(f"    {meta_path} -> {dst_meta}")
            print(f"    {video_path} -> {dst_video}")
            print(f"    {mask_path} -> {dst_mask}")
            return True
        
        # 检查源文件是否存在
        if not video_path.exists():
            print(f"  Warning: Video file not found: {video_path}")
            return False
        if not mask_path.exists():
            print(f"  Warning: Mask file not found: {mask_path}")
            return False
        
        # 复制文件
        shutil.copy2(meta_path, dst_meta)
        shutil.copy2(video_path, dst_video)
        shutil.copy2(mask_path, dst_mask)
        
        return True
    except Exception as e:
        print(f"  Error copying sample {meta_path.name}: {e}")
        return False


def remove_sample(meta_path: Path, video_path: Path, mask_path: Path, 
                  dry_run: bool = False) -> bool:
    """从训练集中删除样本"""
    try:
        if dry_run:
            print(f"  [DRY-RUN] Would remove:")
            print(f"    {meta_path}")
            print(f"    {video_path}")
            print(f"    {mask_path}")
            return True
        
        meta_path.unlink(missing_ok=True)
        video_path.unlink(missing_ok=True)
        mask_path.unlink(missing_ok=True)
        
        return True
    except Exception as e:
        print(f"  Error removing sample {meta_path.name}: {e}")
        return False


def main():
    args = parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    
    # 路径设置
    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root) if args.dst_root else Path(str(src_root) + '_val')
    
    print("=" * 80)
    print("训练集/验证集划分工具")
    print("=" * 80)
    print(f"源训练集路径: {src_root}")
    print(f"目标验证集路径: {dst_root}")
    print(f"随机种子: {args.seed}")
    
    # 收集元数据文件
    print("\n[1/5] 收集元数据文件...")
    meta_files = collect_metadata_files(src_root)
    total_samples = len(meta_files)
    print(f"  找到 {total_samples} 个样本")
    
    # 确定验证集大小
    if args.val_size is not None:
        val_size = min(args.val_size, total_samples)
        print(f"\n[2/5] 使用固定验证集大小: {val_size} 个样本")
    else:
        val_size = max(1, int(total_samples * args.val_ratio))
        print(f"\n[2/5] 使用验证集比例: {args.val_ratio * 100:.1f}% = {val_size} 个样本")
    
    train_size = total_samples - val_size
    print(f"  训练集: {train_size} 个样本 ({train_size/total_samples*100:.1f}%)")
    print(f"  验证集: {val_size} 个样本 ({val_size/total_samples*100:.1f}%)")
    
    # 随机抽样
    print(f"\n[3/5] 随机抽样 {val_size} 个样本...")
    val_samples = random.sample(meta_files, val_size)
    print(f"  完成抽样")
    
    # 创建目标目录
    if not args.dry_run:
        print(f"\n[4/5] 创建验证集目录结构...")
        create_val_directory_structure(dst_root)
        print(f"  目录结构已创建")
    else:
        print(f"\n[4/5] [DRY-RUN] 跳过目录创建")
    
    # 复制样本
    print(f"\n[5/5] 复制样本到验证集...")
    success_count = 0
    failed_samples = []
    
    for i, meta_path in enumerate(val_samples, 1):
        if i % 100 == 0 or i == 1:
            print(f"  处理进度: {i}/{val_size}")
        
        # 解析元数据
        metadata = parse_metadata(meta_path)
        if metadata is None:
            failed_samples.append(meta_path.name)
            continue
        
        try:
            # 获取文件路径
            video_path, mask_path = resolve_file_paths(meta_path, metadata)
            
            # 复制到验证集
            if copy_sample(meta_path, video_path, mask_path, dst_root, args.dry_run):
                success_count += 1
                
                # 从训练集删除
                if not args.no_remove:
                    remove_sample(meta_path, video_path, mask_path, args.dry_run)
            else:
                failed_samples.append(meta_path.name)
                
        except Exception as e:
            print(f"  Error processing {meta_path.name}: {e}")
            failed_samples.append(meta_path.name)
    
    # 生成报告
    print("\n" + "=" * 80)
    print("划分完成！")
    print("=" * 80)
    
    if args.dry_run:
        print("\n⚠️  这是一次模拟运行，没有实际修改任何文件")
    
    print(f"\n📊 统计信息:")
    print(f"  原始训练集样本数: {total_samples}")
    print(f"  成功复制到验证集: {success_count}")
    print(f"  复制失败: {len(failed_samples)}")
    
    if not args.no_remove and not args.dry_run:
        remaining = len(list((src_root / 'crop_unscaled_meta').glob('*.json')))
        print(f"  训练集剩余样本数: {remaining}")
    
    print(f"\n📁 文件位置:")
    print(f"  训练集: {src_root}")
    print(f"  验证集: {dst_root}")
    
    if failed_samples:
        print(f"\n⚠️  失败的样本 ({len(failed_samples)}):")
        for name in failed_samples[:10]:
            print(f"    - {name}")
        if len(failed_samples) > 10:
            print(f"    ... 还有 {len(failed_samples) - 10} 个")
    
    print(f"\n✅ 下一步:")
    print(f"  1. 检查验证集目录: {dst_root}")
    print(f"  2. 更新训练配置文件中的 val_dataloader.dataset.metadata_root_dir")
    print(f"  3. 重新启动训练以使用新的验证集")
    
    if args.dry_run:
        print(f"\n💡 如果预览结果正确，请移除 --dry-run 参数重新运行")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n操作已取消")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

