import argparse
import os
import os.path as osp

# 🔧 必须在导入MMEngine之前设置，否则无效
# 跳过dataloader状态恢复，避免resume时卡在"Advance dataloader N steps"
os.environ['MMENGINE_RESUME_DATALOADER'] = '0'

from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from lada.basicvsrpp import register_all_modules

register_all_modules()

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume', action='store_true', help='Whether to resume checkpoint.')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--load-from', help='specify checkpoint path to start from')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir:  # none or empty str
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # 自动检测是否需要resume
    last_checkpoint_file = osp.join(cfg.work_dir, 'last_checkpoint')
    if args.resume:
        cfg.resume = True
        print("✅ Resume mode enabled (--resume flag)")
    elif osp.exists(last_checkpoint_file):
        # 自动resume：如果存在last_checkpoint文件且没有明确指定--load-from
        if not args.load_from:
            cfg.resume = True
            print(f"✅ Auto-resume enabled: Found checkpoint at {last_checkpoint_file}")
            with open(last_checkpoint_file, 'r') as f:
                last_ckpt_path = f.read().strip()
                print(f"   Resuming from: {last_ckpt_path}")
            print("   ⚡ Fast resume mode: Dataloader will restart (not skip already-seen data)")
            print("      This avoids the slow 'Advance dataloader' step")
        else:
            print(f"⚠️  Found existing checkpoint but --load-from specified, starting fresh")
    
    if args.load_from:
        cfg.load_from = args.load_from
        print(f"📂 Loading weights from: {args.load_from}")

    # build the runner from config
    runner = Runner.from_cfg(cfg)

    print(f'Working directory: {cfg.work_dir}')
    print(f'Log directory: {runner._log_dir}')

    # start training
    runner.train()

    print(f'Log saved under {runner._log_dir}')
    print(f'Checkpoint saved under {cfg.work_dir}')


if __name__ == '__main__':
    main()
