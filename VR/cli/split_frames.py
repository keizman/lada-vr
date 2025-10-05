# CLI for running the frame split stage.
import argparse
import os
import sys
from typing import Optional

# Allow running from repo root without installing as package
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from VR.utils.jsonl_logger import JsonlLogger  # type: ignore
from VR.io.frame_splitter import FrameSplitter, SplitConfig  # type: ignore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VR frame input & split (SBS/OU) with logs & thumbs")
    p.add_argument("--input", required=True, help="Input video path")
    p.add_argument("--out-dir", required=True, help="Output directory for logs & thumbs")
    p.add_argument("--layout", default="auto", choices=["auto", "sbs", "ou", "mono"], help="Force layout or auto-detect")
    p.add_argument("--thumb-interval", type=int, default=10, help="Save thumbnails every N frames (0 disables)")
    p.add_argument("--thumb-max-w", type=int, default=480, help="Max width for thumbnails")
    p.add_argument("--max-frames", type=int, default=None, help="Limit frames for pilot runs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "logs", "split.jsonl")
    logger = JsonlLogger(log_path)
    try:
        cfg = SplitConfig(
            input_path=args.input,
            out_dir=args.out_dir,
            layout=args.layout,  # type: ignore
            thumb_interval=args.thumb_interval,
            thumb_max_w=args.thumb_max_w,
            max_frames=args.max_frames,
        )
        splitter = FrameSplitter(cfg, logger)
        summary = splitter.process()
        logger.info("split:summary", **summary)
        print(f"Done. Frames: {summary['frames_processed']} log: {log_path}")
    finally:
        logger.close()


if __name__ == "__main__":
    main()

