# CLI to run the full VR pipeline (all stages with stage screenshots)
import argparse
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from VR.pipeline import VRPipeline, PipelineConfig  # type: ignore
from VR.utils.jsonl_logger import JsonlLogger  # type: ignore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run VR pipeline: metadata -> split -> undistort -> AI -> redistort -> recompose")
    p.add_argument("--input", required=True, help="Input video path")
    p.add_argument("--out-dir", required=True, help="Output directory for logs/stages")
    p.add_argument("--layout", default="auto", choices=["auto", "sbs", "ou", "mono"], help="Force layout or auto-detect")
    p.add_argument("--projection", default="auto", choices=["auto", "fisheye180", "equirect180", "equirect360"], help="Input projection model")
    p.add_argument("--plane-mode", default="auto", choices=["auto", "persp_single", "persp_tiles", "equirect_plane", "roi_persp", "roi_grid3x3"], help="Processing plane mode for equirect inputs")
    p.add_argument("--tiles-x", type=int, default=4, help="Tiles across yaw for persp_tiles")
    p.add_argument("--tiles-y", type=int, default=2, help="Tiles across pitch for persp_tiles")
    p.add_argument("--fov", type=float, default=110.0, help="Perspective FOV in degrees (single/tiles)")
    p.add_argument("--yaw", type=float, default=0.0, help="Yaw in degrees (single mode)")
    p.add_argument("--pitch", type=float, default=0.0, help="Pitch in degrees (single mode)")
    p.add_argument("--roll", type=float, default=0.0, help="Roll in degrees (single mode)")
    p.add_argument("--save-every-n", type=int, default=10, help="Save stage images every N frames")
    p.add_argument("--max-frames", type=int, default=None, help="Limit frames processed for pilot runs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "logs", "pipeline.jsonl")
    logger = JsonlLogger(log_path)
    try:
        cfg = PipelineConfig(
            input_path=args.input,
            out_dir=args.out_dir,
            layout=args.layout,  # type: ignore
            projection=args.projection,
            plane_mode=args.plane_mode,
            tiles_x=args.tiles_x,
            tiles_y=args.tiles_y,
            fov_deg=args.fov,
            yaw_deg=args.yaw,
            pitch_deg=args.pitch,
            roll_deg=args.roll,
            save_every_n=args.save_every_n,
            max_frames=args.max_frames,
        )
        runner = VRPipeline(cfg, logger)
        summary = runner.run()
        logger.info("summary", **summary)
        print(f"Pipeline done. Frames: {summary['frames_processed']} log: {log_path}")
    finally:
        logger.close()


if __name__ == "__main__":
    main()

