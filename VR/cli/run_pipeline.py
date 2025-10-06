# CLI to run the full VR pipeline (all stages with stage screenshots)
import argparse
import os
import sys
from typing import Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from VR.pipeline import VRPipeline, PipelineConfig  # type: ignore
from VR.undist import list_undistort_modes  # type: ignore
from VR.utils.jsonl_logger import JsonlLogger  # type: ignore


def _parse_resolution(value: str) -> Tuple[int, int]:
    cleaned = value.lower().replace(" ", "")
    if "x" not in cleaned:
        raise argparse.ArgumentTypeError("Resolution must be formatted as WIDTHxHEIGHT, e.g. 3840x2160")
    w_str, h_str = cleaned.split("x", 1)
    try:
        width = int(w_str)
        height = int(h_str)
    except ValueError as exc:  # pragma: no cover - argparse handles error
        raise argparse.ArgumentTypeError("Resolution components must be integers") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("Resolution components must be positive")
    return width, height


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
    available_modes = tuple(list_undistort_modes())
    p.add_argument("--undist-mode", default="opencv_default", choices=available_modes, help="Undistortion strategy")
    p.add_argument("--undist-calibration", default=None, help="Calibration file required by calibrated/woodscape strategies")
    p.add_argument("--undist-balance", type=float, default=None, help="Balance factor used by OpenCV calibrated fisheye modes (default=1.0)")
    p.add_argument("--undist-reference", type=_parse_resolution, default=None, help="Reference resolution for calibration scaling (e.g. 3840x2160)")
    p.add_argument("--undist-hfov", type=float, default=None, help="Horizontal FOV in degrees for Woodscape cylindrical mode (default 190)")
    p.add_argument("--undist-vfov", type=float, default=None, help="Vertical FOV in degrees for Woodscape cylindrical mode (default 143)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "logs", "pipeline.jsonl")
    logger = JsonlLogger(log_path)
    try:
        undist_params = {}
        if args.undist_calibration:
            undist_params["calibration_path"] = args.undist_calibration
        if args.undist_balance is not None:
            undist_params["balance"] = args.undist_balance
        if args.undist_reference is not None:
            undist_params["reference_size"] = args.undist_reference
        if args.undist_hfov is not None:
            undist_params["hfov_deg"] = args.undist_hfov
        if args.undist_vfov is not None:
            undist_params["vfov_deg"] = args.undist_vfov

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
            undist_mode=args.undist_mode,
            undist_params=undist_params,
        )
        runner = VRPipeline(cfg, logger)
        summary = runner.run()
        logger.info("summary", **summary)
        print(f"Pipeline done. Frames: {summary['frames_processed']} log: {log_path}")
    finally:
        logger.close()


if __name__ == "__main__":
    main()
