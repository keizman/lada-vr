# VR Pipeline README

This document describes the VR processing pipeline stages, the output folders, and how to run the pilot pipeline to generate per-stage screenshots and structured logs.

Overview
- Goal: process VR (SBS/OU) frames through multiple geometric stages for inspection and later AI restoration. Each stage saves sampled frame snapshots for visual verification.
- Pilot status: full stage flow exists, with undistortion using an OpenCV fisheye model and a redistortion placeholder (identity). The AI stage is a passthrough. These will be replaced/enhanced later.

Stages and output folders
For an output root OUT_DIR, stage images are saved under OUT_DIR/stages/...

- 00_input/
  - A snapshot of the original input frame (sampled every N frames).
  - Path: OUT_DIR/stages/00_input/frame_XXXXXX.jpg

- 10_split/
  - Left and Right images after splitting SBS or OU layouts.
  - Paths:
    - OUT_DIR/stages/10_split/left/frame_XXXXXX.jpg
    - OUT_DIR/stages/10_split/right/frame_XXXXXX.jpg

- 20_undist/
  - If projection is equirect (360/180): plane modes
    - equirect_plane: keep equirect as the processing plane (no geometric change)
    - persp_single: a single perspective view (configurable FOV/yaw/pitch/roll)
    - persp_tiles: multiple perspective views covering the full sphere/hemisphere; the stage saves a mosaic and per-tile images
    - roi_persp: a small undistorted ROI extracted from the mid-lower 3x3 grid region (X∈[W/6,5W/6], Y∈[H/2,5H/6]); equirect180 uses single view; equirect360 uses two views (±60° yaw)
      - ROI images saved to OUT_DIR/stages/20_roi/left|right/frame_XXXXXX_vN.jpg
  - If projection is fisheye180: this stage saves fisheye undistorted (rectified) images using OpenCV fisheye model.
  - Paths:
    - OUT_DIR/stages/20_undist/left/frame_XXXXXX.jpg (mosaic or single/preserved)
    - OUT_DIR/stages/20_undist/left/frame_XXXXXX/tile_R_C.jpg (if tiles enabled)
    - OUT_DIR/stages/20_roi/... (if roi_persp)
    - OUT_DIR/stages/20_undist/right/... (same for right eye if present)

- 30_ai/
  - Placeholder stage (currently passthrough of undistorted images). This will be replaced by the chosen AI restoration.
  - Paths:
    - OUT_DIR/stages/30_ai/left/frame_XXXXXX.jpg
    - OUT_DIR/stages/30_ai/right/frame_XXXXXX.jpg

- 40_redist/
  - Redistortion stage: maps the processed plane back to the original split-sized images.
    - In roi_persp mode: ROI views are composited back onto the split-sized images with Hanning feathering to avoid seams.
  - Paths:
    - OUT_DIR/stages/40_redist/left/frame_XXXXXX.jpg
    - OUT_DIR/stages/40_redist/right/frame_XXXXXX.jpg

- 50_recompose_sbs/
  - Final recomposed frame (SBS or OU) from the redistorted left/right images.
  - Path:
    - OUT_DIR/stages/50_recompose_sbs/frame_XXXXXX.jpg

Logs
- Structured JSONL logs are saved to:
  - OUT_DIR/logs/pipeline.jsonl (full pipeline)
  - The log contains entries for metadata, pipeline start/finish, per-frame summaries, and undistortion parameter info.

Configuration
- Layout: auto | sbs | ou | mono (auto detects SBS/OU by resolution heuristic)
- save_every_n: Save stage images every N frames (reduce disk IO; set to 1 for every frame)
- max_frames: Limit frames processed (useful for pilots)

Running the pipeline (examples)
- Python dependencies: `pip install opencv-python`
- Example (Windows PowerShell):
  - Single view (perspective):
    - python VR/cli/run_pipeline.py \
      --input "K:\\tmp\\v\\code\\demosaic\\8199881-4k-1m.mp4" \
      --out-dir "K:\\tmp\\v\\code\\demosaic\\VR_out\\pilot_single" \
      --layout sbs --projection equirect360 --plane-mode persp_single \
      --fov 110 --yaw 0 --pitch 0 --roll 0 --save-every-n 10 --max-frames 1200
  - Tiles covering full 360:
    - python VR/cli/run_pipeline.py \
      --input "K:\\tmp\\v\\code\\demosaic\\8199881-4k-1m.mp4" \
      --out-dir "K:\\tmp\\v\\code\\demosaic\\VR_out\\pilot_tiles" \
      --layout sbs --projection equirect360 --plane-mode persp_tiles \
      --tiles-x 4 --tiles-y 2 --fov 100 --save-every-n 10 --max-frames 600
  - ROI mid-lower small views (recommended for testing without full coverage):
    - python VR/cli/run_pipeline.py \
      --input "K:\\tmp\\v\\code\\demosaic\\8199881-4k-1m.mp4" \
      --out-dir "K:\\tmp\\v\\code\\demosaic\\VR_out\\pilot_roi" \
      --layout sbs --projection equirect360 --plane-mode roi_persp \
      --fov 120 --save-every-n 10 --max-frames 600

Notes
- Undistortion parameters (fisheye defaults):
  - fx=fy≈min(w,h)/π, cx=w/2, cy=h/2, k1=-0.25 (k2=k3=k4=0)
  - These are safe defaults for 180° fisheye-like inputs and will be refined.
- Redistortion is currently a placeholder (identity). Images in 40_redist and after are visually same as 30_ai; the true inverse mapping will be implemented next.
- If ffprobe is available in PATH, basic metadata is recorded in logs. Otherwise, pipeline still runs.

Next steps
1) Implement true redistortion maps (plane -> original projection).
2) Add equirect 180 plane option and v360/cuda integration where available.
3) Integrate AI restoration stage and ensure stereo consistency checks.
4) Add VR metadata writing to the final muxed video output.

