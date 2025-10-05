from dataclasses import dataclass
from typing import Optional, Literal, Dict, Any
import os
import time

try:
    import cv2  # type: ignore
    _CV2_OK = True
except Exception:
    _CV2_OK = False

import numpy as np

from VR.utils.jsonl_logger import JsonlLogger
from VR.metadata.ffprobe_wrapper import FFprobe
from VR.metadata.parser import detect_from_ffprobe
from VR.utils.imageops import default_fisheye_params, Undistortor
from VR.utils.projection import equirect_to_perspective, equirect_to_perspective_tiles

Layout = Literal["auto", "sbs", "ou", "mono"]


def _detect_layout(width: int, height: int, override: Layout = "auto") -> Literal["sbs", "ou", "mono"]:
    if override != "auto":
        return override  # type: ignore
    ratio = width / float(height) if height > 0 else 0.0
    inv_ratio = height / float(width) if width > 0 else 0.0
    if ratio >= 1.6 and (width % 2 == 0):
        return "sbs"
    if inv_ratio >= 1.6 and (height % 2 == 0):
        return "ou"
    return "mono"


@dataclass
class PipelineConfig:
    input_path: str
    out_dir: str
    layout: Layout = "auto"
    projection: str = "auto"  # auto | fisheye180 | equirect180 | equirect360
    plane_mode: str = "auto"   # auto | persp_single | persp_tiles | equirect_plane
    fov_deg: float = 110.0
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    tiles_x: int = 4
    tiles_y: int = 2
    save_every_n: int = 10  # save stage images every N frames
    max_frames: Optional[int] = None


class VRPipeline:
    def __init__(self, cfg: PipelineConfig, logger: JsonlLogger) -> None:
        if not _CV2_OK:
            raise RuntimeError("OpenCV (cv2) is required. Please `pip install opencv-python`." )
        self.cfg = cfg
        self.log = logger
        # Prepare dirs per stage
        self.stage_dirs = {
            "00_input": os.path.join(cfg.out_dir, "stages", "00_input"),
            "10_split_left": os.path.join(cfg.out_dir, "stages", "10_split", "left"),
            "10_split_right": os.path.join(cfg.out_dir, "stages", "10_split", "right"),
            "20_undist_left": os.path.join(cfg.out_dir, "stages", "20_undist", "left"),
            "20_undist_right": os.path.join(cfg.out_dir, "stages", "20_undist", "right"),
            "30_ai_left": os.path.join(cfg.out_dir, "stages", "30_ai", "left"),
            "30_ai_right": os.path.join(cfg.out_dir, "stages", "30_ai", "right"),
            "40_redist_left": os.path.join(cfg.out_dir, "stages", "40_redist", "left"),
            "40_redist_right": os.path.join(cfg.out_dir, "stages", "40_redist", "right"),
            "50_recompose_sbs": os.path.join(cfg.out_dir, "stages", "50_recompose_sbs"),
        }
        for d in self.stage_dirs.values():
            os.makedirs(d, exist_ok=True)
        os.makedirs(os.path.join(cfg.out_dir, "logs"), exist_ok=True)
        self._undist = Undistortor()

    def _save_img(self, path: str, img: np.ndarray) -> None:
        cv2.imwrite(path, img)

    def run(self) -> Dict[str, Any]:
        # Stage: metadata
        meta = FFprobe.probe(self.cfg.input_path)
        self.log.info("meta", ffprobe_available=FFprobe.is_available(), meta_present=meta is not None)

        cap = cv2.VideoCapture(self.cfg.input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.cfg.input_path}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        # Initial heuristics
        layout = _detect_layout(width, height, self.cfg.layout)
        proj = None
        # Metadata-driven override
        if meta:
            p_proj, p_layout, pose = detect_from_ffprobe(meta, width, height)
            if self.cfg.layout == "auto" and p_layout in ("sbs", "ou", "mono"):
                layout = p_layout  # override
            proj = p_proj
            # If user didn't set yaw/pitch/roll, seed them from metadata
            if self.cfg.yaw_deg == 0.0 and self.cfg.pitch_deg == 0.0 and self.cfg.roll_deg == 0.0:
                self.cfg.yaw_deg = float(pose.get("yaw", 0.0))
                self.cfg.pitch_deg = float(pose.get("pitch", 0.0))
                self.cfg.roll_deg = float(pose.get("roll", 0.0))
        self.log.info("pipeline:start", width=width, height=height, fps=fps, frames_total=frames_total, layout=layout)

        # Determine projection mode
        if self.cfg.projection == "auto":
            if not proj:
                # Heuristic: full-frame aspect ~2:1 => likely equirect; otherwise fisheye180
                proj = "equirect360" if abs((width / float(height)) - 2.0) < 0.15 else "fisheye180"
        else:
            proj = self.cfg.projection
        self.log.info("projection:mode", projection=proj)

        # Per-eye dimensions
        if layout == "sbs":
            per_w, per_h = width // 2, height
        elif layout == "ou":
            per_w, per_h = width, height // 2
        else:
            per_w, per_h = width, height

        # Prepare undistortion/projection params
        fish_params = default_fisheye_params(per_w, per_h)
        self.log.info("undistort:params", fx=fish_params.fx, fy=fish_params.fy, cx=fish_params.cx, cy=fish_params.cy, k1=fish_params.k1)

        frames_processed = 0
        frame_idx = 0
        start = time.time()
        while True:
            if self.cfg.max_frames is not None and frames_processed >= self.cfg.max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            save_this = (self.cfg.save_every_n > 0) and (frame_idx % self.cfg.save_every_n == 0)
            pts_ms = int((frame_idx / fps) * 1000) if fps > 0 else None

            # Stage 00: input snapshot (for sampled frames)
            if save_this:
                in_path = os.path.join(self.stage_dirs["00_input"], f"frame_{frame_idx:06d}.jpg")
                self._save_img(in_path, frame)

            # Stage 10: split
            if layout == "sbs":
                half = frame.shape[1] // 2
                left = frame[:, :half]
                right = frame[:, half:]
            elif layout == "ou":
                half = frame.shape[0] // 2
                left = frame[:half, :]
                right = frame[half:, :]
            else:
                left = frame
                right = None

            if save_this:
                self._save_img(os.path.join(self.stage_dirs["10_split_left"], f"frame_{frame_idx:06d}.jpg"), left)
                if right is not None:
                    self._save_img(os.path.join(self.stage_dirs["10_split_right"], f"frame_{frame_idx:06d}.jpg"), right)

            # Stage 20: project to processing plane
            und_left, und_right = None, None
            plane_mode = self.cfg.plane_mode
            if plane_mode == "auto":
                if isinstance(proj, str) and proj.startswith("equirect"):
                    plane_mode = "persp_tiles"  # default for 360/180
                else:
                    plane_mode = "persp_single"

            if isinstance(proj, str) and proj.startswith("equirect"):
                coverage = "180" if proj == "equirect180" else "360"
                if plane_mode == "equirect_plane":
                    und_left = left
                    und_right = right if right is not None else None
                    self.log.info("project:frame", frame_idx=frame_idx, mode=f"{proj}:equirect_plane")
                elif plane_mode == "persp_single":
                    und_left = equirect_to_perspective(left, per_w, per_h, self.cfg.fov_deg, self.cfg.yaw_deg, self.cfg.pitch_deg, self.cfg.roll_deg)
                    if right is not None:
                        und_right = equirect_to_perspective(right, per_w, per_h, self.cfg.fov_deg, self.cfg.yaw_deg, self.cfg.pitch_deg, self.cfg.roll_deg)
                    self.log.info("project:frame", frame_idx=frame_idx, mode=f"{proj}:persp_single", fov=self.cfg.fov_deg, yaw=self.cfg.yaw_deg, pitch=self.cfg.pitch_deg, roll=self.cfg.roll_deg)
                else:  # persp_tiles
                    # Generate tiles and save mosaics; also save per-tile images
                    tiles_l, centers_l, mosaic_l = equirect_to_perspective_tiles(left, self.cfg.tiles_x, self.cfg.tiles_y, self.cfg.fov_deg, coverage=coverage)
                    und_left = mosaic_l
                    if right is not None:
                        tiles_r, centers_r, mosaic_r = equirect_to_perspective_tiles(right, self.cfg.tiles_x, self.cfg.tiles_y, self.cfg.fov_deg, coverage=coverage)
                        und_right = mosaic_r
                    # Save per-tile images under subdirectories when sampled
                    if save_this:
                        base_left = os.path.join(self.stage_dirs["20_undist_left"], f"frame_{frame_idx:06d}")
                        os.makedirs(base_left, exist_ok=True)
                        for r in range(len(tiles_l)):
                            for c in range(len(tiles_l[r])):
                                tile_path = os.path.join(base_left, f"tile_{r}_{c}.jpg")
                                self._save_img(tile_path, tiles_l[r][c])
                        if right is not None:
                            base_right = os.path.join(self.stage_dirs["20_undist_right"], f"frame_{frame_idx:06d}")
                            os.makedirs(base_right, exist_ok=True)
                            for r in range(len(tiles_r)):
                                for c in range(len(tiles_r[r])):
                                    tile_path = os.path.join(base_right, f"tile_{r}_{c}.jpg")
                                    self._save_img(tile_path, tiles_r[r][c])
                    self.log.info("project:frame", frame_idx=frame_idx, mode=f"{proj}:persp_tiles", tiles_x=self.cfg.tiles_x, tiles_y=self.cfg.tiles_y, fov=self.cfg.fov_deg)
            else:
                # Fisheye -> rectified plane (as before, with safety)
                und_left, _, info_l = self._undist.undistort_with_info(left, fish_params)
                info_r = None
                if right is not None:
                    und_right, _, info_r = self._undist.undistort_with_info(right, fish_params)
                self.log.info("undistort:frame", frame_idx=frame_idx,
                              left_valid_ratio=info_l.get("valid_ratio", None), left_fallback=info_l.get("fallback", None),
                              right_valid_ratio=(info_r.get("valid_ratio") if info_r else None), right_fallback=(info_r.get("fallback") if info_r else None))

            if save_this:
                self._save_img(os.path.join(self.stage_dirs["20_undist_left"], f"frame_{frame_idx:06d}.jpg"), und_left)
                if und_right is not None:
                    self._save_img(os.path.join(self.stage_dirs["20_undist_right"], f"frame_{frame_idx:06d}.jpg"), und_right)

            # Stage 30: AI (pilot = passthrough)
            ai_left = und_left
            ai_right = und_right
            if save_this:
                self._save_img(os.path.join(self.stage_dirs["30_ai_left"], f"frame_{frame_idx:06d}.jpg"), ai_left)
                if ai_right is not None:
                    self._save_img(os.path.join(self.stage_dirs["30_ai_right"], f"frame_{frame_idx:06d}.jpg"), ai_right)

            # Stage 40: redistort back (pilot stub = identity)
            red_left = self._undist.redistort_stub(ai_left)
            red_right = self._undist.redistort_stub(ai_right) if ai_right is not None else None
            if save_this:
                self._save_img(os.path.join(self.stage_dirs["40_redist_left"], f"frame_{frame_idx:06d}.jpg"), red_left)
                if red_right is not None:
                    self._save_img(os.path.join(self.stage_dirs["40_redist_right"], f"frame_{frame_idx:06d}.jpg"), red_right)

            # Stage 50: recompose SBS (use existing layout, assume two eyes)
            if red_right is None:
                recomposed = red_left
            else:
                if layout == "sbs":
                    recomposed = np.concatenate([red_left, red_right], axis=1)
                elif layout == "ou":
                    recomposed = np.concatenate([red_left, red_right], axis=0)
                else:
                    recomposed = red_left
            if save_this:
                self._save_img(os.path.join(self.stage_dirs["50_recompose_sbs"], f"frame_{frame_idx:06d}.jpg"), recomposed)

            # Log per-frame summary (sampled detailed)
            self.log.info("frame",
                          frame_idx=frame_idx, pts_ms=pts_ms,
                          in_w=int(frame.shape[1]), in_h=int(frame.shape[0]), layout=layout,
                          left_w=int(left.shape[1]), left_h=int(left.shape[0]),
                          right_w=int(right.shape[1]) if right is not None else None,
                          right_h=int(right.shape[0]) if right is not None else None,
                          saved=save_this)

            frames_processed += 1
            frame_idx += 1

        cap.release()
        elapsed = time.time() - start
        self.log.info("pipeline:done", frames_processed=frames_processed, seconds=elapsed)
        return {"frames_processed": frames_processed, "seconds": elapsed, "layout": layout, "fps": fps, "width": width, "height": height}

