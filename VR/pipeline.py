from dataclasses import dataclass, field
from typing import Optional, Literal, Dict, Any, List, Tuple
import os
import time
import re

try:
    import cv2  # type: ignore
    _CV2_OK = True
except Exception:
    _CV2_OK = False

import numpy as np

from VR.utils.jsonl_logger import JsonlLogger
from VR.metadata.ffprobe_wrapper import FFprobe
from VR.metadata.parser import detect_from_ffprobe
from VR.utils.imageops import default_fisheye_params, hanning2d
from VR.utils.projection import equirect_to_perspective, equirect_to_perspective_tiles, build_perspective_from_equirect_patch_map
from VR.utils.roi_stitch import build_roi_grid3x3_combined, composite_back_from_combined
from VR.undist import build_undistorter

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
    plane_mode: str = "auto"   # auto | persp_single | persp_tiles | equirect_plane | roi_persp
    fov_deg: float = 110.0
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    tiles_x: int = 4
    tiles_y: int = 2
    save_every_n: int = 10  # save stage images every N frames
    max_frames: Optional[int] = None
    undist_mode: str = "opencv_default"
    undist_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VariantSpec:
    label: str
    mode: str
    params: Dict[str, Any]


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
            "20_roi_left": os.path.join(cfg.out_dir, "stages", "20_roi", "left"),
            "20_roi_right": os.path.join(cfg.out_dir, "stages", "20_roi", "right"),
            "30_ai_left": os.path.join(cfg.out_dir, "stages", "30_ai", "left"),
            "30_ai_right": os.path.join(cfg.out_dir, "stages", "30_ai", "right"),
            "30_ai_roi_left": os.path.join(cfg.out_dir, "stages", "30_ai_roi", "left"),
            "30_ai_roi_right": os.path.join(cfg.out_dir, "stages", "30_ai_roi", "right"),
            "40_redist_left": os.path.join(cfg.out_dir, "stages", "40_redist", "left"),
            "40_redist_right": os.path.join(cfg.out_dir, "stages", "40_redist", "right"),
            "50_recompose_sbs": os.path.join(cfg.out_dir, "stages", "50_recompose_sbs"),
        }
        for d in self.stage_dirs.values():
            os.makedirs(d, exist_ok=True)
        os.makedirs(os.path.join(cfg.out_dir, "logs"), exist_ok=True)
        self._repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        undist_params = dict(cfg.undist_params)
        if cfg.undist_mode == "auto_lines":
            undist_params.setdefault("input_path", cfg.input_path)
        self._undist_label = self._slug_label(cfg.undist_mode)
        try:
            self._undist = build_undistorter(cfg.undist_mode, params=undist_params)
        except Exception as exc:
            self.log.info("undistort:build_failed", mode=cfg.undist_mode, error=str(exc))
            undist_params = {}
            self._undist_label = self._slug_label("fallback_default")
            self._undist = build_undistorter("opencv_default", params={})
        cfg.undist_params = undist_params
        if cfg.undist_mode == "auto_lines":
            meta = undist_params.get("_auto_meta") if isinstance(undist_params, dict) else None
            cache_path = undist_params.get("_auto_cache") if isinstance(undist_params, dict) else None
            if meta:
                self.log.info("undistort:auto_lines", cache=cache_path, score=meta.get("score"), segments=meta.get("segments_used"), frames=meta.get("frames_sampled"), cache_hit=meta.get("cache_hit"))
        self._variant_unavailable: Dict[str, str] = {}
        self._extra_variants: List[Tuple[str, Any]] = []
        self._extra_variant_dirs: Dict[str, Dict[str, str]] = {}
        self._prepare_extra_undistorters()

    def _save_img(self, path: str, img: np.ndarray) -> None:
        cv2.imwrite(path, img)

    @staticmethod
    def _slug_label(value: str) -> str:
        slug = re.sub(r'[^a-z0-9]+', '_', value.lower()).strip('_')
        return slug or 'variant'

    def _locate_calibration_file(self, filenames: List[str], search_roots: List[str]) -> Optional[str]:
        for fname in filenames:
            for rel_root in search_roots:
                base = os.path.join(self._repo_root, rel_root)
                if not os.path.isdir(base):
                    continue
                for root, _dirs, files in os.walk(base):
                    if fname in files:
                        return os.path.join(root, fname)
        return None

    def _discover_variant_specs(self) -> List[VariantSpec]:
        self._variant_unavailable = {}
        specs: List[VariantSpec] = []

        charuco_path = self._locate_calibration_file(
            ['fisheye_calibration.json'],
            ['tmp/Fisheye_ChArUco_Calibration', 'configs']
        )
        if charuco_path:
            specs.append(VariantSpec(label='charuco', mode='opencv_charuco', params={'calibration_path': charuco_path}))
        else:
            self._variant_unavailable['charuco'] = 'calibration_not_found'

        chessboard_path = self._locate_calibration_file(
            ['fisheye_calibration_data.json'],
            ['tmp/opencv-fisheye-undistortion', 'configs']
        )
        if chessboard_path:
            specs.append(VariantSpec(label='chessboard', mode='opencv_chessboard', params={'calibration_path': chessboard_path}))
        else:
            self._variant_unavailable['chessboard'] = 'calibration_not_found'

        woodscape_path = self._locate_calibration_file(
            ['woodscape_calibration.json', 'woodscape.json', 'woodscape_calib.json'],
            ['tmp', 'configs']
        )
        if woodscape_path:
            specs.append(VariantSpec(label='woodscape', mode='woodscape_cylindrical', params={'calibration_path': woodscape_path}))
        else:
            self._variant_unavailable['woodscape'] = 'calibration_not_found'

        return specs

    def _prepare_extra_undistorters(self) -> None:
        specs = self._discover_variant_specs()
        if self._variant_unavailable:
            for label, reason in self._variant_unavailable.items():
                self.log.info('undistort:variant_unavailable', label=label, reason=reason)

        if not specs:
            self.log.info('undistort:variant_none', fallback_mode=self.cfg.undist_mode, note='edges remain warped without calibration')

        for spec in specs:
            label_slug = self._slug_label(spec.label)
            if label_slug == self._undist_label:
                label_slug = f"{label_slug}_alt"
            try:
                variant = build_undistorter(spec.mode, params=spec.params)
            except Exception as exc:  # pragma: no cover - optional calibrations
                self.log.info('undistort:variant_skip', label=label_slug, reason=str(exc))
                continue
            self._extra_variants.append((label_slug, variant))
            left_dir = os.path.join(self.cfg.out_dir, 'stages', f'20_undist_{label_slug}', 'left')
            right_dir = os.path.join(self.cfg.out_dir, 'stages', f'20_undist_{label_slug}', 'right')
            os.makedirs(left_dir, exist_ok=True)
            os.makedirs(right_dir, exist_ok=True)
            self._extra_variant_dirs[label_slug] = {'left': left_dir, 'right': right_dir}
            self.log.info('undistort:variant_enabled', label=label_slug, mode=spec.mode, calibration=spec.params.get('calibration_path'))

        if self._extra_variants:
            active = [label for label, _ in self._extra_variants]
            self.log.info('undistort:variants_ready', labels=active)

    def _generate_extra_undist_outputs(self, frame_idx: int, left: np.ndarray, right: Optional[np.ndarray], params: Any) -> None:
        for label, variant in self._extra_variants:
            try:
                res_left = variant.undistort(left, params)
            except Exception as exc:  # pragma: no cover - optional calibrations
                self.log.info('undistort:variant_error', frame_idx=frame_idx, label=label, eye='left', error=str(exc))
                continue
            left_dir = self._extra_variant_dirs[label]['left']
            self._save_img(os.path.join(left_dir, f'frame_{frame_idx:06d}.jpg'), res_left.image)
            info_left = res_left.info or {}
            info_right: Dict[str, Any] = {}
            if right is not None:
                try:
                    res_right = variant.undistort(right, params)
                    right_dir = self._extra_variant_dirs[label]['right']
                    self._save_img(os.path.join(right_dir, f'frame_{frame_idx:06d}.jpg'), res_right.image)
                    info_right = res_right.info or {}
                except Exception as exc:  # pragma: no cover - optional calibrations
                    self.log.info('undistort:variant_error', frame_idx=frame_idx, label=label, eye='right', error=str(exc))
            self.log.info('undistort:variant',
                          frame_idx=frame_idx,
                          label=label,
                          left_mode=info_left.get('mode'),
                          left_valid_ratio=info_left.get('valid_ratio'),
                          right_mode=(info_right.get('mode') if info_right else None),
                          right_valid_ratio=(info_right.get('valid_ratio') if info_right else None))

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
        self.log.info("undistort:mode", mode=self.cfg.undist_mode, has_params=bool(self.cfg.undist_params))

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

            # ROI grid3x3 stitch mode (mid-lower)
            if isinstance(proj, str) and proj.startswith("equirect") and plane_mode == "roi_grid3x3":
                # Build combined small 2D from 3x3 parts
                comb_left = build_roi_grid3x3_combined(left)
                und_left = comb_left.image
                roi_grid_state_left = comb_left
                if right is not None:
                    comb_right = build_roi_grid3x3_combined(right)
                    und_right = comb_right.image
                    roi_grid_state_right = comb_right
                else:
                    und_right = None
                # Save
                if save_this:
                    self._save_img(os.path.join(self.stage_dirs["20_roi_left"], f"frame_{frame_idx:06d}_grid.jpg"), und_left)
                    if und_right is not None:
                        self._save_img(os.path.join(self.stage_dirs["20_roi_right"], f"frame_{frame_idx:06d}_grid.jpg"), und_right)
                self.log.info("project:roi_grid3x3", frame_idx=frame_idx, mode=f"{proj}:roi_grid3x3")
            # ROI perspective mode (legacy)
            elif isinstance(proj, str) and proj.startswith("equirect") and plane_mode == "roi_persp":
                # Define ROI rectangle from 3x3 grid: X in [W/6, 5W/6], Y in [H/2, 5H/6]
                def roi_rect(w: int, h: int) -> tuple[int, int, int, int]:
                    return int(w/6), int(h/2), int(5*w/6), int(5*h/6)

                # Helper to compute ROI views (single for 180, dual for 360)
                def roi_views_for(projection: str, w: int, h: int, left_img: np.ndarray):
                    x0, y0, x1, y1 = roi_rect(w, h)
                    cx = (x0 + x1) / 2.0
                    cy = (y0 + y1) / 2.0
                    # equirect pixel -> (lon,lat)
                    lon_center = (cx / float(w)) * 360.0 - 180.0
                    lat_center = (0.5 - (cy / float(h))) * 180.0
                    if projection == "equirect180":
                        yaws = [lon_center]
                    else:
                        yaws = [lon_center - 60.0, lon_center + 60.0]
                    fov_h = 120.0
                    out_w = min(1024, w)
                    out_h = max(256, out_w // 2)
                    views = []
                    for yaw in yaws:
                        img_roi = equirect_to_perspective(left_img, out_w, out_h, fov_h, yaw_deg=float(yaw), pitch_deg=float(lat_center), roll_deg=0.0)
                        alpha = hanning2d(out_w, out_h, border=max(8, out_w//16)).astype(np.float32)
                        # Precompute inverse maps for the patch
                        patch_w = x1 - x0
                        patch_h = y1 - y0
                        mx, my, valid = build_perspective_from_equirect_patch_map(w, h, out_w, out_h, fov_h, float(yaw), float(lat_center), 0.0, x0, y0, patch_w, patch_h)
                        views.append({"yaw": yaw, "pitch": lat_center, "fov_h": fov_h,
                                      "out_w": out_w, "out_h": out_h, "img": img_roi, "alpha": alpha,
                                      "patch": {"x0": x0, "y0": y0, "w": patch_w, "h": patch_h, "mx": mx, "my": my, "valid": valid}})
                    return views

                # Compute views for left/right
                views_left = roi_views_for(proj, per_w, per_h, left)
                views_right = roi_views_for(proj, per_w, per_h, right) if right is not None else None
                # Save small ROI images for inspection
                if save_this:
                    for idx_v, v in enumerate(views_left):
                        self._save_img(os.path.join(self.stage_dirs["20_roi_left"], f"frame_{frame_idx:06d}_v{idx_v}.jpg"), v["img"])
                    if views_right is not None:
                        for idx_v, v in enumerate(views_right):
                            self._save_img(os.path.join(self.stage_dirs["20_roi_right"], f"frame_{frame_idx:06d}_v{idx_v}.jpg"), v["img"])
                # For downstream, set und_left/right to mosaics of ROI (for stage continuity)
                und_left = views_left[0]["img"]
                und_right = views_right[0]["img"] if views_right is not None else None
                # Stash views for redist
                roi_state_left = views_left
                roi_state_right = views_right
                self.log.info("project:roi", frame_idx=frame_idx, mode=f"{proj}:roi_persp", views=len(views_left))
            elif isinstance(proj, str) and proj.startswith("equirect"):
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
                # Fisheye -> rectified plane (strategy-dependent safety)
                result_left = self._undist.undistort(left, fish_params)
                und_left = result_left.image
                info_l = result_left.info or {}
                result_right = None
                und_right = None
                if right is not None:
                    result_right = self._undist.undistort(right, fish_params)
                    und_right = result_right.image
                info_r = (result_right.info if result_right else None) or {}
                if not self._extra_variants and self.cfg.undist_mode == 'opencv_default':
                    self.log.info('undistort:warning', frame_idx=frame_idx, message='opencv_default output is known to warp edges; treat as invalid until calibrated variants succeed.')
                self.log.info("undistort:frame", frame_idx=frame_idx,
                              left_mode=info_l.get("mode"),
                              left_valid_ratio=info_l.get("valid_ratio"), left_fallback=info_l.get("fallback"),
                              left_fold_ratio_x=info_l.get("fold_ratio_x"), left_fold_ratio_y=info_l.get("fold_ratio_y"),
                              right_mode=(info_r.get("mode") if result_right else None),
                              right_valid_ratio=(info_r.get("valid_ratio") if result_right else None),
                              right_fallback=(info_r.get("fallback") if result_right else None),
                              right_fold_ratio_x=(info_r.get("fold_ratio_x") if result_right else None),
                              right_fold_ratio_y=(info_r.get("fold_ratio_y") if result_right else None))

                if save_this and self._extra_variants:
                    self._generate_extra_undist_outputs(frame_idx, left, right, fish_params)

            if save_this:
                self._save_img(os.path.join(self.stage_dirs["20_undist_left"], f"frame_{frame_idx:06d}.jpg"), und_left)
                if und_right is not None:
                    self._save_img(os.path.join(self.stage_dirs["20_undist_right"], f"frame_{frame_idx:06d}.jpg"), und_right)

            # Stage 30: AI (pilot = passthrough)
            ai_left = und_left
            ai_right = und_right
            # If ROI persp mode, save AI ROI images
            if 'roi_state_left' in locals():
                for idx_v, v in enumerate(roi_state_left):
                    ai_img = v["img"]  # passthrough
                    if save_this:
                        self._save_img(os.path.join(self.stage_dirs["30_ai_roi_left"], f"frame_{frame_idx:06d}_v{idx_v}.jpg"), ai_img)
                    v["ai_img"] = ai_img
                if roi_state_right is not None:
                    for idx_v, v in enumerate(roi_state_right):
                        ai_img = v["img"]
                        if save_this:
                            self._save_img(os.path.join(self.stage_dirs["30_ai_roi_right"], f"frame_{frame_idx:06d}_v{idx_v}.jpg"), ai_img)
                        v["ai_img"] = ai_img
            # Also keep old 30_ai stage for continuity
            if save_this:
                self._save_img(os.path.join(self.stage_dirs["30_ai_left"], f"frame_{frame_idx:06d}.jpg"), ai_left)
                if ai_right is not None:
                    self._save_img(os.path.join(self.stage_dirs["30_ai_right"], f"frame_{frame_idx:06d}.jpg"), ai_right)

            # Stage 40: redistort back
            if 'roi_grid_state_left' in locals():
                # Redist from combined small 2D back to full split image
                red_left = composite_back_from_combined(left, ai_left if ai_left is not None else roi_grid_state_left.image, roi_grid_state_left.maps, feather=12)
                if right is not None and 'roi_grid_state_right' in locals():
                    red_right = composite_back_from_combined(right, ai_right if ai_right is not None else roi_grid_state_right.image, roi_grid_state_right.maps, feather=12)
                else:
                    red_right = None
            elif 'roi_state_left' in locals():
                # Composite ROI views back to the split-sized images with feathered blending
                canvas_left = left.copy()
                for v in roi_state_left:
                    x0 = v["patch"]["x0"]; y0 = v["patch"]["y0"]; pw = v["patch"]["w"]; ph = v["patch"]["h"]
                    mx = v["patch"]["mx"]; my = v["patch"]["my"]
                    ai_img = v.get("ai_img", v["img"])
                    # Map AI ROI to patch
                    patch_img = cv2.remap(ai_img, mx, my, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                    # Map alpha to patch
                    alpha_roi = v["alpha"]
                    patch_alpha = cv2.remap(alpha_roi, mx, my, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                    patch_alpha = np.clip(patch_alpha, 0.0, 1.0).astype(np.float32)
                    # Blend into canvas
                    roi_region = canvas_left[y0:y0+ph, x0:x0+pw]
                    if roi_region.shape[:2] != patch_img.shape[:2]:
                        # Safety check
                        continue
                    if patch_img.dtype != np.uint8:
                        patch_img = patch_img.astype(np.uint8)
                    # Ensure float blending
                    roi_f = roi_region.astype(np.float32)
                    patch_f = patch_img.astype(np.float32)
                    alpha_f = patch_alpha[..., None]  # HxWx1
                    blended = (1.0 - alpha_f) * roi_f + alpha_f * patch_f
                    canvas_left[y0:y0+ph, x0:x0+pw] = np.clip(blended, 0, 255).astype(np.uint8)
                red_left = canvas_left
                if right is not None and roi_state_right is not None:
                    canvas_right = right.copy()
                    for v in roi_state_right:
                        x0 = v["patch"]["x0"]; y0 = v["patch"]["y0"]; pw = v["patch"]["w"]; ph = v["patch"]["h"]
                        mx = v["patch"]["mx"]; my = v["patch"]["my"]
                        ai_img = v.get("ai_img", v["img"])
                        patch_img = cv2.remap(ai_img, mx, my, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                        alpha_roi = v["alpha"]
                        patch_alpha = cv2.remap(alpha_roi, mx, my, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                        patch_alpha = np.clip(patch_alpha, 0.0, 1.0).astype(np.float32)
                        roi_region = canvas_right[y0:y0+ph, x0:x0+pw]
                        if roi_region.shape[:2] != patch_img.shape[:2]:
                            continue
                        if patch_img.dtype != np.uint8:
                            patch_img = patch_img.astype(np.uint8)
                        roi_f = roi_region.astype(np.float32)
                        patch_f = patch_img.astype(np.float32)
                        alpha_f = patch_alpha[..., None]
                        blended = (1.0 - alpha_f) * roi_f + alpha_f * patch_f
                        canvas_right[y0:y0+ph, x0:x0+pw] = np.clip(blended, 0, 255).astype(np.uint8)
                    red_right = canvas_right
                else:
                    red_right = None
            else:
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










