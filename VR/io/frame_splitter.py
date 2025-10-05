from dataclasses import dataclass
from typing import Optional, Tuple, Literal, Dict, Any
import os
import time

try:
    import cv2  # type: ignore
    _CV2_OK = True
except Exception:
    _CV2_OK = False

from VR.utils.jsonl_logger import JsonlLogger

Layout = Literal["auto", "sbs", "ou", "mono"]


def _detect_layout(width: int, height: int, override: Layout = "auto") -> Literal["sbs", "ou", "mono"]:
    """
    Infer frame layout (side-by-side or over-under) if not explicitly provided.
    Updated heuristic (more permissive for common VR aspect ratios like 4096x2160):
      - If width/height >= 1.6 and width is even -> SBS
      - Else if height/width >= 1.6 and height is even -> Over-Under (OU)
      - Else mono
    """
    if override != "auto":
        return override  # type: ignore
    ratio = width / float(height) if height > 0 else 0.0
    inv_ratio = height / float(width) if width > 0 else 0.0
    if ratio >= 1.6 and (width % 2 == 0):
        return "sbs"  # side-by-side
    if inv_ratio >= 1.6 and (height % 2 == 0):
        return "ou"   # over-under
    return "mono"


@dataclass
class SplitConfig:
    input_path: str
    out_dir: str
    layout: Layout = "auto"
    thumb_interval: int = 10  # save debug thumbs every N frames (0 to disable)
    thumb_max_w: int = 480    # resize long edge to this width for thumbs
    max_frames: Optional[int] = None  # limit for pilot runs


class FrameSplitter:
    def __init__(self, cfg: SplitConfig, logger: JsonlLogger) -> None:
        self.cfg = cfg
        self.log = logger
        if not _CV2_OK:
            raise RuntimeError(
                "OpenCV (cv2) is required for FrameSplitter. Please `pip install opencv-python`.")

        os.makedirs(os.path.join(cfg.out_dir, "thumbs", "left"), exist_ok=True)
        os.makedirs(os.path.join(cfg.out_dir, "thumbs", "right"), exist_ok=True)
        self._thumb_left_dir = os.path.join(cfg.out_dir, "thumbs", "left")
        self._thumb_right_dir = os.path.join(cfg.out_dir, "thumbs", "right")

    def _save_thumb(self, img, path: str) -> None:
        h, w = img.shape[:2]
        if w > self.cfg.thumb_max_w:
            scale = self.cfg.thumb_max_w / float(w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        cv2.imwrite(path, img)

    def process(self) -> Dict[str, Any]:
        start_ts = time.time()
        cap = cv2.VideoCapture(self.cfg.input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.cfg.input_path}")

        # Try to read basic metadata
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        layout = _detect_layout(width, height, self.cfg.layout)

        self.log.info("split:start",
                      input_path=self.cfg.input_path,
                      width=width,
                      height=height,
                      fps=fps,
                      frame_count=frame_count,
                      layout=layout)

        frames_processed = 0
        split_w = width // 2 if layout == "sbs" else width
        split_h = height // 2 if layout == "ou" else height

        frame_idx = 0
        while True:
            if self.cfg.max_frames is not None and frames_processed >= self.cfg.max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break

            # Approximate PTS using fps if available
            pts_ms = None
            if fps > 0:
                pts_ms = int((frame_idx / fps) * 1000)

            # Split
            if layout == "sbs":
                half = frame.shape[1] // 2
                left = frame[:, :half]
                right = frame[:, half:]
            elif layout == "ou":
                half = frame.shape[0] // 2
                left = frame[:half, :]
                right = frame[half:, :]
            else:  # mono
                left = frame
                right = None

            # Save thumbs if requested
            left_thumb_path = right_thumb_path = None
            if self.cfg.thumb_interval > 0 and (frame_idx % self.cfg.thumb_interval == 0):
                left_thumb_path = os.path.join(self._thumb_left_dir, f"frame_{frame_idx:06d}.jpg")
                self._save_thumb(left, left_thumb_path)
                if right is not None:
                    right_thumb_path = os.path.join(self._thumb_right_dir, f"frame_{frame_idx:06d}.jpg")
                    self._save_thumb(right, right_thumb_path)

            # Log per-frame (sampled: every Nth frame detailed, others brief)
            log_level = "DEBUG" if (self.cfg.thumb_interval > 0 and frame_idx % self.cfg.thumb_interval == 0) else "INFO"
            entry = {
                "stage": "split",
                "frame_idx": frame_idx,
                "pts_ms": pts_ms,
                "in_w": int(frame.shape[1]),
                "in_h": int(frame.shape[0]),
                "layout": layout,
                "out_left_w": int(left.shape[1]),
                "out_left_h": int(left.shape[0]),
                "out_right_w": int(right.shape[1]) if right is not None else None,
                "out_right_h": int(right.shape[0]) if right is not None else None,
                "thumb_left": left_thumb_path,
                "thumb_right": right_thumb_path,
            }
            if log_level == "DEBUG":
                self.log.debug("frame", **entry)
            else:
                self.log.info("frame", **entry)

            frames_processed += 1
            frame_idx += 1

        cap.release()
        elapsed = time.time() - start_ts
        self.log.info("split:done", frames_processed=frames_processed, seconds=elapsed)
        return {
            "frames_processed": frames_processed,
            "seconds": elapsed,
            "width": width,
            "height": height,
            "fps": fps,
            "layout": layout,
            "split_w": split_w,
            "split_h": split_h,
        }

