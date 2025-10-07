from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except Exception as exc:  # pragma: no cover
    cv2 = None  # type: ignore

from VR.undist.opencv_calibrated import CalibrationData


@dataclass
class AutoCalibrationResult:
    calibration: CalibrationData
    score: float
    segments_used: int
    frames_sampled: int


def _detect_layout(width: int, height: int) -> str:
    ratio = width / float(height) if height else 0.0
    inv = height / float(width) if width else 0.0
    if ratio >= 1.6 and (width % 2 == 0):
        return "sbs"
    if inv >= 1.6 and (height % 2 == 0):
        return "ou"
    return "mono"


def _camera_matrix(f: float, w: int, h: int) -> np.ndarray:
    cx = w / 2.0
    cy = h / 2.0
    return np.array([[f, 0.0, cx],
                     [0.0, f, cy],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def _segment_points(line: np.ndarray, points: int) -> np.ndarray:
    x0, y0, x1, y1 = line
    xs = np.linspace(x0, x1, points, dtype=np.float64)
    ys = np.linspace(y0, y1, points, dtype=np.float64)
    return np.stack([xs, ys], axis=1)


def _line_cost(points: np.ndarray) -> float:
    if points.shape[0] < 3:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    var_x = np.var(x)
    var_y = np.var(y)
    if var_x < 1e-6 and var_y < 1e-6:
        return 0.0
    if var_x >= var_y:
        A = np.vstack([x, np.ones_like(x)]).T
        sol, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        m, b = sol
        denom = math.sqrt(m * m + 1.0)
        residual = np.abs(y - (m * x + b)) / denom
    else:
        A = np.vstack([y, np.ones_like(y)]).T
        sol, _, _, _ = np.linalg.lstsq(A, x, rcond=None)
        m, b = sol
        denom = math.sqrt(m * m + 1.0)
        residual = np.abs(x - (m * y + b)) / denom
    return float(np.mean(residual))


def _valid_ratio(map1: np.ndarray, map2: np.ndarray, w: int, h: int) -> float:
    valid_x = (map1 >= 0.0) & (map1 <= (w - 1))
    valid_y = (map2 >= 0.0) & (map2 <= (h - 1))
    valid = valid_x & valid_y
    return float(np.count_nonzero(valid)) / float(map1.size)


def _evaluate_cost(segments: List[np.ndarray], K: np.ndarray, D: np.ndarray) -> float:
    if not segments:
        return float("inf")
    total = 0.0
    count = 0
    for seg in segments:
        pts = seg.reshape(-1, 1, 2).astype(np.float64)
        und = cv2.fisheye.undistortPoints(pts, K, D, P=K)
        und_pts = und.reshape(-1, 2)
        cost = _line_cost(und_pts)
        if not np.isfinite(cost):
            continue
        total += cost
        count += 1
    if count == 0:
        return float("inf")
    return total / count


def _collect_segments(frames: Iterable[np.ndarray], max_segments: int = 200) -> Tuple[List[np.ndarray], Tuple[int, int]]:
    if cv2 is None:
        raise RuntimeError("OpenCV is required for auto_lines calibration")
    lsd = getattr(cv2, "createLineSegmentDetector", None)
    if lsd is None:
        raise RuntimeError("cv2.createLineSegmentDetector is unavailable; install opencv-contrib-python")
    refine = getattr(cv2, "LSD_REFINE_ADV", 1)
    detector = lsd(refine) if callable(lsd) else lsd
    segments: List[np.ndarray] = []
    shape: Tuple[int, int] = (0, 0)
    for frame in frames:
        if frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lines, _, _, _ = detector.detect(gray)
        if lines is None:
            continue
        h, w = gray.shape
        shape = (h, w)
        for line in lines:
            x0, y0, x1, y1 = line[0]
            length = math.hypot(x1 - x0, y1 - y0)
            if length < min(h, w) * 0.15:
                continue
            pts = _segment_points(np.array([x0, y0, x1, y1], dtype=np.float64), points=32)
            segments.append(pts)
            if len(segments) >= max_segments:
                return segments, (h, w)
    return segments, shape


def _sample_frames(video_path: str, sample_count: int) -> Tuple[List[np.ndarray], int, Tuple[int, int]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video for auto calibration: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if total <= 0:
        total = sample_count
    indices = np.linspace(0, max(total - 1, 0), sample_count, dtype=int)
    frames: List[np.ndarray] = []
    idx_iter = iter(sorted(set(int(i) for i in indices)))
    next_idx = next(idx_iter, None)
    current = 0
    layout = _detect_layout(width, height)
    while next_idx is not None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, next_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        if layout == "sbs":
            frame = frame[:, :frame.shape[1] // 2]
        elif layout == "ou":
            frame = frame[:frame.shape[0] // 2, :]
        frames.append(frame)
        current += 1
        next_idx = next(idx_iter, None)
    cap.release()
    if not frames:
        raise RuntimeError("auto_lines calibration could not sample frames from video")
    h, w = frames[0].shape[:2]
    return frames, current, (h, w)


def estimate_calibration(video_path: str,
                         sample_frames: int = 12,
                         max_segments: int = 200) -> AutoCalibrationResult:
    frames, frames_sampled, eye_shape = _sample_frames(video_path, sample_frames)
    segments, (h, w) = _collect_segments(frames, max_segments=max_segments)
    if not segments:
        raise RuntimeError("auto_lines calibration failed: no reliable line segments detected")
    base_f = min(w, h) / math.pi
    K_base = _camera_matrix(base_f, w, h)

    k1_candidates = np.linspace(-0.60, -0.05, 20)
    scale_candidates = np.linspace(0.85, 1.35, 15)
    best_score = float("inf")
    best_params: Optional[Tuple[float, float]] = None
    best_valid = 0.0
    for scale in scale_candidates:
        f = base_f * scale
        K = _camera_matrix(f, w, h)
        for k1 in k1_candidates:
            D_vec = np.array([k1, 0.0, 0.0, 0.0], dtype=np.float64).reshape(4, 1)
            new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D_vec, (w, h), np.eye(3), balance=1.0)
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D_vec, np.eye(3), new_K, (w, h), cv2.CV_32FC1)
            valid_ratio = _valid_ratio(map1, map2, w, h)
            if valid_ratio < 0.35:
                continue
            score = _evaluate_cost(segments, K, D_vec)
            if not np.isfinite(score):
                continue
            if score < best_score or (math.isclose(score, best_score, rel_tol=1e-3) and valid_ratio > best_valid):
                best_score = score
                best_params = (scale, k1)
                best_valid = valid_ratio
    if best_params is None:
        raise RuntimeError("auto_lines calibration failed to converge on parameters")
    scale, k1 = best_params
    final_f = base_f * scale
    K = _camera_matrix(final_f, w, h)
    D = np.array([k1, 0.0, 0.0, 0.0], dtype=np.float64).reshape(4, 1)
    calibration = CalibrationData(K=K,
                                  D=D,
                                  reference_size=(w, h),
                                  source=f"auto_lines:{Path(video_path).name}")
    return AutoCalibrationResult(calibration=calibration,
                                 score=float(best_score),
                                 segments_used=len(segments),
                                 frames_sampled=frames_sampled)


def ensure_cached_calibration(video_path: str,
                              cache_dir: Optional[str] = None) -> Tuple[CalibrationData, Path, Dict[str, float]]:
    repo_root = Path(__file__).resolve().parents[2]
    cache_root = Path(cache_dir) if cache_dir else repo_root / "tmp" / "auto_calibration"
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_file = cache_root / f"{Path(video_path).stem}_auto_lines.json"
    if cache_file.exists():
        data = json.loads(cache_file.read_text(encoding='utf-8'))
        K = np.asarray(data['K'], dtype=np.float64)
        D = np.asarray(data['D'], dtype=np.float64).reshape(4, 1)
        ref = tuple(int(v) for v in data['reference_size'])  # type: ignore
        calib = CalibrationData(K=K, D=D, reference_size=ref, source=str(cache_file))
        metadata = {
            'score': float(data.get('score', 0.0)),
            'segments_used': int(data.get('segments_used', 0)),
            'frames_sampled': int(data.get('frames_sampled', 0)),
            'cache_hit': 1.0,
        }
        return calib, cache_file, metadata
    result = estimate_calibration(video_path)
    calib = result.calibration
    payload = {
        'mode': 'auto_lines',
        'video': os.path.abspath(video_path),
        'K': calib.K.tolist(),
        'D': calib.D.reshape(-1).tolist(),
        'reference_size': list(calib.reference_size),
        'score': result.score,
        'segments_used': result.segments_used,
        'frames_sampled': result.frames_sampled,
    }
    cache_file.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    metadata = {
        'score': result.score,
        'segments_used': float(result.segments_used),
        'frames_sampled': float(result.frames_sampled),
        'cache_hit': 0.0,
    }
    return calib, cache_file, metadata


