import math
from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np

try:
    import cv2  # type: ignore
except Exception as e:
    cv2 = None

from VR.maps.lut_cache import LutCache

@dataclass
class FisheyeParams:
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float = -0.25
    k2: float = 0.0
    k3: float = 0.0
    k4: float = 0.0

    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx],
                         [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]], dtype=np.float64)

    def D(self) -> np.ndarray:
        return np.array([self.k1, self.k2, self.k3, self.k4], dtype=np.float64)


def default_fisheye_params(w: int, h: int) -> FisheyeParams:
    # Equidistant default: f ≈ min(w,h)/pi
    f = min(w, h) / math.pi
    return FisheyeParams(fx=f, fy=f, cx=w / 2.0, cy=h / 2.0)


def hanning2d(w: int, h: int, border: int) -> np.ndarray:
    """
    Create a 2D feathering mask (0..1) with a flat 1 region and Hanning falloff near edges.
    border: number of pixels for the falloff on each side.
    """
    border = max(1, int(border))
    wx = np.ones((w,), dtype=np.float32)
    if border * 2 >= w:
        wx = np.hanning(max(2, w)).astype(np.float32)
    else:
        edge = np.hanning(border * 2)
        wx[:border] = edge[:border]
        wx[-border:] = edge[-border:]
    wy = np.ones((h,), dtype=np.float32)
    if border * 2 >= h:
        wy = np.hanning(max(2, h)).astype(np.float32)
    else:
        edge = np.hanning(border * 2)
        wy[:border] = edge[:border]
        wy[-border:] = edge[-border:]
    mask = np.outer(wy, wx)
    return mask


class Undistortor:
    def __init__(self) -> None:
        self._cache = LutCache()

    def _identity_maps(self, w: int, h: int) -> Tuple[np.ndarray, np.ndarray]:
        x = np.tile(np.arange(w, dtype=np.float32), (h, 1))
        y = np.tile(np.arange(h, dtype=np.float32).reshape(h, 1), (1, w))
        return x, y

    def maps_for_rectified(self, w: int, h: int, params: FisheyeParams, *, balance: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build maps to transform distorted fisheye -> rectified plane using OpenCV fisheye model.
        Returns (map_x, map_y, new_K)
        """
        assert cv2 is not None, "OpenCV is required"
        K = params.K()
        D = params.D()
        # Use balance=1.0 to preserve FOV and reduce out-of-bounds likelihood for pilot.
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (w, h), np.eye(3), balance=float(balance))
        key = self._cache.key(w, h, {"fx": params.fx, "fy": params.fy, "cx": params.cx, "cy": params.cy, "k1": params.k1, "bal": float(balance)})
        cached = self._cache.get(key)
        if cached is None:
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, (w, h), cv2.CV_32FC1)
            self._cache.put(key, (map1, map2))
        else:
            map1, map2 = cached
        return map1, map2, new_K

    def _fold_ratio(self, map_arr: np.ndarray, axis: int) -> float:
        diffs = np.diff(map_arr, axis=axis)
        if diffs.size == 0:
            return 0.0
        finite = np.isfinite(diffs)
        if not finite.any():
            return 0.0
        diffs = diffs[finite]
        bad = diffs < -1e-3
        return float(bad.sum()) / float(diffs.size)

    def _valid_ratio(self, map1: np.ndarray, map2: np.ndarray, w: int, h: int) -> float:
        # Count how many mapped coords fall inside the source image
        valid_x = (map1 >= 0.0) & (map1 <= (w - 1))
        valid_y = (map2 >= 0.0) & (map2 <= (h - 1))
        valid = valid_x & valid_y
        return float(np.count_nonzero(valid)) / float(map1.size)

    def undistort_with_info(self, img: np.ndarray, params: FisheyeParams) -> Tuple[np.ndarray, np.ndarray, dict]:
        h, w = img.shape[:2]
        map1, map2, new_K = self.maps_for_rectified(w, h, params, balance=1.0)
        vr = self._valid_ratio(map1, map2, w, h)
        fold_x = self._fold_ratio(map1, axis=1)
        fold_y = self._fold_ratio(map2, axis=0)
        info = {"valid_ratio": vr, "fold_ratio_x": fold_x, "fold_ratio_y": fold_y, "fallback": None}
        fallback_reason = None
        if vr < 0.2:
            fallback_reason = "oob"
        elif fold_x > 0.1 or fold_y > 0.1:
            fallback_reason = "fold"
        if fallback_reason is not None:
            # Too many samples out of bounds or the map folds back on itself; fallback to identity remap
            imap1, imap2 = self._identity_maps(w, h)
            rectified = cv2.remap(img, imap1, imap2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            info["fallback"] = fallback_reason
            return rectified, new_K, info
        rectified = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return rectified, new_K, info

    def undistort(self, img: np.ndarray, params: FisheyeParams) -> Tuple[np.ndarray, np.ndarray]:
        rectified, new_K, _ = self.undistort_with_info(img, params)
        return rectified, new_K

    def redistort_stub(self, rectified: np.ndarray) -> np.ndarray:
        """
        Placeholder: currently identity (no-op). A true inverse remap will be implemented next.
        """
        return rectified.copy()

