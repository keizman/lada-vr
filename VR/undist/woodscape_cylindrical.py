from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except ImportError as exc:  # pragma: no cover - handled in pipeline init
    cv2 = None  # type: ignore

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    yaml = None  # type: ignore

from VR.undist.base import Undistorter, UndistortionResult


def _ensure_cv2() -> None:
    if cv2 is None:
        raise RuntimeError("OpenCV is required for WoodScape cylindrical undistortion")


def _load_structured(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        if yaml is None:
            raise ValueError(f"Failed to parse WoodScape calibration {path}. Install PyYAML or provide JSON.")
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"WoodScape calibration {path} must decode to a dictionary")
    return data


def _normalize_quaternion(q: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    arr = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(arr)
    if norm == 0.0:
        return 1.0, 0.0, 0.0, 0.0
    arr = arr / norm
    return float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])


def _quaternion_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    """Convert a quaternion in (x, y, z, w) format to a rotation matrix."""
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def load_woodscape_calibration(path: str) -> Dict[str, Any]:
    calibration_path = Path(path)
    if not calibration_path.exists():
        raise FileNotFoundError(f"WoodScape calibration not found: {path}")
    data = _load_structured(calibration_path)
    if "intrinsic" not in data:
        raise ValueError(f"WoodScape calibration {path} missing 'intrinsic' section")
    intrinsic = data["intrinsic"]
    required = ["width", "height", "cx_offset", "cy_offset", "aspect_ratio"]
    for key in required:
        if key not in intrinsic:
            raise ValueError(f"WoodScape calibration {path} intrinsic missing '{key}'")
    for key in ("k1", "k2", "k3", "k4"):
        intrinsic.setdefault(key, 0.0)
    data.setdefault("extrinsic", {})
    return data


class WoodscapeCylindricalUndistorter(Undistorter):
    """Implements the cylindrical remap described in Elad Plaut's WoodScape tutorial."""

    def __init__(
        self,
        calibration: Dict[str, Any],
        *,
        hfov_deg: float = 190.0,
        vfov_deg: float = 143.0,
    ) -> None:
        _ensure_cv2()
        self._calib = calibration
        self._hfov = float(hfov_deg)
        self._vfov = float(vfov_deg)
        self._map_cache: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._output_size: Optional[Tuple[int, int]] = None

    def _build_maps(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._map_cache is not None:
            return self._map_cache
        intrinsic = self._calib["intrinsic"]
        hfov = np.deg2rad(self._hfov)
        vfov = np.deg2rad(self._vfov)

        quaternion = self._calib.get("extrinsic", {}).get("quaternion", [0.0, 0.0, 0.0, 1.0])
        if len(quaternion) != 4:
            quaternion = [0.0, 0.0, 0.0, 1.0]
        x, y, z, w = _normalize_quaternion(tuple(quaternion))
        R = _quaternion_to_matrix(x, y, z, w).T
        rdf_to_flu = np.array(
            [
                [0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=np.float64,
        )
        R = R @ rdf_to_flu
        denom = np.sqrt(R[0, 2] ** 2 + R[2, 2] ** 2)
        denom = max(denom, 1e-9)
        azimuth = np.arccos(np.clip(R[2, 2] / denom, -1.0, 1.0))
        if R[0, 2] < 0:
            azimuth = 2 * np.pi - azimuth
        tilt = -np.arccos(np.clip(np.sqrt(R[0, 2] ** 2 + R[2, 2] ** 2), -1.0, 1.0))
        Ry = np.array(
            [
                [np.cos(azimuth), 0.0, np.sin(azimuth)],
                [0.0, 1.0, 0.0],
                [-np.sin(azimuth), 0.0, np.cos(azimuth)],
            ],
            dtype=np.float64,
        ).T
        R = R @ Ry

        f = float(intrinsic.get("k1", 1.0))
        if f <= 0:
            raise ValueError("WoodScape calibration must provide positive k1 as effective focal length")
        out_h = int(max(2, round(2.0 * f * np.tan(vfov / 2.0))))
        out_w = int(max(2, round(f * hfov)))
        self._output_size = (out_w, out_h)
        K = np.array(
            [
                [f, 0.0, out_w / 2.0],
                [0.0, f, f * np.tan(vfov / 2.0 + tilt)],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        K_inv = np.linalg.inv(K)

        xv, yv = np.meshgrid(np.arange(out_w, dtype=np.float64), np.arange(out_h, dtype=np.float64), indexing="xy")
        ones = np.ones_like(xv)
        pixels = np.stack([xv, yv, ones], axis=-1).reshape(-1, 3).T  # 3 x (H*W)
        rays = (K_inv @ pixels)
        rays = rays / rays[2:3, :]
        rays = rays.reshape(3, out_h, out_w).transpose(1, 2, 0)

        cart = np.empty_like(rays)
        cart[..., 0] = np.sin(rays[..., 0])
        cart[..., 1] = rays[..., 1]
        cart[..., 2] = np.cos(rays[..., 0])
        cart = cart.reshape(-1, 3).T
        cart = (R @ cart).reshape(3, out_h, out_w).transpose(1, 2, 0)

        norms = np.linalg.norm(cart, axis=2, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        theta = np.arccos(np.clip(cart[..., [2]] / norms, -1.0, 1.0))

        c_X = float(intrinsic.get("cx_offset", 0.0)) + intrinsic.get("width", 0.0) / 2.0 - 0.5
        c_Y = float(intrinsic.get("cy_offset", 0.0)) + intrinsic.get("height", 0.0) / 2.0 - 0.5
        aspect_ratio = float(intrinsic.get("aspect_ratio", 1.0))
        k1 = float(intrinsic.get("k1", 0.0))
        k2 = float(intrinsic.get("k2", 0.0))
        k3 = float(intrinsic.get("k3", 0.0))
        k4 = float(intrinsic.get("k4", 0.0))

        rho = k1 * theta + k2 * theta ** 2 + k3 * theta ** 3 + k4 * theta ** 4
        chi = np.linalg.norm(cart[..., :2], axis=2, keepdims=True)
        chi = np.where(chi == 0.0, 1.0, chi)
        u = rho * cart[..., [0]] / chi
        v = rho * cart[..., [1]] / chi
        mapx = u[..., 0] + c_X
        mapy = v[..., 0] * aspect_ratio + c_Y
        self._map_cache = (mapx.astype(np.float32), mapy.astype(np.float32))
        return self._map_cache

    def undistort(self, img: np.ndarray, params: Optional[Any] = None) -> UndistortionResult:
        mapx, mapy = self._build_maps()
        rectified = cv2.remap(img, mapx, mapy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        info = {
            "mode": "woodscape_cylindrical",
            "hfov_deg": self._hfov,
            "vfov_deg": self._vfov,
            "output_size": self._output_size,
        }
        return UndistortionResult(image=rectified, new_camera_matrix=None, info=info)

    def redistort_stub(self, img: np.ndarray) -> np.ndarray:
        return img.copy()
