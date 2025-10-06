from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple

from VR.undist.base import Undistorter
from VR.undist.opencv_default import OpenCVDefaultUndistorter
from VR.undist.opencv_calibrated import OpenCVCalibratedFisheyeUndistorter, CalibrationData, load_calibration
from VR.undist.woodscape_cylindrical import WoodscapeCylindricalUndistorter, load_woodscape_calibration
from VR.undist.auto_lines import ensure_cached_calibration


def _parse_reference_size(params: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    ref = params.get("reference_size")
    if isinstance(ref, (list, tuple)) and len(ref) >= 2:
        return int(ref[0]), int(ref[1])
    if isinstance(ref, str):
        lowered = ref.lower().replace(" ", "")
        if "x" in lowered:
            w_str, h_str = lowered.split("x", 1)
            try:
                return int(w_str), int(h_str)
            except ValueError:
                pass
    width = params.get("reference_width")
    height = params.get("reference_height")
    if isinstance(width, (int, float)) and isinstance(height, (int, float)):
        return int(width), int(height)
    return None


def build_undistorter(mode: str, *, params: Optional[Dict[str, Any]] = None) -> Undistorter:
    """Factory that returns an undistortion strategy based on configuration."""
    params = dict(params or {})
    mode_lower = mode.lower()
    if mode_lower in ("opencv_default", "default"):
        return OpenCVDefaultUndistorter()

    if mode_lower in ("opencv_calibrated", "opencv_charuco", "opencv_chessboard"):
        calib_path = params.get("calibration_path")
        if not calib_path:
            raise ValueError(f"undist_mode '{mode}' requires 'calibration_path' in undist_params")
        balance = float(params.get("balance", 1.0))
        ref_size = _parse_reference_size(params)
        calibration: CalibrationData = load_calibration(calib_path, fallback_size=ref_size)
        label = {
            "opencv_charuco": "opencv_charuco",
            "opencv_chessboard": "opencv_chessboard",
        }.get(mode_lower, "opencv_calibrated")
        return OpenCVCalibratedFisheyeUndistorter(calibration, balance=balance, mode_label=label)

    if mode_lower in ("woodscape", "woodscape_cylindrical"):
        calib_path = params.get("calibration_path")
        if not calib_path:
            raise ValueError(f"undist_mode '{mode}' requires 'calibration_path' in undist_params")
        hfov = float(params.get("hfov_deg", 190.0))
        vfov = float(params.get("vfov_deg", 143.0))
        calibration = load_woodscape_calibration(calib_path)
        return WoodscapeCylindricalUndistorter(calibration, hfov_deg=hfov, vfov_deg=vfov)

    if mode_lower in ("auto_lines", "autolines"):
        video_path = params.get("input_path")
        if not video_path:
            raise ValueError("undist_mode 'auto_lines' requires 'input_path' in undist_params")
        cache_dir = params.get("cache_dir")
        calibration, cache_file, meta = ensure_cached_calibration(video_path, cache_dir=cache_dir)
        balance = float(params.get("balance", 1.0))
        undistorter = OpenCVCalibratedFisheyeUndistorter(calibration, balance=balance, mode_label="auto_lines")
        params.setdefault("_auto_meta", meta)
        params.setdefault("_auto_cache", str(cache_file))
        return undistorter

    raise ValueError(f"Unknown undistortion mode '{mode}'.")


def list_undistort_modes() -> Iterable[str]:
    return (
        "opencv_default",
        "opencv_calibrated",
        "opencv_charuco",
        "opencv_chessboard",
        "woodscape_cylindrical",
        "auto_lines",
    )


__all__ = ["build_undistorter", "list_undistort_modes"]
