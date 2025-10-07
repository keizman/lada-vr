from __future__ import annotations

from typing import Optional

import numpy as np

from VR.utils.imageops import FisheyeParams, Undistortor as CoreUndistortor, default_fisheye_params
from VR.undist.base import Undistorter, UndistortionResult


class OpenCVDefaultUndistorter(Undistorter):
    """Wrapper around the existing VR.utils.imageops.Undistortor."""

    def __init__(self) -> None:
        self._core = CoreUndistortor()

    def undistort(self, img: np.ndarray, params: Optional[FisheyeParams] = None) -> UndistortionResult:
        if params is None:
            h, w = img.shape[:2]
            params = default_fisheye_params(w, h)
        rectified, new_K, info = self._core.undistort_with_info(img, params)
        info = dict(info)
        info.setdefault("mode", "opencv_default")
        return UndistortionResult(image=rectified, new_camera_matrix=new_K, info=info)

    def redistort_stub(self, img: np.ndarray) -> np.ndarray:
        return self._core.redistort_stub(img)
