from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol

import numpy as np


@dataclass
class UndistortionResult:
    """Container for undistortion outputs returned by every strategy."""
    image: np.ndarray
    new_camera_matrix: Optional[np.ndarray] = None
    info: Dict[str, Any] = field(default_factory=dict)


class Undistorter(Protocol):
    """Protocol implemented by undistortion strategies."""

    def undistort(self, img: np.ndarray, params: Optional[Any] = None) -> UndistortionResult:
        ...

    def redistort_stub(self, img: np.ndarray) -> np.ndarray:
        ...
