from __future__ import annotations
import hashlib
from typing import Dict, Tuple, Optional
import numpy as np

class LutCache:
    """
    Simple in-memory cache for OpenCV remap maps keyed by (w,h,params_hash).
    """
    def __init__(self) -> None:
        self._store: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    @staticmethod
    def key(w: int, h: int, params: Dict[str, float]) -> str:
        m = hashlib.sha256()
        m.update(f"{w}x{h}".encode())
        for k in sorted(params.keys()):
            m.update(f"|{k}={params[k]}".encode())
        return m.hexdigest()[:16]

    def get(self, key: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        return self._store.get(key)

    def put(self, key: str, maps: Tuple[np.ndarray, np.ndarray]) -> None:
        self._store[key] = maps

