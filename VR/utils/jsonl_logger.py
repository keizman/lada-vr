import json
import os
import time
from typing import Optional, Dict, Any

class JsonlLogger:
    """
    Simple JSON Lines logger for structured, machine-parsable logs.
    Each log entry is a single line JSON object.
    """

    def __init__(self, log_path: str) -> None:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self._log_path = log_path
        # Open in append mode to allow multiple runs to write to the same file.
        self._fh = open(self._log_path, "a", encoding="utf-8")

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass

    def log(self, entry: Dict[str, Any], *, ts: Optional[float] = None) -> None:
        if ts is None:
            ts = time.time()
        enriched = {"ts": ts, **entry}
        self._fh.write(json.dumps(enriched, ensure_ascii=False) + "\n")
        self._fh.flush()

    def info(self, message: str, **kwargs: Any) -> None:
        self.log({"level": "INFO", "msg": message, **kwargs})

    def debug(self, message: str, **kwargs: Any) -> None:
        self.log({"level": "DEBUG", "msg": message, **kwargs})

    def error(self, message: str, **kwargs: Any) -> None:
        self.log({"level": "ERROR", "msg": message, **kwargs})

