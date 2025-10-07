# Helpers to read metadata using ffprobe
import json
import shutil
import subprocess
from typing import Any, Dict, Optional

class FFprobe:
    @staticmethod
    def is_available() -> bool:
        return shutil.which("ffprobe") is not None

    @staticmethod
    def probe(path: str, *, timeout: int = 30) -> Optional[Dict[str, Any]]:
        if not FFprobe.is_available():
            return None
        cmd = [
            "ffprobe", "-v", "error",
            "-print_format", "json",
            "-show_streams", "-show_format",
            path,
        ]
        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=False)
            if proc.returncode != 0:
                return None
            return json.loads(proc.stdout.decode("utf-8", errors="ignore"))
        except Exception:
            return None

