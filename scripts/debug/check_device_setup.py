"""Utility script to diagnose GPU availability and model placement.

Run with e.g.:

    python scripts/debug/check_device_setup.py --device cuda:0 \
        --nsfw-model model_weights/lada_nsfw_detection_model_v1.3.pt

The script prints torch/ultralytics diagnostics and attempts to
instantiate the NSFW detector on the requested device.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def fmt_bool(value: bool | None) -> str:
    if value is None:
        return "unknown"
    return "yes" if value else "no"


def section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main() -> int:
    parser = argparse.ArgumentParser("Check device configuration")
    parser.add_argument("--device", default="cuda:0", help="Device string to test (default: %(default)s)")
    parser.add_argument(
        "--nsfw-model",
        default="model_weights/lada_nsfw_detection_model_v1.3.pt",
        help="Path to NSFW detection model weights for a load test",
    )
    args = parser.parse_args()

    section("Torch diagnostics")
    try:
        import torch
    except Exception as exc:  # pragma: no cover - diagnostic script
        print("Failed to import torch:", exc)
        return 1

    print(json.dumps(
        {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            if torch.cuda.is_available() else [],
        },
        indent=2,
    ))

    section("Ultralytics diagnostics")
    try:
        from ultralytics import YOLO, __version__ as ultralytics_version
    except Exception as exc:  # pragma: no cover - diagnostic script
        print("Failed to import ultralytics:", exc)
        return 1

    print(json.dumps({"ultralytics_version": ultralytics_version}, indent=2))

    section("resolve_torch_device")
    try:
        from lada.lib.ultralytics_utils import resolve_torch_device
    except Exception as exc:  # pragma: no cover - diagnostic script
        print("Failed to import resolve_torch_device:", exc)
        return 1

    try:
        device_str, torch_device = resolve_torch_device(args.device, description="diagnostic")
        print(json.dumps({"requested": args.device, "resolved": device_str, "torch_device": str(torch_device)}, indent=2))
    except Exception as exc:  # pragma: no cover - diagnostic script
        print("resolve_torch_device raised:", exc)
        return 1

    section("NSFW detector instantiation")
    model_path = Path(args.nsfw_model)
    print(f"Using model weights: {model_path}")
    if not model_path.exists():
        print("Model weights file is missing; skipping detector test.")
        return 1

    try:
        model = YOLO(str(model_path))
    except Exception as exc:  # pragma: no cover - diagnostic script
        print("Failed to load YOLO model:", exc)
        return 1

    try:
        import queue
        from lada.lib.nsfw_scene_detector import FileProcessingOptions, NsfwDetector
    except Exception as exc:  # pragma: no cover - diagnostic script
        print("Failed to import NSFW detector helpers:", exc)
        return 1

    fpo = FileProcessingOptions(
        input_dir=str(Path(".")),
        output_dir=Path("."),
        start_index=0,
        stride_length=0,
        scene_min_length=1,
        scene_max_length=2,
        scene_max_memory=128,
        random_extend_masks=True,
        skip4k=False,
    )

    try:
        detector = NsfwDetector(
            nsfw_detection_model=model,
            device=args.device,
            file_queue=queue.Queue(),
            frame_queue=queue.Queue(),
            scene_queue=queue.Queue(),
            file_processing_options=fpo,
        )
    except Exception as exc:  # pragma: no cover - diagnostic script
        print("Failed to instantiate NsfwDetector:", exc)
        return 1

    print(json.dumps(
        {
            "detector_device": detector.device,
            "use_half_precision": getattr(detector, "_use_half_precision", "unknown"),
        },
        indent=2,
    ))

    section("All good")
    print("Device setup appears to be working.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
