import argparse
import os
import sys
from typing import Any, Dict, List, Tuple

import cv2  # type: ignore

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from VR.undist import build_undistorter  # type: ignore


def _parse_resolution(value: str) -> Tuple[int, int]:
    cleaned = value.lower().replace(" ", "")
    if "x" not in cleaned:
        raise argparse.ArgumentTypeError("Resolution must be formatted as WIDTHxHEIGHT")
    w_str, h_str = cleaned.split("x", 1)
    try:
        width = int(w_str)
        height = int(h_str)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Resolution components must be integers") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("Resolution components must be positive")
    return width, height


def _parse_variant(spec: str) -> Tuple[str, Dict[str, float]]:
    label = ""
    params: Dict[str, float] = {}
    tokens = [token.strip() for token in spec.split(",") if token.strip()]
    if not tokens:
        raise argparse.ArgumentTypeError("WoodScape variant spec cannot be empty")
    for token in tokens:
        if "=" not in token:
            raise argparse.ArgumentTypeError(
                "Each WoodScape variant item must be key=value (e.g. hfov=185)"
            )
        key, value = token.split("=", 1)
        key = key.strip().lower()
        value = value.strip()
        if key in {"label", "name"}:
            if not value:
                raise argparse.ArgumentTypeError("Variant label cannot be empty")
            label = value
        elif key in {"hfov", "hfov_deg"}:
            params["hfov_deg"] = float(value)
        elif key in {"vfov", "vfov_deg"}:
            params["vfov_deg"] = float(value)
        else:
            raise argparse.ArgumentTypeError(
                f"Unsupported WoodScape variant key '{key}'. Use hfov/vfov/label."
            )
    if not params:
        raise argparse.ArgumentTypeError("WoodScape variant must set at least hfov or vfov")
    return label, params


def _slugify(value: str) -> str:
    if not value:
        return "woodscape_cylindrical"
    safe = [c if c.isalnum() or c in ("-", "_") else "_" for c in value.strip()]
    slug = "".join(safe)
    return slug or "woodscape_cylindrical"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate undistorted variants of a single fisheye frame using multiple strategies"
    )
    p.add_argument("--input", required=True, help="Input fisheye image path")
    p.add_argument("--out-dir", required=True, help="Directory where undistorted variants will be saved")
    p.add_argument("--calib-charuco", help="Calibration JSON/YAML for Charuco-based OpenCV fisheye")
    p.add_argument("--calib-chessboard", help="Calibration JSON/YAML for chessboard-based OpenCV fisheye")
    p.add_argument("--calib-woodscape", help="WoodScape-format calibration JSON/YAML for cylindrical mapping")
    p.add_argument("--charuco-balance", type=float, default=None, help="Balance factor for Charuco OpenCV fisheye (default 1.0)")
    p.add_argument("--chessboard-balance", type=float, default=None, help="Balance factor for chessboard OpenCV fisheye (default 1.0)")
    p.add_argument("--charuco-reference", type=_parse_resolution, default=None, help="Reference resolution for Charuco calibration scaling (e.g. 3840x3840)")
    p.add_argument("--chessboard-reference", type=_parse_resolution, default=None, help="Reference resolution for chessboard calibration scaling")
    p.add_argument("--woodscape-hfov", type=float, default=None, help="Horizontal FOV for WoodScape cylindrical mode (default 190)")
    p.add_argument("--woodscape-vfov", type=float, default=None, help="Vertical FOV for WoodScape cylindrical mode (default 143)")
    p.add_argument(
        "--woodscape-variant",
        action="append",
        default=None,
        help="Additional WoodScape variants, e.g. --woodscape-variant hfov=185,vfov=130,label=soft",
    )
    p.add_argument("--prefix", default="undist", help="Filename prefix for saved outputs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    img = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Failed to read image: {args.input}")

    os.makedirs(args.out_dir, exist_ok=True)

    strategies: List[Tuple[str, Any]] = []
    strategies.append(("opencv_default", build_undistorter("opencv_default", params={})))

    if args.calib_charuco:
        params = {"calibration_path": args.calib_charuco}
        if args.charuco_balance is not None:
            params["balance"] = args.charuco_balance
        if args.charuco_reference is not None:
            params["reference_size"] = args.charuco_reference
        strategies.append(("opencv_charuco", build_undistorter("opencv_charuco", params=params)))

    if args.calib_chessboard:
        params = {"calibration_path": args.calib_chessboard}
        if args.chessboard_balance is not None:
            params["balance"] = args.chessboard_balance
        if args.chessboard_reference is not None:
            params["reference_size"] = args.chessboard_reference
        strategies.append(("opencv_chessboard", build_undistorter("opencv_chessboard", params=params)))

    if args.calib_woodscape:
        base_params: Dict[str, float] = {"calibration_path": args.calib_woodscape}
        if args.woodscape_hfov is not None:
            base_params["hfov_deg"] = args.woodscape_hfov
        if args.woodscape_vfov is not None:
            base_params["vfov_deg"] = args.woodscape_vfov

        woodscape_sets: List[Tuple[str, Dict[str, float]]] = []
        woodscape_sets.append(("woodscape_cylindrical", dict(base_params)))

        if args.woodscape_variant:
            for idx, spec in enumerate(args.woodscape_variant, start=1):
                label_override, overrides = _parse_variant(spec)
                params = dict(base_params)
                params.update(overrides)
                label = label_override or f"woodscape_cylindrical_v{idx}"
                label = _slugify(label)
                woodscape_sets.append((label, params))

        for label, params in woodscape_sets:
            strategies.append((label, build_undistorter("woodscape_cylindrical", params=params)))

    if not strategies:
        raise SystemExit(
            "No undistortion strategies configured. Provide at least one calibration or rely on default."
        )

    for name, strategy in strategies:
        result = strategy.undistort(img)
        out_name = f"{args.prefix}_{_slugify(name)}.png"
        out_path = os.path.join(args.out_dir, out_name)
        if not cv2.imwrite(out_path, result.image):
            raise SystemExit(f"Failed to write output: {out_path}")
        info = result.info or {}
        print(f"Saved {name} -> {out_path} (mode={info.get('mode', name)})")


if __name__ == "__main__":
    main()


