import argparse
import concurrent.futures as concurrent_futures
from fractions import Fraction
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from ultralytics import YOLO

from lada.lib import mask_utils
from lada.lib.audio_utils import combine_audio_video_files
from lada.lib.mosaic_utils import addmosaic_base, get_random_parameter
from lada.lib.ultralytics_utils import convert_yolo_mask
from lada.lib.video_utils import VideoWriter, get_video_meta_data, is_video_file


def parse_args():
    parser = argparse.ArgumentParser("Apply mosaic censorship to clean videos")
    parser.add_argument("--input", type=Path, required=True, help="Video file or directory containing videos to censor")
    parser.add_argument("--output-root", type=Path, default=Path("mosaic_videos"),
                        help="Directory where censored videos will be written")
    parser.add_argument("--model", type=str, default="model_weights/lada_nsfw_detection_model_v1.3.pt",
                        help="Path to NSFW detection segmentation model")
    parser.add_argument("--model-device", type=str, default="cuda:0",
                        help="Device used for the NSFW detection model (e.g. 'cpu', 'cuda', 'cuda:0')")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size for YOLO")
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml", help="Tracker config used by YOLO track")
    parser.add_argument("--codec", type=str, default="h264", help="Encoder codec (passed to av/ffmpeg)")
    parser.add_argument("--crf", type=int, default=None, help="Encoder CRF / QP value")
    parser.add_argument("--preset", type=str, default=None, help="Encoder preset")
    parser.add_argument("--moov-front", default=False, action=argparse.BooleanOptionalAction,
                        help="Enable faststart by moving moov atom to front")
    parser.add_argument("--custom-encoder-options", type=str, default=None,
                        help="Additional encoder options passed verbatim (e.g. '-b:v 5M -maxrate 8M')")
    parser.add_argument("--output-suffix", type=str, default=".mosaic",
                        help="Suffix appended before file extension in the output file name")
    parser.add_argument("--overwrite", default=False, action=argparse.BooleanOptionalAction,
                        help="Overwrite existing output videos")
    parser.add_argument("--parallel-file-count", type=int, default=1,
                        help="Number of videos to process concurrently")
    parser.add_argument("--randomize-mosaic-size", default=True, action=argparse.BooleanOptionalAction,
                        help="Randomise mosaic block sizes per detection")
    parser.add_argument("--fixed-mosaic-size", type=int, default=None,
                        help="Use a fixed mosaic block size (overrides random sizing if provided)")
    parser.add_argument("--mask-dilate-iterations", type=int, default=1,
                        help="Number of dilation iterations applied to the detection mask")
    parser.add_argument("--mask-dilate-size", type=int, default=11,
                        help="Kernel size used for mask dilation")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Only process at most this many files (useful for smoke tests)")
    parser.add_argument("--skip-existing", default=True, action=argparse.BooleanOptionalAction,
                        help="Skip processing if the output file already exists")
    return parser.parse_args()


def iter_video_paths(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    video_files: list[Path] = []
    for path in sorted(input_path.iterdir()):
        if path.is_file() and is_video_file(str(path)):
            video_files.append(path)
    return video_files


def build_output_path(output_root: Path, input_file: Path, suffix: str) -> Path:
    return output_root.joinpath(f"{input_file.stem}{suffix}{input_file.suffix}")


def compute_frame_pts(frame_index: int, start_pts: int, fps: Fraction, time_base: Fraction) -> int:
    fps_float = float(fps)
    time_base_float = float(time_base)
    if fps_float <= 0 or time_base_float <= 0:
        return start_pts + frame_index
    timestamp_seconds = frame_index / fps_float
    pts_offset = int(round(timestamp_seconds / time_base_float))
    return start_pts + pts_offset


def derive_mosaic_parameters(mask: np.ndarray, randomize: bool, fixed_size: int | None) -> tuple[int, str, float, int]:
    if fixed_size is not None:
        size = max(3, fixed_size)
        return size, 'squa_avg', 1.6, -1
    if randomize:
        return get_random_parameter(mask, randomize_size=True)
    base_size, mod, rect_ratio, feather = get_random_parameter(mask, randomize_size=False)
    return base_size, mod, rect_ratio, feather


def apply_mosaic_to_results(frame: np.ndarray,
                            results,
                            randomize_size: bool,
                            fixed_size: int | None,
                            dilate_size: int,
                            dilate_iterations: int) -> np.ndarray:
    if results is None or results.masks is None or results.boxes is None:
        return frame

    masks = results.masks
    num_masks = len(masks)
    if num_masks == 0:
        return frame

    mosaic_frame = frame
    for mask_idx in range(num_masks):
        try:
            yolo_mask = masks[mask_idx]
        except IndexError:
            continue
        mask_img = convert_yolo_mask(yolo_mask, mosaic_frame.shape)
        if mask_img is None or mask_img.sum() == 0:
            continue
        mask_img = mask_utils.fill_holes(mask_img)
        if dilate_iterations > 0:
            mask_img = mask_utils.dilate_mask(mask_img, dilatation_size=dilate_size, iterations=dilate_iterations)

        mosaic_size, mod, rect_ratio, feather = derive_mosaic_parameters(mask_img, randomize_size, fixed_size)
        mosaic_size = max(3, mosaic_size)
        mosaic_frame, _ = addmosaic_base(mosaic_frame, mask_img, mosaic_size,
                                         model=mod, rect_ratio=rect_ratio, feather=feather)
    return mosaic_frame


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def process_single_video(args,
                         model: YOLO,
                         video_path: Path,
                         output_path: Path,
                         tmp_output_path: Path,
                         prefix: str):
    metadata = get_video_meta_data(str(video_path))
    ensure_dir(output_path.parent)

    fps_exact = metadata.video_fps_exact
    time_base = metadata.time_base
    start_pts = metadata.start_pts or 0

    width = metadata.video_width
    height = metadata.video_height

    use_half = str(args.model_device).startswith("cuda") and torch.cuda.is_available()

    stream_kwargs = dict(
        source=str(video_path),
        stream=True,
        verbose=False,
        device=args.model_device,
        imgsz=args.imgsz,
        conf=args.conf,
        tracker=args.tracker,
        half=use_half,
    )

    frame_index = 0
    with VideoWriter(str(tmp_output_path), width, height, fps_exact, codec=args.codec,
                     crf=args.crf, preset=args.preset, time_base=time_base,
                     moov_front=args.moov_front, custom_encoder_options=args.custom_encoder_options) as video_writer:
        for results in model.track(**stream_kwargs):
            frame = results.orig_img
            if frame is None:
                continue
            mosaic_frame = apply_mosaic_to_results(frame.copy(),
                                                   results,
                                                   randomize_size=args.randomize_mosaic_size,
                                                   fixed_size=args.fixed_mosaic_size,
                                                   dilate_size=args.mask_dilate_size,
                                                   dilate_iterations=args.mask_dilate_iterations)
            frame_pts = compute_frame_pts(frame_index, start_pts, fps_exact, time_base)
            video_writer.write(mosaic_frame, frame_pts=frame_pts, bgr2rgb=True)
            frame_index += 1

    combine_audio_video_files(metadata, str(tmp_output_path), str(output_path))
    print(f"{prefix} Wrote {output_path.name} ({frame_index} frames)")


def process_video_subset(args,
                         video_entries: Sequence[tuple[int, Path]],
                         pipeline_idx: int):
    prefix = f"[P{pipeline_idx}]"
    if not video_entries:
        print(f"{prefix} No files assigned; skipping")
        return

    print(f"{prefix} Initialising model on {args.model_device}")
    model = YOLO(args.model)

    for global_index, video_path in video_entries:
        print(f"{prefix} Processing {global_index}, {video_path.name}")
        output_path = build_output_path(args.output_root, video_path, args.output_suffix)
        tmp_output_path = output_path.with_suffix(f".tmp{output_path.suffix}")

        if output_path.exists():
            if args.skip_existing and not args.overwrite:
                print(f"{prefix} Skipping existing output: {output_path.name}")
                continue
            if args.overwrite:
                output_path.unlink()

        try:
            process_single_video(args, model, video_path, output_path, tmp_output_path, prefix)
        except Exception as exc:  # pragma: no cover - diagnostic/logging path
            print(f"{prefix} Error processing {video_path.name}: {exc}")
            if tmp_output_path.exists():
                try:
                    tmp_output_path.unlink()
                except OSError:
                    pass


def chunk_video_entries(indexed_files: Sequence[tuple[int, Path]], chunk_count: int) -> list[list[tuple[int, Path]]]:
    chunks: list[list[tuple[int, Path]]] = [[] for _ in range(chunk_count)]
    for idx, entry in enumerate(indexed_files):
        target = idx % chunk_count
        chunks[target].append(entry)
    return [chunk for chunk in chunks if chunk]


def main():
    args = parse_args()

    ensure_dir(args.output_root)

    video_files = iter_video_paths(args.input)
    if args.max_files is not None:
        video_files = video_files[:args.max_files]

    indexed_video_files = list(enumerate(video_files))
    if not indexed_video_files:
        print("No input videos found. Nothing to do.")
        return

    parallel_count = max(1, args.parallel_file_count)
    parallel_count = min(parallel_count, len(indexed_video_files))

    chunks = chunk_video_entries(indexed_video_files, parallel_count)
    print(f"Found {len(indexed_video_files)} video(s). Running {len(chunks)} pipeline(s) in parallel.")
    for idx, chunk in enumerate(chunks):
        file_list = ', '.join(path.name for _, path in chunk[:3])
        if len(chunk) > 3:
            file_list += ", …"
        print(f"[P{idx}] Assigned {len(chunk)} file(s): {file_list}")

    if len(chunks) == 1:
        process_video_subset(args, chunks[0], 0)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(chunks)) as executor:
            futures = [executor.submit(process_video_subset, args, chunk, idx)
                       for idx, chunk in enumerate(chunks)]
            for future in concurrent.futures.as_completed(futures):
                future.result()


if __name__ == "__main__":
    main()


'''
  python scripts/mosaic_creation/apply-mosaic-to-videos.py \
      --input "/mnt/k/tmp/v/t/clean_videos" \
      --output-root "/mnt/k/tmp/v/t/clean_videos_mosaic" \
      --model-device cuda:0 \
      --parallel-file-count 2 \
      --codec h264_nvenc --crf 23

'''