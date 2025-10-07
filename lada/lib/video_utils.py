import json
import os
import re
import subprocess
from contextlib import contextmanager
from fractions import Fraction
from typing import Callable
import unicodedata

import av
import cv2
import numpy as np

from lada.lib import Image, Mask, VideoMetadata


def read_video_frames(path: str, float32: bool = True, start_idx: int = 0, end_idx: int | None = None, normalize_neg1_pos1 = False, binary_frames=False) -> list[np.ndarray]:
    # Resolve cross-OS filename compatibility (Windows-escaped names, Unicode normalizations)
    path = resolve_path_compat(path)
    with VideoReaderOpenCV(path) as video_reader:
        frames = []
        i = 0
        while video_reader.isOpened():
            ret, frame = video_reader.read()
            if ret and (end_idx is None or i < end_idx):
                if binary_frames:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frame = np.expand_dims(frame, axis=-1)
                if i >= start_idx:
                    if float32:
                        if normalize_neg1_pos1:
                            frame = (frame.astype(np.float32) / 255.0 - 0.5) / 0.5
                        else:
                            frame = frame.astype(np.float32) / 255.
                    frames.append(frame)
                i += 1
            else:
                break
    return frames


def resize_video_frames(frames: list, size: int | tuple[int, int]):
    resized = []
    target_size = size if isinstance(size, (list, tuple)) else (size, size)
    for frame in frames:
        if frame.shape[:2] == target_size:
            resized.append(frame)
        else:
            resized.append(cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR))
    return resized


def pad_to_compatible_size_for_video_codecs(imgs):
    # dims need to be divisible by 2 by most codecs. given the chroma / pix format dims must be divisible by 4
    h, w = imgs[0].shape[:2]
    pad_h = 0 if h % 4 == 0 else 4 - (h % 4)
    pad_w = 0 if w % 4 == 0 else 4 - (w % 4)
    if pad_h == 0 and pad_w == 0:
        return imgs
    else:
        return [np.pad(img, ((0, pad_h), (0, pad_w), (0,0))).astype(np.uint8) for img in imgs]


# --------------------
# Cross-OS path compatibility helpers
# --------------------

def _hashu_decode(s: str) -> str:
    """Convert '#UXXXX' (hex codepoint) sequences to the corresponding Unicode chars."""
    return re.sub(r"#U([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), s)


def _hashu_encode(s: str) -> str:
    """Encode non-ASCII chars to '#UXXXX' (lowercase hex) sequences."""
    out = []
    for ch in s:
        code = ord(ch)
        if code < 128:
            out.append(ch)
        else:
            out.append(f"#U{code:04x}")
    return ''.join(out)


def resolve_path_compat(path: str) -> str:
    """Return a path that exists on this OS by trying common cross-platform variants.

    Tries in order:
    - original path
    - decode '#UXXXX' -> Unicode
    - encode Unicode -> '#UXXXX'
    - Unicode NFC/NFD normalizations
    Returns the first variant that exists; otherwise the original path.
    """
    try:
        # Normalize separators (just in case of stray Windows backslashes)
        path = path.replace('\\', os.sep)
        if os.path.exists(path):
            return path

        d = os.path.dirname(path)
        b = os.path.basename(path)
        candidates = []

        # '#Uxxxx' -> Unicode
        if '#U' in b:
            candidates.append(os.path.join(d, _hashu_decode(b)))

        # Unicode -> '#Uxxxx'
        if any(ord(c) > 127 for c in b):
            candidates.append(os.path.join(d, _hashu_encode(b)))

        # Unicode normalization variants
        try:
            candidates.append(os.path.join(d, unicodedata.normalize('NFC', b)))
            candidates.append(os.path.join(d, unicodedata.normalize('NFD', b)))
        except Exception:
            pass

        for cand in candidates:
            cand = cand.replace('\\', os.sep)
            if os.path.exists(cand):
                return cand
    except Exception:
        # If anything goes wrong, fall back to the original path
        pass
    return path


@contextmanager
def VideoReaderOpenCV(*args, **kwargs):
    cap = cv2.VideoCapture(*args, **kwargs)
    if not cap.isOpened():
        raise Exception(f"Unable to open video file:", *args)
    try:
        yield cap
    finally:
        cap.release()


class VideoReader:
    def __init__(self, file):
        self.file = file
        self.container = None

    def __enter__(self):
        self.container = av.open(self.file)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.container.close()

    def frames(self):
        for frame in self.container.decode(video=0):
            frame_img = frame.to_ndarray(format='bgr24')
            yield frame_img, frame.pts

    def seek(self, offset_ns):
        offset = int((offset_ns / 1_000_000_000) * av.time_base)
        self.container.seek(offset)


def get_video_meta_data(path: str) -> VideoMetadata:
    # Resolve cross-OS path variants before probing
    path = resolve_path_compat(path)
    cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-select_streams', 'v', '-show_streams', '-show_format', path]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err =  p.communicate()
    if p.returncode != 0:
        raise Exception(f"error running ffprobe: {err.strip()}. Code: {p.returncode}, cmd: {cmd}")
    json_output = json.loads(out)
    json_video_stream = json_output["streams"][0]
    json_video_format = json_output["format"]

    value = [int(num) for num in json_video_stream['avg_frame_rate'].split("/")]
    average_fps = value[0]/value[1] if len(value) == 2 else value[0]

    value = [int(num) for num in json_video_stream['r_frame_rate'].split("/")]
    fps = value[0]/value[1] if len(value) == 2 else value[0]
    fps_exact = Fraction(value[0], value[1])

    value = [int(num) for num in json_video_stream['time_base'].split("/")]
    time_base = Fraction(value[0], value[1])

    frame_count = json_video_stream.get('nb_frames')
    if not frame_count:
        # print("frame count ffmpeg", frame_count)
        cap = cv2.VideoCapture(path)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        # print("frame count opencv", frame_count)
    frame_count=int(frame_count)

    start_pts = json_video_stream.get('start_pts')

    metadata = VideoMetadata(
        video_file=path,
        video_height=int(json_video_stream['height']),
        video_width=int(json_video_stream['width']),
        video_fps=fps,
        average_fps=average_fps,
        video_fps_exact=fps_exact,
        codec_name=json_video_stream['codec_name'],
        frames_count=frame_count,
        duration=float(json_video_stream.get('duration', json_video_format['duration'])),
        time_base=time_base,
        start_pts=start_pts
    )
    return metadata


def offset_ns_to_frame_num(offset_ns, video_fps_exact):
    return int(Fraction(offset_ns, 1_000_000_000) * video_fps_exact)


def write_frames_to_video_file(frames: list[Image], output_path, fps: int | float | Fraction, codec='x264', preset='medium', crf=None):
    assert frames[0].ndim == 3
    width = frames[0].shape[1]
    height = frames[0].shape[0]
    ffmpeg_output = [
        'nice', '-n', '19', 'ffmpeg', '-y',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{width}x{height}', '-r', f"{fps.numerator}/{fps.denominator}" if type(fps) == Fraction else str(fps),
        '-i', '-', '-an', '-preset', preset
    ]
    if codec == 'x265':
        ffmpeg_output.extend(['-tag:v', 'hvc1', '-vcodec', 'libx265', '-crf', str(crf) if crf else '18'])
    elif codec == 'x264':
        ffmpeg_output.extend(['-vcodec', 'libx264', '-crf', str(crf) if crf else '15'])
    ffmpeg_output.append(output_path)

    ffmpeg_process = subprocess.Popen(ffmpeg_output, stdin=subprocess.PIPE, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ffmpeg_process.stdin.write(frame.tobytes())
    ffmpeg_process.stdin.close()
    ffmpeg_process.wait()
    if ffmpeg_process.returncode != 0:
        print(f"ERROR when writing video via ffmpeg to file: {output_path}, return code: {ffmpeg_process.returncode}")
