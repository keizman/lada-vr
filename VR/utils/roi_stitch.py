import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Rect:
    x0: int
    y0: int
    w: int
    h: int

@dataclass
class PartMap:
    src: Rect  # rect in original single-eye image
    dst: Rect  # rect in combined 2D image

@dataclass
class CombinedResult:
    image: np.ndarray
    maps: List[PartMap]


def _grid_rects(w: int, h: int, grid: int) -> List[Rect]:
    xs = [int(round((w * i) / grid)) for i in range(grid + 1)]
    ys = [int(round((h * i) / grid)) for i in range(grid + 1)]
    rects: List[Rect] = []
    for r in range(grid):
        for c in range(grid):
            x0 = xs[c]
            y0 = ys[r]
            rect_w = max(1, xs[c + 1] - x0)
            rect_h = max(1, ys[r + 1] - y0)
            rects.append(Rect(x0=x0, y0=y0, w=rect_w, h=rect_h))
    return rects


def build_roi_grid3x3_combined(img: np.ndarray,
                                *,
                                grid: int = 5,
                                row_indices: Tuple[int, ...] = (2, 3, 4),
                                col_indices: Tuple[int, ...] = (1, 2, 3),
                                row_h_ratio: float = 0.18) -> CombinedResult:
    """
    Build a combined 2D image by sampling rectangular cells from a uniform grid.
    By default this takes the bottom 3x3 group from a conceptual 5x5 split
    (cells 12-14, 17-19, 22-24 in 1-based indexing), which corresponds to
    the lower-center portion of the eye image.
    """
    if grid <= 1:
        raise ValueError("grid must be > 1")
    if not row_indices or not col_indices:
        raise ValueError("row_indices and col_indices cannot be empty")
    if any(r < 0 or r >= grid for r in row_indices):
        raise ValueError("row_indices out of range for grid")
    if any(c < 0 or c >= grid for c in col_indices):
        raise ValueError("col_indices out of range for grid")

    h, w = img.shape[:2]
    rects = _grid_rects(w, h, grid)

    def crop(rect: Rect) -> np.ndarray:
        return img[rect.y0:rect.y0 + rect.h, rect.x0:rect.x0 + rect.w]

    row_src: List[List[Tuple[Rect, np.ndarray]]] = []
    for r in row_indices:
        row_parts: List[Tuple[Rect, np.ndarray]] = []
        for c in col_indices:
            src_rect = rects[r * grid + c]
            row_parts.append((src_rect, crop(src_rect)))
        row_src.append(row_parts)

    row_h = max(32, int(h * row_h_ratio))

    def scale_to_row(part: np.ndarray) -> Tuple[np.ndarray, int]:
        ph, pw = part.shape[:2]
        if ph <= 0 or pw <= 0:
            return np.zeros((row_h, 1, 3), dtype=img.dtype), 1
        scale = row_h / float(ph)
        new_w = max(1, int(round(pw * scale)))
        resized = cv2.resize(part, (new_w, row_h), interpolation=cv2.INTER_AREA)
        return resized, new_w

    scaled_rows: List[List[Tuple[Rect, np.ndarray]]] = []
    row_images: List[np.ndarray] = []
    for parts in row_src:
        scaled: List[Tuple[Rect, np.ndarray]] = []
        tiles: List[np.ndarray] = []
        for rect_src, patch in parts:
            resized, _ = scale_to_row(patch)
            scaled.append((rect_src, resized))
            tiles.append(resized)
        if tiles:
            row_img = cv2.hconcat(tiles)
        else:
            row_img = np.zeros((row_h, 1, 3), dtype=img.dtype)
        scaled_rows.append(scaled)
        row_images.append(row_img)

    canvas_w = max((row.shape[1] for row in row_images), default=1)

    padded_rows: List[np.ndarray] = []
    offsets_x: List[int] = []
    for row_img in row_images:
        w_now = row_img.shape[1]
        pad = max(0, canvas_w - w_now)
        pad_left = pad // 2
        pad_right = pad - pad_left
        padded_rows.append(cv2.copyMakeBorder(row_img, 0, 0, pad_left, pad_right, cv2.BORDER_REFLECT))
        offsets_x.append(pad_left)

    combined = cv2.vconcat(padded_rows)

    maps: List[PartMap] = []
    y_cursor = 0
    for row_idx, scaled in enumerate(scaled_rows):
        x_cursor = offsets_x[row_idx]
        for rect_src, resized in scaled:
            width = resized.shape[1]
            maps.append(PartMap(src=rect_src, dst=Rect(x0=x_cursor, y0=y_cursor, w=width, h=row_h)))
            x_cursor += width
        y_cursor += row_h

    return CombinedResult(image=combined, maps=maps)


def composite_back_from_combined(original: np.ndarray,
                                 combined_fixed: np.ndarray,
                                 maps: List[PartMap],
                                 feather: int = 12) -> np.ndarray:
    """
    Paste processed combined 2D back to original using recorded maps.
    Each dst rect in combined is resized to src rect size and alpha-blended (Hanning).
    """
    out = original.copy()
    for m in maps:
        x0_d, y0_d, w_d, h_d = m.dst.x0, m.dst.y0, m.dst.w, m.dst.h
        x0_s, y0_s, w_s, h_s = m.src.x0, m.src.y0, m.src.w, m.src.h
        if w_d <= 0 or h_d <= 0 or w_s <= 0 or h_s <= 0:
            continue
        patch = combined_fixed[y0_d:y0_d + h_d, x0_d:x0_d + w_d]
        if patch.size == 0:
            continue
        patch_rs = cv2.resize(patch, (w_s, h_s), interpolation=cv2.INTER_LINEAR)
        # Feather
        fx = min(feather, w_s // 4)
        fy = min(feather, h_s // 4)
        if fx < 1:
            fx = 1
        if fy < 1:
            fy = 1
        wx = np.ones((w_s,), dtype=np.float32)
        edge_x = np.hanning(max(2, fx * 2)).astype(np.float32)
        wx[:fx] = edge_x[:fx]
        wx[-fx:] = edge_x[-fx:]
        wy = np.ones((h_s,), dtype=np.float32)
        edge_y = np.hanning(max(2, fy * 2)).astype(np.float32)
        wy[:fy] = edge_y[:fy]
        wy[-fy:] = edge_y[-fy:]
        alpha = np.outer(wy, wx)[..., None]
        roi = out[y0_s:y0_s + h_s, x0_s:x0_s + w_s].astype(np.float32)
        patch_f = patch_rs.astype(np.float32)
        blended = (1.0 - alpha) * roi + alpha * patch_f
        out[y0_s:y0_s + h_s, x0_s:x0_s + w_s] = np.clip(blended, 0, 255).astype(np.uint8)
    return out
