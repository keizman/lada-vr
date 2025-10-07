import math
import numpy as np

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None


def _rot_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    roll = math.radians(roll_deg)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cx, sx = math.cos(pitch), math.sin(pitch)
    cz, sz = math.cos(roll), math.sin(roll)
    # Yaw (Y), Pitch (X), Roll (Z) -> R = Rz * Rx * Ry
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    R = (Rz @ Rx) @ Ry
    return R.astype(np.float32)


def build_equirect_to_perspective_map(src_w: int,
                                      src_h: int,
                                      out_w: int,
                                      out_h: int,
                                      fov_deg: float,
                                      yaw_deg: float = 0.0,
                                      pitch_deg: float = 0.0,
                                      roll_deg: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Build remap matrices (map_x, map_y) to transform an equirectangular image (src_w x src_h)
    into a perspective (rectilinear) view of size out_w x out_h with given FOV and orientation.

    Coordinates convention:
    - Equirect u in [0, src_w), v in [0, src_h)
    - Perspective x right, y down; camera looks along +Z
    """
    fov = math.radians(float(fov_deg))
    # Horizontal FOV = fov; derive vertical FOV from aspect
    half_fov_x = fov / 2.0
    half_fov_y = math.atan(math.tan(half_fov_x) * (out_h / float(out_w)))

    # Grid of pixel centers
    j = np.linspace(0, out_w - 1, out_w, dtype=np.float32)
    i = np.linspace(0, out_h - 1, out_h, dtype=np.float32)
    xv, yv = np.meshgrid(j, i)

    # Normalized camera plane coordinates via tangent mapping
    x = (2.0 * (xv + 0.5) / float(out_w) - 1.0) * math.tan(half_fov_x)
    y = (2.0 * (yv + 0.5) / float(out_h) - 1.0) * math.tan(half_fov_y)
    y = -y  # image coords to camera coords (y up)
    z = np.ones_like(x, dtype=np.float32)

    # Normalize direction vectors
    inv_norm = 1.0 / np.sqrt(x * x + y * y + z * z)
    x *= inv_norm
    y *= inv_norm
    z *= inv_norm

    # Rotate by yaw/pitch/roll
    R = _rot_matrix(yaw_deg, pitch_deg, roll_deg)
    xr = R[0, 0] * x + R[0, 1] * y + R[0, 2] * z
    yr = R[1, 0] * x + R[1, 1] * y + R[1, 2] * z
    zr = R[2, 0] * x + R[2, 1] * y + R[2, 2] * z

    # Convert direction to spherical (lon, lat)
    lon = np.arctan2(xr, zr)
    lat = np.arctan2(yr, np.sqrt(xr * xr + zr * zr))

    # Map to equirect pixel coordinates
    u = (lon + math.pi) / (2.0 * math.pi) * float(src_w)
    v = (math.pi / 2.0 - lat) / math.pi * float(src_h)

    map_x = np.clip(u, 0.0, float(src_w - 1)).astype(np.float32)
    map_y = np.clip(v, 0.0, float(src_h - 1)).astype(np.float32)
    return map_x, map_y


def equirect_to_perspective(img: np.ndarray,
                             out_w: int,
                             out_h: int,
                             fov_deg: float,
                             yaw_deg: float = 0.0,
                             pitch_deg: float = 0.0,
                             roll_deg: float = 0.0) -> np.ndarray:
    """
    Remap an equirectangular image to a perspective image using precomputed maps.
    """
    assert cv2 is not None, "OpenCV is required"
    src_h, src_w = img.shape[:2]
    mx, my = build_equirect_to_perspective_map(src_w, src_h, out_w, out_h, fov_deg, yaw_deg, pitch_deg, roll_deg)
    out = cv2.remap(img, mx, my, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return out


def equirect_to_perspective_tiles(img: np.ndarray,
                                  tiles_x: int,
                                  tiles_y: int,
                                  fov_deg: float,
                                  coverage: str = "360",
                                  out_w: int = 0,
                                  out_h: int = 0) -> tuple[list[list[np.ndarray]], list[list[tuple[float, float]]], np.ndarray]:
    """
    Generate a grid of perspective tiles that cover the sphere (360) or hemisphere (180).
    Returns (tiles_grid, centers_grid, mosaic_image)
      - tiles_grid: tiles_y x tiles_x list of images
      - centers_grid: same shape list of (yaw, pitch) centers in degrees
      - mosaic_image: a concatenated preview image
    If out_w/out_h are 0, each tile defaults to img height/tiles_y by img width/tiles_x (approx keep scale).
    """
    assert cv2 is not None, "OpenCV is required"
    src_h, src_w = img.shape[:2]
    # Determine per-tile output size if not specified
    if out_w <= 0 or out_h <= 0:
        # Heuristic: distribute resolution roughly evenly across grid
        tile_w = max(256, (src_w // max(2, tiles_x)))
        tile_h = max(256, (src_h // max(2, tiles_y)))
    else:
        tile_w, tile_h = out_w, out_h

    # Coverage ranges
    if coverage == "180":
        yaw_min, yaw_range = -90.0, 180.0
    else:  # 360
        yaw_min, yaw_range = -180.0, 360.0
    pitch_min, pitch_range = -90.0, 180.0

    tiles: list[list[np.ndarray]] = []
    centers: list[list[tuple[float, float]]] = []
    # Centers at equal-area-ish grid (simple uniform in lon/lat centers)
    for j in range(tiles_y):
        row_tiles: list[np.ndarray] = []
        row_centers: list[tuple[float, float]] = []
        pitch = pitch_min + (j + 0.5) * (pitch_range / tiles_y)
        for i in range(tiles_x):
            yaw = yaw_min + (i + 0.5) * (yaw_range / tiles_x)
            tile = equirect_to_perspective(img, tile_w, tile_h, fov_deg, yaw_deg=yaw, pitch_deg=pitch, roll_deg=0.0)
            row_tiles.append(tile)
            row_centers.append((yaw, pitch))
        tiles.append(row_tiles)
        centers.append(row_centers)

    # Build mosaic
    rows = [np.concatenate(row, axis=1) for row in tiles]
    mosaic = np.concatenate(rows, axis=0) if rows else np.zeros((tile_h, tile_w, 3), dtype=img.dtype)
    return tiles, centers, mosaic


def build_perspective_from_equirect_patch_map(src_w: int,
                                              src_h: int,
                                              out_w: int,
                                              out_h: int,
                                              fov_deg: float,
                                              yaw_deg: float,
                                              pitch_deg: float,
                                              roll_deg: float,
                                              patch_x0: int,
                                              patch_y0: int,
                                              patch_w: int,
                                              patch_h: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build maps to sample from ROI perspective image back onto a patch of the equirect image.
    Returns (map_x, map_y, valid_mask) where map_* have shape (patch_h, patch_w) and index ROI image coords.
    """
    assert cv2 is not None, "OpenCV is required"
    fov = math.radians(float(fov_deg))
    half_fov_x = fov / 2.0
    half_fov_y = math.atan(math.tan(half_fov_x) * (out_h / float(out_w)))
    tanx = math.tan(half_fov_x)
    tany = math.tan(half_fov_y)

    # Grid in source (patch) pixel centers
    j = np.linspace(0, patch_w - 1, patch_w, dtype=np.float32) + 0.5
    i = np.linspace(0, patch_h - 1, patch_h, dtype=np.float32) + 0.5
    xv, yv = np.meshgrid(j, i)
    u = (patch_x0 + xv)
    v = (patch_y0 + yv)

    # Equirect -> lon/lat (world)
    lon = (u / float(src_w)) * (2.0 * math.pi) - math.pi
    lat = (0.5 - (v / float(src_h))) * math.pi

    # Direction vector in world
    coslat = np.cos(lat)
    wx = coslat * np.sin(lon)
    wy = np.sin(lat)
    wz = coslat * np.cos(lon)

    # World -> camera (inverse rotation)
    R = _rot_matrix(yaw_deg, pitch_deg, roll_deg)
    Rt = R.T
    xc = Rt[0, 0] * wx + Rt[0, 1] * wy + Rt[0, 2] * wz
    yc = Rt[1, 0] * wx + Rt[1, 1] * wy + Rt[1, 2] * wz
    zc = Rt[2, 0] * wx + Rt[2, 1] * wy + Rt[2, 2] * wz

    # Perspective projection to ROI image coords
    eps = 1e-6
    zc_safe = np.where(zc == 0.0, eps, zc)
    x_plane = xc / zc_safe
    y_plane = yc / zc_safe

    valid = (zc > 0.0) & (np.abs(x_plane) <= tanx) & (np.abs(y_plane) <= tany)

    x_norm = (x_plane / tanx)
    y_norm = (y_plane / tany)
    x_out = ((x_norm + 1.0) * 0.5) * float(out_w) - 0.5
    y_out = ((-y_norm + 1.0) * 0.5) * float(out_h) - 0.5

    map_x = x_out.astype(np.float32)
    map_y = y_out.astype(np.float32)
    valid_mask = valid.astype(np.uint8)  # 0/1
    return map_x, map_y, valid_mask

