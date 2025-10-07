import sys
import pathlib
from argparse import ArgumentParser

import cv2
import numpy as np

try:
    import vpi
except ImportError:  # pragma: no cover - optional dependency
    vpi = None


def parse_checkerboard(value: str) -> np.ndarray:
    parts = value.split(',')
    if len(parts) != 2:
        raise ValueError("checkerboard must be specified as W,H")
    try:
        dims = np.array([int(p) for p in parts], dtype=np.int32)
    except ValueError as exc:
        raise ValueError("checkerboard dimensions must be integers") from exc
    if np.any(dims <= 1):
        raise ValueError("checkerboard dimensions must be > 1")
    return dims


def find_corners(images, vertices_count):
    img_size = None
    corners_all = []
    for name in images:
        img = cv2.imread(str(name))
        if img is None:
            raise RuntimeError(f"failed to load image {name}")
        current_size = (img.shape[1], img.shape[0])
        if img_size is None:
            img_size = current_size
        elif current_size != img_size:
            raise RuntimeError("all images must have the same size")
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(img, tuple(vertices_count), flags)
        if not found:
            raise RuntimeError(f"checkerboard pattern not found in image {name}")
        corners_all.append((img, corners))
    return img_size, corners_all


def refine_corners(corners_all, search_window):
    if search_window is None or search_window < 2:
        return [c for _, c in corners_all]
    win = (search_window // 2, search_window // 2)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.0001)
    refined = []
    for img, corners in corners_all:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        refined.append(cv2.cornerSubPix(gray, corners, win, (-1, -1), criteria))
    return refined


def calibrate(cb_dims, img_size, corners_2d):
    vertices_count = cb_dims - 1
    cb_corners = np.zeros((1, vertices_count[0] * vertices_count[1], 3), dtype=np.float32)
    grid = np.mgrid[0:vertices_count[0], 0:vertices_count[1]].T.reshape(-1, 2)
    cb_corners[0, :, :2] = grid
    corners_3d = [cb_corners.copy() for _ in corners_2d]
    cam_matrix = np.eye(3)
    coeffs = np.zeros((4,))
    rms, cam_matrix, coeffs, rvecs, tvecs = cv2.fisheye.calibrate(
        corners_3d,
        corners_2d,
        img_size,
        cam_matrix,
        coeffs,
        flags=cv2.fisheye.CALIB_FIX_SKEW,
    )
    return rms, cam_matrix, coeffs


def choose_backend():
    if vpi is None:
        raise RuntimeError("python-vpi bindings are not available")
    backends = [vpi.Backend.CUDA, vpi.Backend.VIC, vpi.Backend.CPU]
    last_error = None
    for backend in backends:
        try:
            with backend:
                pass
            return backend
        except Exception as exc:  # pragma: no cover
            last_error = exc
    raise RuntimeError("no VPI backend available") from last_error


def undistort_images_vpi(images, cam_matrix, coeffs, img_size):
    grid = vpi.WarpGrid(img_size)
    K23 = cam_matrix[:2, :]
    undist_map = vpi.WarpMap.fisheye_correction(
        grid,
        K=K23,
        X=np.eye(3, 4),
        coeffs=coeffs,
        mapping=vpi.FisheyeMapping.EQUIDISTANT,
    )
    backend = choose_backend()
    outputs = []
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError(f"failed to load image {img_path}")
        with backend:
            with vpi.asimage(img) as vimg:
                remapped = vimg.remap(
                    undist_map,
                    interp=vpi.Interp.CATMULL_ROM,
                    border=vpi.Border.ZERO,
                )
                out_img = remapped.cpu()
        out_path = img_path.with_name(img_path.stem + "_undist" + img_path.suffix)
        cv2.imwrite(str(out_path), out_img)
        outputs.append(out_path)
    return outputs


def undistort_images_cv(images, cam_matrix, coeffs, img_size):
    outputs = []
    R = np.eye(3)
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        cam_matrix,
        coeffs,
        img_size,
        R,
        balance=1.0,
    )
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        cam_matrix,
        coeffs,
        R,
        new_K,
        img_size,
        cv2.CV_32FC1,
    )
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError(f"failed to load image {img_path}")
        out_img = cv2.remap(
            img,
            map1,
            map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        out_path = img_path.with_name(img_path.stem + "_undist" + img_path.suffix)
        cv2.imwrite(str(out_path), out_img)
        outputs.append(out_path)
    return outputs


def undistort_images(images, cam_matrix, coeffs, img_size):
    if vpi is None:
        print("Warning: python-vpi not installed, falling back to OpenCV remap.")
        return undistort_images_cv(images, cam_matrix, coeffs, img_size)
    return undistort_images_vpi(images, cam_matrix, coeffs, img_size)



def main(argv=None):
    version = cv2.__version__.split('.')
    major = int(version[0])
    minor = int(version[1])
    if major * 100 + minor >= 410:
        raise RuntimeError("OpenCV >= 4.10 isn't supported by this sample")

    parser = ArgumentParser(description="NVIDIA VPI fisheye calibration + undistortion sample")
    parser.add_argument('-c', metavar='W,H', required=True, help='Checkerboard with WxH squares')
    parser.add_argument('-s', metavar='win', type=int, help='Search window width used in corner refinement')
    parser.add_argument('images', nargs='+', help='Input images taken with a fisheye lens camera')
    args = parser.parse_args(argv)

    cb_dims = parse_checkerboard(args.c)
    vertices_count = cb_dims - 1

    image_paths = [pathlib.Path(p).resolve() for p in args.images]

    img_size, corners_all = find_corners(image_paths, vertices_count)
    corners_2d = refine_corners(corners_all, args.s)

    rms, cam_matrix, coeffs = calibrate(cb_dims, img_size, corners_2d)
    print(f"rms error: {rms}")
    print(f"Fisheye coefficients: {coeffs}")
    print("Camera matrix:")
    print(cam_matrix)

    outputs = undistort_images(image_paths, cam_matrix, coeffs, img_size)
    for out_path in outputs:
        print(f"Saved undistorted image to {out_path}")


if __name__ == '__main__':
    main()
