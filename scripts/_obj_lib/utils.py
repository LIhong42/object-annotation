from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence, Tuple

import cv2
import numpy as np

from .serialization import to_json_safe
from .types import BBox


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_bgr(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {path}")
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        rgb = img[:, :, :3].astype(np.float32)
        white = np.full_like(rgb, 255.0)
        return np.clip(rgb * alpha + white * (1.0 - alpha), 0, 255).astype(np.uint8)
    return img[:, :, :3].copy()


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def clip_bbox(box: Sequence[float], width: int, height: int) -> BBox:
    x1, y1, x2, y2 = box
    x1 = int(np.floor(max(0.0, min(float(width - 1), float(x1)))))
    y1 = int(np.floor(max(0.0, min(float(height - 1), float(y1)))))
    x2 = int(np.ceil(max(float(x1 + 1), min(float(width), float(x2)))))
    y2 = int(np.ceil(max(float(y1 + 1), min(float(height), float(y2)))))
    return x1, y1, x2, y2


def bbox_size(box: BBox) -> Tuple[int, int]:
    return max(1, box[2] - box[0]), max(1, box[3] - box[1])


def bbox_area(box: BBox) -> int:
    w, h = bbox_size(box)
    return w * h


def bbox_center(box: BBox) -> Tuple[float, float]:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def inset_bbox(box: BBox, px: int, width: int, height: int) -> BBox:
    x1, y1, x2, y2 = box
    if (x2 - x1) <= 2 * px + 2 or (y2 - y1) <= 2 * px + 2:
        return clip_bbox(box, width, height)
    return clip_bbox((x1 + px, y1 + px, x2 - px, y2 - px), width, height)


def expand_bbox(
    box: BBox,
    ratio_x: float,
    ratio_y: float,
    width: int,
    height: int,
    min_px: int = 0,
    max_px: int | None = None,
) -> BBox:
    w, h = bbox_size(box)
    mx = max(min_px, int(round(w * ratio_x)))
    my = max(min_px, int(round(h * ratio_y)))
    if max_px is not None:
        mx = min(mx, max_px)
        my = min(my, max_px)
    return clip_bbox((box[0] - mx, box[1] - my, box[2] + mx, box[3] + my), width, height)


def transform_points(points: np.ndarray, affine: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    ones = np.ones((len(points), 1), dtype=np.float32)
    homo = np.concatenate([points, ones], axis=1)
    return homo @ affine.T


def transform_bbox(box: BBox, affine: np.ndarray, width: int, height: int) -> BBox:
    x1, y1, x2, y2 = box
    corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    warped = transform_points(corners, affine)
    return clip_bbox(
        (
            float(warped[:, 0].min()),
            float(warped[:, 1].min()),
            float(warped[:, 0].max()),
            float(warped[:, 1].max()),
        ),
        width,
        height,
    )


def bbox_contains(outer: BBox, inner: BBox, tolerance: int = 0) -> bool:
    return (
        outer[0] - tolerance <= inner[0]
        and outer[1] - tolerance <= inner[1]
        and outer[2] + tolerance >= inner[2]
        and outer[3] + tolerance >= inner[3]
    )


def bbox_iou(a: BBox, b: BBox) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = bbox_area(a) + bbox_area(b) - inter
    return float(inter / union) if union > 0 else 0.0


def mask_bbox(mask: np.ndarray) -> BBox | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def save_json(path: str | Path, data: object) -> None:
    payload = to_json_safe(data)
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def draw_bbox(img: np.ndarray, box: BBox, color: Tuple[int, int, int], thickness: int = 2) -> None:
    cv2.rectangle(img, (box[0], box[1]), (box[2] - 1, box[3] - 1), color, thickness, cv2.LINE_AA)


def safe_crop(img: np.ndarray, box: BBox) -> np.ndarray:
    x1, y1, x2, y2 = box
    return img[y1:y2, x1:x2]


def percentile_normalize(arr: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    arr = arr.astype(np.float32)
    scale = float(np.percentile(arr, percentile))
    if scale < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip(arr / scale, 0.0, 1.0)


def gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    gray_f = gray.astype(np.float32) / 255.0
    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    return percentile_normalize(cv2.magnitude(gx, gy))


def edge_map(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    med = float(np.median(blur))
    low = int(max(10, 0.66 * med))
    high = int(min(255, max(low + 20, 1.33 * med)))
    return cv2.Canny(blur, low, high)
