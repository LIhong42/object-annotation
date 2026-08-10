"""Extract solid-red instance masks and map them into the original image."""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np


def solid_red_pixels(bgr: np.ndarray) -> np.ndarray:
    """Return pixels close to pure annotation red (BGR input)."""
    values = np.asarray(bgr)[..., :3].astype(np.int16)
    b, g, r = values[..., 0], values[..., 1], values[..., 2]
    return (
        (r >= 220)
        & (g <= 45)
        & (b <= 45)
        & (r - np.maximum(g, b) >= 170)
    )


def extract_filled_instances(
    labeling_bgr: np.ndarray,
    original_bgr: np.ndarray,
    affine_label_to_original: np.ndarray,
    *,
    min_pixels: int = 6,
    change_threshold: int = 35,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Return filled-red instances in labeling coordinates.

    Existing red artwork is rejected by comparing the labeled image with the
    original warped into the labeling coordinate system.  Holes caused by red
    pixels already present inside a newly filled object are filled again per
    connected component.
    """
    lh, lw = labeling_bgr.shape[:2]
    inverse = cv2.invertAffineTransform(
        np.asarray(affine_label_to_original, dtype=np.float64)
    )
    original_in_label = cv2.warpAffine(
        original_bgr,
        inverse,
        (lw, lh),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    red = solid_red_pixels(labeling_bgr)
    difference = np.max(
        np.abs(
            labeling_bgr[..., :3].astype(np.int16)
            - original_in_label[..., :3].astype(np.int16)
        ),
        axis=2,
    )
    added = red & (difference >= max(1, int(change_threshold)))
    added = cv2.morphologyEx(
        added.astype(np.uint8),
        cv2.MORPH_CLOSE,
        np.ones((3, 3), np.uint8),
        iterations=1,
    )

    count, labels, stats, _ = cv2.connectedComponentsWithStats(added, 8)
    instances: List[np.ndarray] = []
    for component_id in range(1, count):
        if int(stats[component_id, cv2.CC_STAT_AREA]) < int(min_pixels):
            continue
        component = (labels == component_id).astype(np.uint8)
        contours, _ = cv2.findContours(
            component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        filled = np.zeros_like(component)
        if contours:
            cv2.drawContours(filled, contours, -1, 1, cv2.FILLED)
        if int(filled.sum()) >= int(min_pixels):
            instances.append(filled)

    instances.sort(key=_mask_sort_key)
    union = np.zeros((lh, lw), np.uint8)
    for mask in instances:
        union |= mask
    return instances, union


def map_masks_to_original(
    masks: List[np.ndarray],
    affine_label_to_original: np.ndarray,
    original_width: int,
    original_height: int,
    *,
    min_pixels: int = 6,
) -> List[np.ndarray]:
    """Warp instance masks with nearest-neighbour sampling into original space."""
    mapped: List[np.ndarray] = []
    for mask in masks:
        warped = cv2.warpAffine(
            mask.astype(np.uint8),
            np.asarray(affine_label_to_original, dtype=np.float64),
            (int(original_width), int(original_height)),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        warped = (warped > 0).astype(np.uint8)
        if int(warped.sum()) >= int(min_pixels):
            mapped.append(warped)
    mapped.sort(key=_mask_sort_key)
    return mapped


def _mask_sort_key(mask: np.ndarray) -> Tuple[int, int, int]:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return (mask.shape[0], mask.shape[1], 0)
    return int(ys.min()), int(xs.min()), -int(xs.size)
