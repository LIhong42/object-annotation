"""Extract parameterized color-overlay masks and map them to the source."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .annotation_colors import normalize_annotation_color


def annotation_color_pixels(
    bgr: np.ndarray, annotation_color: str = "red"
) -> np.ndarray:
    """Return high-confidence pixels for one supported annotation color."""
    color = normalize_annotation_color(annotation_color)
    values = np.asarray(bgr)[..., :3].astype(np.int16)
    if color in {"red", "green", "blue"}:
        channel = {"blue": 0, "green": 1, "red": 2}[color]
        selected = values[..., channel]
        others = np.max(np.delete(values, channel, axis=2), axis=2)
        return (selected >= 220) & (others <= 45) & ((selected - others) >= 170)
    if color == "black":
        return np.max(values, axis=2) <= 45
    return np.min(values, axis=2) >= 220


def solid_red_pixels(bgr: np.ndarray) -> np.ndarray:
    """Backward-compatible alias for red annotation pixels."""
    return annotation_color_pixels(bgr, "red")


def _soft_color_support(
    labeling_bgr: np.ndarray,
    original_in_label: np.ndarray,
    *,
    annotation_color: str,
    change_threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return permissive support for a generated overlay of the chosen color."""
    color = normalize_annotation_color(annotation_color)
    label = labeling_bgr[..., :3].astype(np.float32)
    source = original_in_label[..., :3].astype(np.float32)
    difference = np.linalg.norm(label - source, axis=2)
    if color in {"red", "green", "blue"}:
        channel = {"blue": 0, "green": 1, "red": 2}[color]
        other_channels = [index for index in range(3) if index != channel]
        selected = label[..., channel]
        source_selected = source[..., channel]
        other = np.max(label[..., other_channels], axis=2)
        dominance = (
            (selected >= 100.0)
            & (selected >= 1.60 * other + 5.0)
            & ((selected - other) >= 70.0)
        )
        direction = (selected - source_selected >= 8.0) | np.all(
            source[..., other_channels] - label[..., other_channels] >= 8.0,
            axis=2,
        )
        color_support = dominance & direction
    else:
        spread = np.max(label, axis=2) - np.min(label, axis=2)
        label_mean = np.mean(label, axis=2)
        source_mean = np.mean(source, axis=2)
        if color == "black":
            color_support = (
                (np.max(label, axis=2) <= 145.0)
                & (spread <= 75.0)
                & ((source_mean - label_mean) >= 8.0)
            )
        else:
            color_support = (
                (np.min(label, axis=2) >= 110.0)
                & (spread <= 75.0)
                & ((label_mean - source_mean) >= 8.0)
            )
    support = color_support & (difference >= float(change_threshold))
    return support, difference


def _grow_from_seeds(seeds: np.ndarray, support: np.ndarray) -> np.ndarray:
    """Geodesically grow strong seeds only through same-color support."""
    current = (seeds & support).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    while True:
        grown = cv2.dilate(current, kernel, iterations=1)
        grown = ((grown > 0) & support).astype(np.uint8)
        grown |= current
        if np.array_equal(grown, current):
            return grown
        current = grown


def _fill_supported_holes(
    component: np.ndarray,
    difference: np.ndarray,
    *,
    change_threshold: float,
) -> np.ndarray:
    """Fill enclosed texture holes only when the source-to-label edit supports it."""
    filled = component.astype(np.uint8).copy()
    inverse = (filled == 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(inverse, 8)
    height, width = filled.shape
    for component_id in range(1, count):
        x = int(stats[component_id, cv2.CC_STAT_LEFT])
        y = int(stats[component_id, cv2.CC_STAT_TOP])
        w = int(stats[component_id, cv2.CC_STAT_WIDTH])
        h = int(stats[component_id, cv2.CC_STAT_HEIGHT])
        if x == 0 or y == 0 or x + w == width or y + h == height:
            continue
        hole = labels == component_id
        if float(difference[hole].mean()) >= float(change_threshold):
            filled[hole] = 1
    return filled


def extract_filled_instances(
    labeling_bgr: np.ndarray,
    original_bgr: np.ndarray,
    affine_label_to_original: np.ndarray,
    *,
    annotation_color: str = "red",
    min_pixels: int = 6,
    change_threshold: int = 35,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Return chosen-color instances in labeling coordinates.

    Existing same-color artwork is rejected by comparing the labeled image with the
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
    color = normalize_annotation_color(annotation_color)
    strong = annotation_color_pixels(labeling_bgr, color)
    strict_count, _, strict_stats, _ = cv2.connectedComponentsWithStats(
        strong.astype(np.uint8), 8
    )
    max_difference = np.max(
        np.abs(
            labeling_bgr[..., :3].astype(np.int16)
            - original_in_label[..., :3].astype(np.int16)
        ),
        axis=2,
    )
    added = strong & (max_difference >= max(1, int(change_threshold)))

    support_threshold = max(12.0, float(change_threshold) * 0.55)
    support, euclidean_difference = _soft_color_support(
        labeling_bgr,
        original_in_label,
        annotation_color=color,
        change_threshold=support_threshold,
    )
    candidate = _grow_from_seeds(added, support)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate.astype(np.uint8), 8
    )
    instances: List[np.ndarray] = []
    for component_id in range(1, count):
        if int(stats[component_id, cv2.CC_STAT_AREA]) < int(min_pixels):
            continue
        component = (labels == component_id).astype(np.uint8)
        if not np.any((component > 0) & added):
            continue
        filled = _fill_supported_holes(
            component,
            euclidean_difference,
            change_threshold=support_threshold,
        )
        if int(filled.sum()) >= int(min_pixels):
            instances.append(filled)

    instances.sort(key=_mask_sort_key)
    union = np.zeros((lh, lw), np.uint8)
    for mask in instances:
        union |= mask
    if diagnostics is not None:
        diagnostics.update({
            "method": "annotation_color_change_hysteresis",
            "annotation_color": color,
            "strong_color_pixels": int(strong.sum()),
            "strong_color_components": int(strict_count - 1),
            "strong_color_components_at_least_min_pixels": int(
                np.count_nonzero(
                    strict_stats[1:, cv2.CC_STAT_AREA] >= int(min_pixels)
                )
            ),
            "color_change_support_pixels": int(support.sum()),
            "candidate_components": int(count - 1),
            "returned_instances": int(len(instances)),
        })
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
