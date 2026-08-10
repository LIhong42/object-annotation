from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from .types import RegistrationResult
from .utils import gradient_magnitude, transform_points


def _feature_detector():
    if hasattr(cv2, "SIFT_create"):
        return cv2.SIFT_create(nfeatures=6000, contrastThreshold=0.008, edgeThreshold=20), cv2.NORM_L2, "SIFT"
    return cv2.AKAZE_create(), cv2.NORM_HAMMING, "AKAZE"


def estimate_global_affine(
    labeling_bgr: np.ndarray,
    object_bgr: np.ndarray,
    red_mask: np.ndarray,
) -> Tuple[np.ndarray, RegistrationResult]:
    """估计 labeling -> object 的全局锚点变换。

    同时估计相似变换和完整仿射，按实际内点数、内点比例和重投影误差
    选择结果。除矩阵必须有限且可逆外，不预设缩放、各向异性或平移界限。
    返回的首次全局矩阵直接用于最终 mask 映射，不进入局部修正。
    """
    label_gray = cv2.cvtColor(labeling_bgr, cv2.COLOR_BGR2GRAY)
    object_gray = cv2.cvtColor(object_bgr, cv2.COLOR_BGR2GRAY)

    invalid = cv2.dilate(red_mask, np.ones((9, 9), np.uint8), iterations=1)
    valid = cv2.bitwise_not(invalid)

    detector, norm_type, detector_name = _feature_detector()
    kp1, des1 = detector.detectAndCompute(label_gray, valid)
    kp2, des2 = detector.detectAndCompute(object_gray, None)

    if des1 is not None and des2 is not None and len(kp1) >= 8 and len(kp2) >= 8:
        matcher = cv2.BFMatcher(norm_type)
        pairs = matcher.knnMatch(des1, des2, k=2)
        ratio = 0.76 if norm_type == cv2.NORM_L2 else 0.82
        good = [m for m, n in pairs if m.distance < ratio * n.distance]
        if len(good) >= 8:
            src = np.float32([kp1[m.queryIdx].pt for m in good])
            dst = np.float32([kp2[m.trainIdx].pt for m in good])
            partial, partial_mask = cv2.estimateAffinePartial2D(
                src,
                dst,
                method=cv2.RANSAC,
                ransacReprojThreshold=3.0,
                maxIters=12000,
                confidence=0.999,
                refineIters=60,
            )
            full, full_mask = cv2.estimateAffine2D(
                src,
                dst,
                method=cv2.RANSAC,
                ransacReprojThreshold=3.0,
                maxIters=12000,
                confidence=0.999,
                refineIters=60,
            )
            candidates = []
            for transform_name, affine, inlier_mask in (
                ("similarity", partial, partial_mask),
                ("full_affine", full, full_mask),
            ):
                evaluated = _evaluate_affine_candidate(
                    transform_name,
                    affine,
                    inlier_mask,
                    src,
                    dst,
                )
                if evaluated is not None:
                    candidates.append(evaluated)
            selected = _select_affine_candidate(candidates)
            if selected is not None:
                affine, inliers, median_error, transform_name = selected
                result = RegistrationResult(
                    affine=affine.astype(float).tolist(),
                    method=(
                        f"{detector_name}+KNN+RANSAC+{transform_name}"
                    ),
                    matches=len(good),
                    inliers=int(inliers.sum()),
                    inlier_ratio=float(inliers.mean()),
                    median_error=median_error,
                )
                return affine.astype(np.float32), result

    affine = _phase_correlation_fallback(label_gray, object_gray, invalid)
    result = RegistrationResult(
        affine=affine.astype(float).tolist(),
        method="gradient_phase_correlation_fallback",
        matches=0,
        inliers=0,
        inlier_ratio=0.0,
        median_error=999.0,
    )
    return affine, result


def _evaluate_affine_candidate(
    transform_name: str,
    affine: np.ndarray | None,
    inlier_mask: np.ndarray | None,
    src: np.ndarray,
    dst: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, str] | None:
    if affine is None or inlier_mask is None:
        return None
    matrix = np.asarray(affine, dtype=np.float64)
    if matrix.shape != (2, 3) or not np.all(np.isfinite(matrix)):
        return None
    if abs(float(np.linalg.det(matrix[:, :2]))) <= np.finfo(np.float64).eps:
        return None
    inliers = np.asarray(inlier_mask).reshape(-1).astype(bool)
    if len(inliers) != len(src) or int(inliers.sum()) < 6:
        return None
    pred = transform_points(src, matrix)
    errors = np.linalg.norm(pred - dst, axis=1)
    inlier_errors = errors[inliers]
    median_error = (
        float(np.median(inlier_errors))
        if inlier_errors.size
        else 999.0
    )
    return matrix.astype(np.float32), inliers, median_error, transform_name


def _select_affine_candidate(
    candidates: list[tuple[np.ndarray, np.ndarray, float, str]],
) -> tuple[np.ndarray, np.ndarray, float, str] | None:
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (
            int(item[1].sum()),
            float(item[1].mean()),
            -float(item[2]),
        ),
    )


def _phase_correlation_fallback(
    label_gray: np.ndarray,
    object_gray: np.ndarray,
    invalid_mask: np.ndarray,
) -> np.ndarray:
    """Fallback with dimension scaling plus residual translation.

    The previous fallback was translation-only, so a feature-poor image2 at a
    different resolution could never map a mask correctly.  Resize the label
    raster to original dimensions first, then estimate only the remaining
    translation in original coordinates.
    """
    oh, ow = object_gray.shape[:2]
    lh, lw = label_gray.shape[:2]
    scale_x, scale_y = ow / max(1.0, float(lw)), oh / max(1.0, float(lh))
    resized_label = cv2.resize(label_gray, (ow, oh), interpolation=cv2.INTER_AREA)
    resized_invalid = cv2.resize(
        invalid_mask, (ow, oh), interpolation=cv2.INTER_NEAREST
    )
    label_clean = cv2.inpaint(resized_label, resized_invalid, 3, cv2.INPAINT_TELEA)

    g1 = gradient_magnitude(label_clean)
    g2 = gradient_magnitude(object_gray)
    window = cv2.createHanningWindow((ow, oh), cv2.CV_32F)
    shift, _response = cv2.phaseCorrelate(
        g1.astype(np.float32), g2.astype(np.float32), window
    )
    tx, ty = shift
    # Only reject non-finite numerical output; do not impose a translation cap.
    if not np.isfinite(tx) or not np.isfinite(ty):
        tx = ty = 0.0
    return np.array(
        [[scale_x, 0.0, tx], [0.0, scale_y, ty]], dtype=np.float32
    )
