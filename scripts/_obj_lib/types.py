from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

BBox = Tuple[int, int, int, int]


@dataclass
class Annotation:
    object_id: int
    bbox: BBox
    pixel_area: int
    parent_id: Optional[int] = None
    children_ids: Optional[List[int]] = None
    # 处理层级：叶子对象为 0；父对象为 1 + 所有子对象中的最大层级。
    # 因此按 level 升序处理时，子对象一定先于复合父对象。
    level: int = 0
    depth: int = 0
    root_id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "object_id": int(self.object_id),
            "bbox": list(self.bbox),
            "pixel_area": int(self.pixel_area),
            "parent_id": (
                int(self.parent_id) if self.parent_id is not None else None
            ),
            "children_ids": (
                [int(value) for value in self.children_ids]
                if self.children_ids is not None
                else None
            ),
            "level": int(self.level),
            "depth": int(self.depth),
            "root_id": int(self.root_id) if self.root_id is not None else None,
        }


@dataclass
class RegistrationResult:
    affine: List[List[float]]
    method: str
    matches: int
    inliers: int
    inlier_ratio: float
    median_error: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "affine": [
                [float(value) for value in row] for row in self.affine
            ],
            "method": str(self.method),
            "matches": int(self.matches),
            "inliers": int(self.inliers),
            "inlier_ratio": float(self.inlier_ratio),
            "median_error": float(self.median_error),
        }


@dataclass
class LocalizationResult:
    rough_bbox: BBox
    refined_bbox: Optional[BBox]
    score: float
    scale_x: float
    scale_y: float
    search_bbox: Optional[BBox] = None
    status: str = "localized"
    failure_reason: Optional[str] = None
    # 最终精框的轮廓相似度分（top-N 候选最大值），与 score 同源。
    # 融合由轮廓相似度驱动后，score 即 contour_score；单独保留以便审计。
    contour_score: float = 0.0

    @property
    def success(self) -> bool:
        return self.refined_bbox is not None and self.status == "localized"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rough_bbox": list(self.rough_bbox),
            "refined_bbox": (
                list(self.refined_bbox)
                if self.refined_bbox is not None
                else None
            ),
            "search_bbox": (
                list(self.search_bbox)
                if self.search_bbox is not None
                else None
            ),
            "score": float(self.score),
            "scale_x": float(self.scale_x),
            "scale_y": float(self.scale_y),
            "status": str(self.status),
            "failure_reason": self.failure_reason,
            "contour_score": float(self.contour_score),
        }


@dataclass
class SegmentationResult:
    mask_score: float
    sam_score: float
    status: str
    crop_bbox: BBox
    positive_points: List[List[float]]
    negative_points: List[List[float]]
    mask_path: str
    rgba_path: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mask_score": float(self.mask_score),
            "sam_score": float(self.sam_score),
            "status": str(self.status),
            "crop_bbox": list(self.crop_bbox),
            "positive_points": [
                [float(value) for value in point]
                for point in self.positive_points
            ],
            "negative_points": [
                [float(value) for value in point]
                for point in self.negative_points
            ],
            "mask_path": str(self.mask_path),
            "rgba_path": str(self.rgba_path),
        }
