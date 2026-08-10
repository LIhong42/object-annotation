#!/usr/bin/env python3
"""把 image2 的实心红色实例区域直接映射为 COCO 标注。

每次调用只处理一种对象类别；多类别标注通过 ``--append`` 依次累加
到同一个 COCO JSON。

用法示例::

    python extract_object_annotations.py \
        --image ori.png \
        --labeled object-labeling.png \
        --object-name "person" \
        --output annotations.json

    # 第二个类别追加进同一个 JSON
    python extract_object_annotations.py \
        --image ori.png \
        --labeled dog-labeling.png \
        --object-name "dog" \
        --output annotations.json \
        --append
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# ----------------------------------------------------------------------------
# 依赖：本脚本同目录下的内嵌库 _obj_lib（从 png_to_pptx 对象提取链路抽取的
# 最小构建块）。把脚本所在目录加入 sys.path，使脚本可独立于原项目运行。
# ----------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))


def _import_building_blocks():
    from _obj_lib.filled_mask import (
        extract_filled_instances,
        map_masks_to_original,
    )
    from _obj_lib.registration import estimate_global_affine
    from _obj_lib.utils import read_bgr
    return {
        "extract_filled_instances": extract_filled_instances,
        "map_masks_to_original": map_masks_to_original,
        "estimate_global_affine": estimate_global_affine,
        "read_bgr": read_bgr,
    }


# ----------------------------------------------------------------------------
# 红色填充区域 -> 原图掩码
# ----------------------------------------------------------------------------

def _extract_registered_masks(
    blocks: Dict[str, Any],
    labeling_bgr: np.ndarray,
    original_bgr: np.ndarray,
) -> Tuple[List[np.ndarray], np.ndarray]:
    mapped, affine, _ = _extract_registered_masks_detailed(
        blocks, labeling_bgr, original_bgr
    )
    return mapped, affine


def _extract_registered_masks_detailed(
    blocks: Dict[str, Any],
    labeling_bgr: np.ndarray,
    original_bgr: np.ndarray,
) -> Tuple[List[np.ndarray], np.ndarray, Dict[str, Any]]:
    """Map solid-red masks using the first global affine result, unchanged."""
    estimate_global_affine = blocks["estimate_global_affine"]
    from _obj_lib.filled_mask import solid_red_pixels
    provisional_red_mask = solid_red_pixels(labeling_bgr).astype(np.uint8) * 255
    affine, registration = estimate_global_affine(
        labeling_bgr, original_bgr, provisional_red_mask
    )
    label_masks, _ = blocks["extract_filled_instances"](
        labeling_bgr, original_bgr, affine
    )
    oh, ow = original_bgr.shape[:2]
    mapped = blocks["map_masks_to_original"](
        label_masks, affine, ow, oh
    )
    details = {
        "global": registration.to_dict(),
        "label_size": [int(labeling_bgr.shape[1]), int(labeling_bgr.shape[0])],
        "original_size": [int(ow), int(oh)],
    }
    return mapped, affine, details


# ----------------------------------------------------------------------------
# 掩码 -> COCO
# ----------------------------------------------------------------------------

def _mask_to_coco(mask: np.ndarray) -> Dict[str, Any]:
    """把二值掩码转为 COCO 的 bbox / area / segmentation(polygon) 三元组。"""
    mask_bbox = _mask_bbox_local(mask)
    if mask_bbox is None:
        return {"bbox": [0, 0, 0, 0], "area": 0, "segmentation": []}
    x1, y1, x2, y2 = mask_bbox
    bbox = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
    area = int((mask > 0).sum())
    segmentation = _mask_to_polygons(mask)
    return {"bbox": bbox, "area": area, "segmentation": segmentation}


def _mask_bbox_local(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _mask_to_polygons(mask: np.ndarray) -> List[List[float]]:
    """掩码 -> COCO 多边形列表（每个外部轮廓一条多边形，>=3 个顶点）。"""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    polygons: List[List[float]] = []
    for contour in contours:
        pts = contour.reshape(-1, 2)
        if len(pts) < 3:
            continue
        polygons.append(pts.flatten().astype(float).tolist())
    return polygons


# ----------------------------------------------------------------------------
# COCO JSON 装配 / 追加
# ----------------------------------------------------------------------------

def _load_existing_coco(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {"images": [], "categories": [], "annotations": []}
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    data.setdefault("images", [])
    data.setdefault("categories", [])
    data.setdefault("annotations", [])
    return data


def _next_id(items: List[Dict[str, Any]], key: str) -> int:
    return max((int(item.get(key, 0)) for item in items), default=0) + 1


def _ensure_image_entry(
    coco: Dict[str, Any],
    image_id: int,
    file_name: str,
    width: int,
    height: int,
) -> int:
    for image in coco["images"]:
        if int(image.get("id")) == int(image_id):
            return int(image["id"])
    coco["images"].append(
        {
            "id": int(image_id),
            "file_name": str(file_name),
            "width": int(width),
            "height": int(height),
        }
    )
    return int(image_id)


def _ensure_category_entry(
    coco: Dict[str, Any],
    name: str,
    category_id: Optional[int],
) -> int:
    # 同名类别复用，避免重复标注同一类别时产生重复 category。
    for cat in coco["categories"]:
        if str(cat.get("name")) == str(name):
            return int(cat["id"])
    if category_id is None:
        category_id = _next_id(coco["categories"], "id")
    # 若指定 id 已被别的类别占用，则自动取下一个可用 id。
    existing_ids = {int(c.get("id")) for c in coco["categories"]}
    while int(category_id) in existing_ids:
        category_id = int(category_id) + 1
    coco["categories"].append(
        {
            "id": int(category_id),
            "name": str(name),
            "supercategory": str(name),
        }
    )
    return int(category_id)


def _build_coco(
    coco: Dict[str, Any],
    *,
    image_id: int,
    file_name: str,
    width: int,
    height: int,
    object_name: str,
    category_id_override: Optional[int],
    masks: List[np.ndarray],
    min_mask_pixels: int,
) -> Tuple[int, int]:
    """把掩码写进 COCO 结构，返回 (category_id, 实例数)。"""
    resolved_image_id = _ensure_image_entry(coco, image_id, file_name, width, height)
    category_id = _ensure_category_entry(coco, object_name, category_id_override)

    next_ann_id = _next_id(coco["annotations"], "id")
    accepted = 0
    for mask in masks:
        if int((mask > 0).sum()) < int(min_mask_pixels):
            continue
        coco_data = _mask_to_coco(mask)
        if not coco_data["segmentation"]:
            # 没有可用多边形（掩码退化）则跳过，保证 COCO 合法。
            continue
        coco["annotations"].append(
            {
                "id": next_ann_id,
                "image_id": resolved_image_id,
                "category_id": category_id,
                "bbox": coco_data["bbox"],
                "area": coco_data["area"],
                "iscrowd": 0,
                "segmentation": coco_data["segmentation"],
            }
        )
        next_ann_id += 1
        accepted += 1
    return category_id, accepted


def remove_category_annotations(
    coco: Dict[str, Any], *, image_id: int, category_id: int
) -> int:
    """Remove an existing image/category slice before deterministically rebuilding it."""
    before = len(coco.get("annotations", []))
    coco["annotations"] = [
        ann
        for ann in coco.get("annotations", [])
        if not (
            int(ann.get("image_id", -1)) == int(image_id)
            and int(ann.get("category_id", -1)) == int(category_id)
        )
    ]
    return before - len(coco["annotations"])


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "对象标注提取：原始图 + 实心红色对象掩码图 -> COCO JSON。"
            "红色不规则区域直接映射为实例掩码，不运行 SAM。"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image", "-i", type=Path, required=True, help="原始输入图片路径"
    )
    parser.add_argument(
        "--labeled",
        "-l",
        type=Path,
        required=True,
        help="对象被纯红色完整填充的图片路径（image2）",
    )
    parser.add_argument(
        "--object-name",
        type=str,
        required=True,
        help="本次标注的对象/类别名称（每次只处理一种对象）",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="输出 COCO JSON 路径",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="追加到已有 COCO JSON（多类别依次标注时，第二个类别起启用）",
    )
    parser.add_argument(
        "--replace-category",
        action="store_true",
        help=(
            "与 --append 配合：先删除当前 image/category 的旧 annotations 再写入，"
            "使同一类别重跑不会产生重复标注"
        ),
    )
    parser.add_argument(
        "--image-id",
        type=int,
        default=1,
        help="本图片在 COCO 中的 image id",
    )
    parser.add_argument(
        "--category-id",
        type=int,
        default=None,
        help="指定类别 id；默认自动分配。同名类别会自动复用",
    )
    parser.add_argument(
        "--min-mask-pixels",
        type=int,
        default=6,
        help="掩码像素少于此值则丢弃，避免退化标注",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    image_path = args.image.resolve()
    labeled_path = args.labeled.resolve()
    output_path = args.output.resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"原始图片不存在：{image_path}")
    if not labeled_path.is_file():
        raise FileNotFoundError(f"标注图片不存在：{labeled_path}")

    blocks = _import_building_blocks()

    original_bgr = blocks["read_bgr"](image_path)
    labeling_bgr = blocks["read_bgr"](labeled_path)
    oh, ow = original_bgr.shape[:2]

    # 1) 配准并把 image2 的实心红色实例区域直接映射到原图。
    masks, _affine = _extract_registered_masks(
        blocks, labeling_bgr, original_bgr
    )
    print(
        f"[掩码] 检测并映射 {len(masks)} 个实心红色实例"
    )
    # 2) 掩码的最小外接矩形直接作为 bbox，并装配/追加 COCO JSON。
    coco = (
        _load_existing_coco(output_path)
        if args.append
        else {"images": [], "categories": [], "annotations": []}
    )
    # 先确保类别存在，才能按稳定 category_id 替换该切片。
    category_id = _ensure_category_entry(coco, args.object_name, args.category_id)
    if args.replace_category:
        remove_category_annotations(
            coco, image_id=args.image_id, category_id=category_id
        )
    category_id, accepted = _build_coco(
        coco,
        image_id=args.image_id,
        file_name=image_path.name,
        width=ow,
        height=oh,
        object_name=args.object_name,
        category_id_override=category_id,
        masks=masks,
        min_mask_pixels=args.min_mask_pixels,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(coco, fh, ensure_ascii=False, indent=2)

    print(
        f"[完成] 类别 '{args.object_name}' (id={category_id}) "
        f"写入 {accepted} 个实例 -> {output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
