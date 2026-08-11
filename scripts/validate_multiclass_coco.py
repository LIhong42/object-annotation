#!/usr/bin/env python3
"""Validate merged COCO structure and quality-index coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        value = json.load(fh)
    if not isinstance(value, dict):
        raise ValueError(f"JSON 顶层必须是对象：{path}")
    return value


def _unique_ids(items: Any, label: str) -> set[int]:
    if not isinstance(items, list):
        raise ValueError(f"{label} 必须是数组")
    ids = [int(item["id"]) for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{label} 包含重复 id")
    return set(ids)


def validate(coco: dict[str, Any], quality: dict[str, Any]) -> None:
    image_ids = _unique_ids(coco.get("images"), "images")
    category_ids = _unique_ids(coco.get("categories"), "categories")
    _unique_ids(coco.get("annotations"), "annotations")
    if len(image_ids) != 1:
        raise ValueError("当前多类别交付必须只含一张图片")
    for index, annotation in enumerate(coco["annotations"], start=1):
        if int(annotation.get("image_id", -1)) not in image_ids:
            raise ValueError(f"annotation[{index}] image_id 不存在")
        if int(annotation.get("category_id", -1)) not in category_ids:
            raise ValueError(f"annotation[{index}] category_id 不存在")
        bbox = annotation.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
            raise ValueError(f"annotation[{index}] bbox 无效")
        if float(annotation.get("area", 0)) <= 0:
            raise ValueError(f"annotation[{index}] area 无效")
        segmentation = annotation.get("segmentation")
        if not isinstance(segmentation, list) or not segmentation:
            raise ValueError(f"annotation[{index}] segmentation 为空")

    entries = quality.get("categories")
    if not isinstance(entries, list):
        raise ValueError("quality-index.categories 必须是数组")
    indexed_ids = {int(item["category_id"]) for item in entries}
    if indexed_ids != category_ids or len(entries) != len(indexed_ids):
        raise ValueError("quality index 必须且只能覆盖全部 COCO 类别")
    actual = {category_id: 0 for category_id in category_ids}
    for annotation in coco["annotations"]:
        actual[int(annotation["category_id"])] += 1
    for entry in entries:
        category_id = int(entry["category_id"])
        if entry.get("status") not in {"eligible", "skipped"}:
            raise ValueError(f"category_id={category_id} 的 status 无效")
        if int(entry.get("annotation_count", -1)) != actual[category_id]:
            raise ValueError(f"category_id={category_id} 的 annotation_count 不一致")
        if entry.get("status") == "skipped" and actual[category_id] != 0:
            raise ValueError(f"skipped 类别 {category_id} 不得含 annotation")
    summary = quality.get("summary", {})
    if int(summary.get("annotation_count", -1)) != len(coco["annotations"]):
        raise ValueError("quality-index summary.annotation_count 不一致")


def main() -> int:
    parser = argparse.ArgumentParser(description="校验合并后的 COCO 与质量索引")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--quality-index", type=Path, required=True)
    args = parser.parse_args()
    validate(_load(args.annotations), _load(args.quality_index))
    print("[通过] 多类别 COCO 结构及质量索引覆盖有效")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
