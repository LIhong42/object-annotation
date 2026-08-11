#!/usr/bin/env python3
"""Merge audited single-category COCO files into one deterministic dataset."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from validate_eligibility_report import validate as validate_eligibility
from validate_quality_report import validate as validate_quality


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        value = json.load(fh)
    if not isinstance(value, dict):
        raise ValueError(f"JSON 顶层必须是对象：{path}")
    return value


def _resolve(base: Path, value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} 必须是非空路径")
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build(manifest_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = manifest_path.resolve()
    manifest = _load(manifest_path)
    base = manifest_path.parent
    image_path = _resolve(base, manifest.get("image"), "manifest.image")
    if not image_path.is_file():
        raise FileNotFoundError(f"原图不存在：{image_path}")
    image_id = int(manifest.get("image_id", 1))
    categories = manifest.get("categories")
    if not isinstance(categories, list) or not categories:
        raise ValueError("manifest.categories 必须是非空数组")

    merged: dict[str, Any] = {"images": [], "categories": [], "annotations": []}
    quality_index: dict[str, Any] = {
        "schema_version": "1.0",
        "image": str(image_path),
        "categories": [],
        "summary": {},
    }
    category_ids: set[int] = set()
    category_names: set[str] = set()
    canonical_image: dict[str, Any] | None = None
    next_annotation_id = 1
    eligible_count = 0

    for index, entry in enumerate(categories, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"categories[{index}] 必须是对象")
        category_id = int(entry.get("id"))
        name = str(entry.get("name", "")).strip()
        status = entry.get("status")
        if category_id < 1 or not name:
            raise ValueError(f"categories[{index}] 的 id/name 无效")
        if category_id in category_ids or name in category_names:
            raise ValueError(f"类别 id 或 name 重复：{category_id}/{name}")
        if status not in {"eligible", "skipped"}:
            raise ValueError(f"类别 {name} 的 status 必须为 eligible 或 skipped")
        category_ids.add(category_id)
        category_names.add(name)
        merged["categories"].append(
            {"id": category_id, "name": name, "supercategory": name}
        )

        eligibility_path = _resolve(
            base, entry.get("eligibility_report"), f"{name}.eligibility_report"
        )
        eligibility = _load(eligibility_path)
        validate_eligibility(eligibility)
        if eligibility.get("target_category") != name or eligibility.get("status") != status:
            raise ValueError(f"类别 {name} 与 eligibility report 不一致")

        quality_entry: dict[str, Any] = {
            "category_id": category_id,
            "category_name": name,
            "status": status,
            "eligibility_report": str(eligibility_path),
            "annotation_count": 0,
        }
        if status == "skipped":
            quality_index["categories"].append(quality_entry)
            continue

        annotation_path = _resolve(base, entry.get("annotations"), f"{name}.annotations")
        quality_path = _resolve(base, entry.get("quality_report"), f"{name}.quality_report")
        coco = _load(annotation_path)
        report = _load(quality_path)
        validate_quality(coco, report)
        if len(coco.get("categories", [])) != 1:
            raise ValueError(f"{name} 的输入 COCO 必须只含一个类别")
        source_category = coco["categories"][0]
        if str(source_category.get("name")) != name:
            raise ValueError(f"{name} 的 COCO 类别名不一致")
        images = coco.get("images")
        if not isinstance(images, list) or len(images) != 1:
            raise ValueError(f"{name} 的输入 COCO 必须只含一张图片")
        source_image = images[0]
        candidate = {
            "id": image_id,
            "file_name": image_path.name,
            "width": int(source_image["width"]),
            "height": int(source_image["height"]),
        }
        if canonical_image is None:
            canonical_image = candidate
        elif (
            candidate["width"] != canonical_image["width"]
            or candidate["height"] != canonical_image["height"]
        ):
            raise ValueError(f"{name} 的图片尺寸与其他类别不一致")

        source_category_id = int(source_category["id"])
        for annotation in coco.get("annotations", []):
            if int(annotation.get("category_id")) != source_category_id:
                raise ValueError(f"{name} 包含非本类别 annotation")
            item = dict(annotation)
            item["id"] = next_annotation_id
            item["image_id"] = image_id
            item["category_id"] = category_id
            merged["annotations"].append(item)
            next_annotation_id += 1
        annotation_count = len(coco.get("annotations", []))
        eligible_count += 1
        quality_entry.update({
            "annotations": str(annotation_path),
            "quality_report": str(quality_path),
            "annotation_count": annotation_count,
            "quality_summary": report.get("summary", {}),
        })
        quality_index["categories"].append(quality_entry)

    if canonical_image is None:
        from PIL import Image
        with Image.open(image_path) as image:
            width, height = image.size
        canonical_image = {
            "id": image_id,
            "file_name": image_path.name,
            "width": width,
            "height": height,
        }
    merged["images"] = [canonical_image]
    quality_index["summary"] = {
        "category_count": len(categories),
        "eligible_category_count": eligible_count,
        "skipped_category_count": len(categories) - eligible_count,
        "annotation_count": len(merged["annotations"]),
    }
    return merged, quality_index


def main() -> int:
    parser = argparse.ArgumentParser(description="合并单类别 COCO 与质量索引")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quality-index", type=Path, required=True)
    args = parser.parse_args()
    merged, quality_index = build(args.manifest)
    _atomic_json(args.output.resolve(), merged)
    _atomic_json(args.quality_index.resolve(), quality_index)
    print(
        f"[完成] {len(merged['categories'])} 个类别、"
        f"{len(merged['annotations'])} 个实例 -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
