#!/usr/bin/env python3
"""Combine one-time image2 label outputs into a multi-class COCO file."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _obj_lib.annotation_colors import normalize_annotation_color

_SPEC = importlib.util.spec_from_file_location(
    "extract_object_annotations", SCRIPT_DIR / "extract_object_annotations.py"
)
extractor = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
_SPEC.loader.exec_module(extractor)


def _load_manifest(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data.get("categories"), list) or not data["categories"]:
        raise ValueError("manifest.categories 必须是非空数组")
    return data


def _label_paths(value: Any) -> List[str]:
    names = value if isinstance(value, list) else [value]
    if names == []:
        return []
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("labeled 必须是字符串、字符串数组或表示跳过类别的空数组")
    for name in names:
        normalized = Path(name.replace("\\\\", "/"))
        if not normalized.parts or normalized.parts[0] != "image2-labels":
            raise ValueError(f"image2 原始标注图必须位于 image2-labels/：{name}")
    return names


def build_from_manifest(manifest_path: Path, output_path: Path) -> Dict[str, Any]:
    manifest = _load_manifest(manifest_path)
    base = manifest_path.resolve().parent
    image_path = (base / manifest["image"]).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"原图不存在：{image_path}")

    blocks = extractor._import_building_blocks()
    original = blocks["read_bgr"](image_path)
    height, width = original.shape[:2]
    image_id = int(manifest.get("image_id", 1))
    min_pixels = int(manifest.get("min_mask_pixels", 6))
    coco: Dict[str, Any] = {"images": [], "categories": [], "annotations": []}
    extraction_files: List[Dict[str, Any]] = []
    category_counts: Dict[str, int] = {}

    seen_names = set()
    for index, item in enumerate(manifest["categories"], start=1):
        name = str(item.get("name", "")).strip()
        if not name:
            raise ValueError(f"第 {index} 个类别缺少 name")
        if name in seen_names:
            raise ValueError(f"manifest 中类别重复：{name}")
        seen_names.add(name)

        masks = []
        try:
            annotation_color = normalize_annotation_color(
                item.get("annotation_color", "red")
            )
        except ValueError as exc:
            raise ValueError(f"类别 {name} 的 annotation_color 无效：{exc}") from exc
        annotation_color_reason = str(item.get("annotation_color_reason", "")).strip()
        labeled_names = _label_paths(item.get("labeled"))
        excluded_instances = item.get("excluded_instances", [])
        if not isinstance(excluded_instances, list):
            raise ValueError(f"类别 {name} 的 excluded_instances 必须是数组")
        if not labeled_names and not excluded_instances:
            raise ValueError(
                f"类别 {name} 使用空 labeled 时必须记录 excluded_instances"
            )
        for labeled_name in labeled_names:
            labeled_path = (base / labeled_name).resolve()
            if not labeled_path.is_file():
                raise FileNotFoundError(f"类别 {name} 的标注图不存在：{labeled_path}")
            labeled = blocks["read_bgr"](labeled_path)
            file_masks, _, details = extractor._extract_registered_masks_detailed(
                blocks, labeled, original, annotation_color
            )
            masks.extend(file_masks)
            extraction_files.append({
                "category": name,
                "annotation_color": annotation_color,
                "annotation_color_reason": annotation_color_reason,
                "labeled": labeled_name,
                "extracted_instances": len(file_masks),
                "registration": details["global"],
                "mask_extraction": details["mask_extraction"],
                "label_size": details["label_size"],
                "original_size": details["original_size"],
            })

        requested_id = item.get("id")
        category_id, accepted = extractor._build_coco(
            coco,
            image_id=image_id,
            file_name=image_path.name,
            width=width,
            height=height,
            object_name=name,
            category_id_override=int(requested_id) if requested_id is not None else None,
            masks=masks,
            min_mask_pixels=min_pixels,
        )
        category_counts[name] = accepted
        if not labeled_names:
            extraction_files.append({
                "category": name,
                "annotation_color": annotation_color,
                "annotation_color_reason": annotation_color_reason,
                "labeled": [],
                "extracted_instances": 0,
                "skipped": True,
                "excluded_instances": excluded_instances,
            })
        print(
            f"[类别 {index}/{len(manifest['categories'])}] "
            f"{name}(id={category_id}): 提取 {accepted} 个实例"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(coco, fh, ensure_ascii=False, indent=2)

    summary = {
        "image": str(image_path),
        "annotation_file": str(output_path),
        "category_count": len(coco["categories"]),
        "annotation_count": len(coco["annotations"]),
        "category_instance_counts": category_counts,
        "image2_outputs": extraction_files,
        "excluded_instance_count": sum(
            len(item.get("excluded_instances", []))
            for item in manifest["categories"]
        ),
    }
    return {"output": str(output_path), "summary": summary}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="合并 image2 单次标注输出；不强制实例数量，不执行仿射后修复"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_from_manifest(args.manifest.resolve(), args.output.resolve())
    if args.summary:
        summary_path = args.summary.resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as fh:
            json.dump(result["summary"], fh, ensure_ascii=False, indent=2)
        print(f"[摘要] -> {summary_path}")
    print(f"[COCO] -> {result['output']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
