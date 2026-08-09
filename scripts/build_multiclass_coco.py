#!/usr/bin/env python3
"""Sequentially combine one-image2-output-per-category into final COCO data."""

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


def _simplified_coco(coco: Dict[str, Any]) -> Dict[str, Any]:
    """返回 COCO 的简化副本：每个 annotation 去掉 area/iscrowd/segmentation。

    直接由完整 annotations.json 派生，作为 annotation-summary.json 的内容。
    """
    drop = ("area", "iscrowd", "segmentation")
    return {
        "images": coco.get("images", []),
        "categories": coco.get("categories", []),
        "annotations": [
            {k: v for k, v in ann.items() if k not in drop}
            for ann in coco.get("annotations", [])
        ],
    }


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

    seen_names = set()
    for index, item in enumerate(manifest["categories"], start=1):
        name = str(item.get("name", "")).strip()
        if not name:
            raise ValueError(f"第 {index} 个类别缺少 name")
        if name in seen_names:
            raise ValueError(f"manifest 中类别重复：{name}")
        seen_names.add(name)
        labeled_value = item["labeled"]
        labeled_names = labeled_value if isinstance(labeled_value, list) else [labeled_value]
        if not labeled_names:
            raise ValueError(f"类别 {name} 的 labeled 不能为空")
        require_single = bool(item.get("require_single_instance", isinstance(labeled_value, list)))
        refine_edges = bool(item.get("refine_edges", True))
        max_local_shift = int(item.get("max_local_shift", 8))
        fail_on_local_limit = bool(item.get("fail_on_local_limit", True))
        masks = []
        for labeled_name in labeled_names:
            labeled_path = (base / labeled_name).resolve()
            if not labeled_path.is_file():
                raise FileNotFoundError(f"类别 {name} 的标注图不存在：{labeled_path}")
            labeled = blocks["read_bgr"](labeled_path)
            file_masks, _, details = extractor._extract_registered_masks_detailed(
                blocks,
                labeled,
                original,
                refine_edges=refine_edges,
                max_local_shift=max_local_shift,
            )
            if require_single and len(file_masks) != 1:
                raise ValueError(
                    f"类别 {name} 的实例级标注 {labeled_path.name} 应包含 1 个红色实例，"
                    f"实际提取 {len(file_masks)} 个；请定向重做该实例"
                )
            if fail_on_local_limit and any(
                entry.get("at_search_limit", False)
                for entry in details.get("local_refinement", [])
            ):
                raise ValueError(
                    f"类别 {name} 的 {labeled_path.name} 局部校正到达 "
                    f"±{max_local_shift}px 搜索边界；视为配准失败，请定向重做，"
                    "不要扩大搜索范围掩盖偏移"
                )
            masks.extend(file_masks)
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
        print(f"[类别 {index}/{len(manifest['categories'])}] {name}(id={category_id}): {accepted} 个实例")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(coco, fh, ensure_ascii=False, indent=2)
    return {
        "image": str(image_path),
        "output": str(output_path),
        "annotation_count": len(coco["annotations"]),
        "simplified": _simplified_coco(coco),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="按类别串行合并 image2 实心红色标注图，生成一个最终 COCO JSON"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--summary", type=Path, default=None,
        help="简化标注 JSON（annotations.json 去掉 area/iscrowd/segmentation）输出路径",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_from_manifest(args.manifest.resolve(), args.output.resolve())
    if args.summary:
        summary_path = args.summary.resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as fh:
            json.dump(result["simplified"], fh, ensure_ascii=False, indent=2)
        print(f"[摘要] 简化标注 -> {summary_path}")
    print(f"[最终 COCO] {result['annotation_count']} 个实例 -> {result['output']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
