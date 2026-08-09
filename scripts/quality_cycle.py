#!/usr/bin/env python3
"""Validate visual-QA reports and apply at-most-once instance replacements."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent

def load_extractor():
    """Load OpenCV-dependent extraction code only for the apply command."""
    spec = importlib.util.spec_from_file_location("extractor", SCRIPT_DIR / "extract_object_annotations.py")
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module

VALID_LABELS = {"标注合格", "漏标", "多标", "边界偏移", "掩码不足", "掩码溢出"}

def load(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)

def validate(coco: Dict[str, Any], report: Dict[str, Any], final: bool = False) -> None:
    anns = {int(a["id"]): a for a in coco.get("annotations", [])}
    items = report.get("instances")
    if not isinstance(items, list) or not items:
        raise ValueError("report.instances 必须是非空数组")
    keys = set()
    referenced = set()
    for i, item in enumerate(items, 1):
        key = str(item.get("instance_key", "")).strip()
        if not key or key in keys:
            raise ValueError(f"第 {i} 项 instance_key 缺失或重复")
        keys.add(key)
        ev = item.get("initial_evaluation") or {}
        label = ev.get("label")
        if label not in VALID_LABELS:
            raise ValueError(f"{key}: 非法 initial_evaluation.label={label}")
        if bool(ev.get("passed")) != (label == "标注合格"):
            raise ValueError(f"{key}: passed 与 label 不一致")
        ann_id = item.get("annotation_id")
        if label == "漏标":
            if ann_id is not None:
                raise ValueError(f"{key}: 漏标项 annotation_id 必须为 null")
        else:
            if ann_id is None or int(ann_id) not in anns:
                raise ValueError(f"{key}: annotation_id 不存在")
            if int(ann_id) in referenced:
                raise ValueError(f"annotation_id={ann_id} 被重复评价")
            referenced.add(int(ann_id))
        retry = item.get("retry") or {}
        count = int(retry.get("retry_count", 0))
        if count not in (0, 1):
            raise ValueError(f"{key}: retry_count 只能为 0 或 1")
        if ev.get("passed") and (retry.get("attempted") or count):
            raise ValueError(f"{key}: 合格实例不得重标")
        if bool(retry.get("attempted")) != (count == 1):
            raise ValueError(f"{key}: attempted 与 retry_count 不一致")
        if count == 1 and not retry.get("labeled"):
            raise ValueError(f"{key}: 已重标但缺少 retry.labeled")
        if final:
            fe = item.get("final_evaluation")
            if count == 1:
                if not fe or fe.get("label") not in VALID_LABELS:
                    raise ValueError(f"{key}: 缺少合法 final_evaluation")
            elif fe is not None:
                raise ValueError(f"{key}: 未重标实例 final_evaluation 应为 null")
    if referenced != set(anns):
        missing = sorted(set(anns) - referenced)
        raise ValueError(f"已有 annotations 未逐实例评价: {missing}")

def retry_plan(report: Dict[str, Any]) -> Dict[str, Any]:
    items = []
    for x in report["instances"]:
        if not x["initial_evaluation"]["passed"]:
            items.append({
                "instance_key": x["instance_key"],
                "annotation_id": x.get("annotation_id"),
                "description": x.get("description"),
                "location": x.get("location"),
                "reason": x["initial_evaluation"]["label"],
                "max_retries": 1,
            })
    return {"retry_count": len(items), "instances": items}

def apply(image: Path, coco: Dict[str, Any], report: Dict[str, Any], report_base: Path) -> Dict[str, Any]:
    extractor = load_extractor()
    validate(coco, report, final=False)
    out = deepcopy(coco)
    ann_by_id = {int(a["id"]): a for a in out["annotations"]}
    blocks = extractor._import_building_blocks()
    original = blocks["read_bgr"](image)
    next_id = max(ann_by_id, default=0) + 1
    for item in report["instances"]:
        retry = item.get("retry") or {}
        if int(retry.get("retry_count", 0)) != 1:
            continue
        labeled = (report_base / retry["labeled"]).resolve()
        label_img = blocks["read_bgr"](labeled)
        masks, _, _ = extractor._extract_registered_masks_detailed(
            blocks, label_img, original, refine_edges=True, max_local_shift=5
        )
        if len(masks) != 1:
            raise ValueError(f"{item['instance_key']}: 重标图必须且只能提取 1 个实例，实际 {len(masks)}")
        data = extractor._mask_to_coco(masks[0])
        if not data["segmentation"]:
            raise ValueError(f"{item['instance_key']}: 重标掩码退化")
        old_id = item.get("annotation_id")
        if old_id is None:
            ann_id = next_id; next_id += 1
            cat_id = int(report["category"]["id"])
            ann = {"id": ann_id, "image_id": int(out["images"][0]["id"]), "category_id": cat_id, "iscrowd": 0}
            out["annotations"].append(ann)
        else:
            ann_id = int(old_id)
            ann = ann_by_id[ann_id]
        ann.update(data)
        retry["replacement_annotation_id"] = ann_id
    out["annotations"].sort(key=lambda a: int(a["id"]))
    return out

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate"); v.add_argument("--annotations", type=Path, required=True); v.add_argument("--report", type=Path, required=True); v.add_argument("--final", action="store_true")
    q = sub.add_parser("plan"); q.add_argument("--report", type=Path, required=True); q.add_argument("--output", type=Path, required=True)
    a = sub.add_parser("apply"); a.add_argument("--image", type=Path, required=True); a.add_argument("--annotations", type=Path, required=True); a.add_argument("--report", type=Path, required=True); a.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    if args.cmd == "validate": validate(load(args.annotations), load(args.report), args.final); print("[通过] 质量报告结构与重标次数合法")
    elif args.cmd == "plan": dump(args.output, retry_plan(load(args.report))); print(f"[完成] 重标清单 -> {args.output}")
    else: dump(args.output, apply(args.image.resolve(), load(args.annotations), load(args.report), args.report.resolve().parent)); print(f"[完成] 定向替换 -> {args.output}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
