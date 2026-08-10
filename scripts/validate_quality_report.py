#!/usr/bin/env python3
"""Validate a record-only annotation quality report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

VALID_LABELS = {
    "标注合格", "误标", "多标", "边界偏移",
    "掩码不足", "掩码溢出", "粘连",
}
FORBIDDEN_FIELDS = {"retry", "replacement", "final_evaluation", "retry_plan"}


def load(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _reject_repair_fields(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN_FIELDS or key.startswith("retry_"):
                raise ValueError(f"{path}.{key}: 质量报告不得包含修复或重标字段")
            _reject_repair_fields(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_repair_fields(child, f"{path}[{index}]")


def validate(coco: Dict[str, Any], report: Dict[str, Any]) -> None:
    _reject_repair_fields(report)
    coco_annotations = {int(item["id"]): item for item in coco.get("annotations", [])}
    report_items = report.get("annotations")
    if not isinstance(report_items, list):
        raise ValueError("report.annotations 必须是数组")

    seen = set()
    for index, item in enumerate(report_items, start=1):
        annotation_id = int(item.get("annotation_id", -1))
        if annotation_id not in coco_annotations:
            raise ValueError(f"第 {index} 项 annotation_id 不存在：{annotation_id}")
        if annotation_id in seen:
            raise ValueError(f"annotation_id 重复评价：{annotation_id}")
        seen.add(annotation_id)
        evaluation = item.get("evaluation") or {}
        label = evaluation.get("label")
        if label not in VALID_LABELS:
            raise ValueError(f"annotation_id={annotation_id}: 非法标签 {label}")
        if bool(evaluation.get("passed")) != (label == "标注合格"):
            raise ValueError(f"annotation_id={annotation_id}: passed 与 label 不一致")

    if seen != set(coco_annotations):
        missing = sorted(set(coco_annotations) - seen)
        raise ValueError(f"COCO annotation 尚未评价：{missing}")

    for field in ("missing_instances", "unexpected_instances"):
        if not isinstance(report.get(field), list):
            raise ValueError(f"report.{field} 必须是数组")
    if not isinstance(report.get("summary"), dict):
        raise ValueError("report.summary 必须是对象")


def main() -> int:
    parser = argparse.ArgumentParser(description="校验只记录、不修复的质量报告")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    validate(load(args.annotations), load(args.report))
    print("[通过] 质量报告结构有效，且不包含重标或修复字段")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
