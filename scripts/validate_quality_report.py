#!/usr/bin/env python3
"""Validate a record-only annotation quality report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

VALID_LABELS = {
    "标注合格", "误标", "多标", "边界偏移",
    "掩码不足", "掩码溢出", "粘连", "实例拆分",
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

    for field in ("missing_instances", "excluded_instances", "unexpected_instances"):
        if not isinstance(report.get(field), list):
            raise ValueError(f"report.{field} 必须是数组")
    missing_keys = []
    for index, item in enumerate(report["missing_instances"], start=1):
        if not isinstance(item, dict):
            raise ValueError(f"missing_instances[{index}] 必须是对象")
        key = item.get("instance_key")
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"missing_instances[{index}].instance_key 不能为空")
        missing_keys.append(key)
    if len(missing_keys) != len(set(missing_keys)):
        raise ValueError("missing_instances.instance_key 不得重复")

    unexpected_ids = []
    for value in report["unexpected_instances"]:
        annotation_id = int(
            value.get("annotation_id", -1) if isinstance(value, dict) else value
        )
        if annotation_id not in coco_annotations:
            raise ValueError(f"unexpected_instances 引用不存在的 annotation：{annotation_id}")
        unexpected_ids.append(annotation_id)
    if len(unexpected_ids) != len(set(unexpected_ids)):
        raise ValueError("unexpected_instances 不得重复")

    summary = report.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("report.summary 必须是对象")
    if int(summary.get("extracted_annotation_count", -1)) != len(coco_annotations):
        raise ValueError("summary.extracted_annotation_count 与 COCO 不一致")
    if int(summary.get("excluded_instance_count", -1)) != len(report["excluded_instances"]):
        raise ValueError("summary.excluded_instance_count 与 excluded_instances 不一致")
    labels = summary.get("labels")
    if not isinstance(labels, dict):
        raise ValueError("summary.labels 必须是对象")
    expected_labels: Dict[str, int] = {}
    for item in report_items:
        label = item["evaluation"]["label"]
        expected_labels[label] = expected_labels.get(label, 0) + 1
    if report["missing_instances"]:
        expected_labels["漏标"] = len(report["missing_instances"])
    reported_labels = {str(key): int(value) for key, value in labels.items() if int(value)}
    if reported_labels != expected_labels:
        raise ValueError(
            f"summary.labels 与逐项评价不一致：期望 {expected_labels}，实际 {reported_labels}"
        )
    expected_issue_count = len(report["missing_instances"]) + sum(
        1 for item in report_items
        if item["evaluation"]["label"] != "标注合格"
    )
    if int(summary.get("quality_issue_count", -1)) != expected_issue_count:
        raise ValueError(
            "summary.quality_issue_count 与非合格 annotation 及漏标数量不一致"
        )


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
