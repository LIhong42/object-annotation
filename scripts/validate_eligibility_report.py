#!/usr/bin/env python3
"""Validate the fail-closed category eligibility decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


VALID_STATUS = {"eligible", "skipped"}


def load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        value = json.load(fh)
    if not isinstance(value, dict):
        raise ValueError("eligibility report 顶层必须是对象")
    return value


def _nonempty_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} 必须是非空字符串")
    return value.strip()


def validate(report: dict[str, Any]) -> None:
    _nonempty_text(report.get("image"), "image")
    _nonempty_text(report.get("target_category"), "target_category")
    status = report.get("status")
    if status not in VALID_STATUS:
        raise ValueError("status 必须为 eligible 或 skipped")

    count = report.get("observed_instance_count")
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("observed_instance_count 必须是大于等于 1 的整数")

    separated = report.get("all_instances_separated")
    connected = report.get("all_instances_connected")
    if not isinstance(separated, bool) or not isinstance(connected, bool):
        raise ValueError(
            "all_instances_separated 和 all_instances_connected 必须是布尔值"
        )

    inventory = report.get("target_inventory")
    exclusions = report.get("exclusions")
    blocking = report.get("blocking_relations")
    if not isinstance(inventory, list) or not isinstance(exclusions, list):
        raise ValueError("target_inventory 和 exclusions 必须是数组")
    if not isinstance(blocking, list):
        raise ValueError("blocking_relations 必须是数组")

    keys: set[str] = set()
    for index, item in enumerate(inventory, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"target_inventory[{index}] 必须是对象")
        key = _nonempty_text(item.get("instance_key"), f"target_inventory[{index}].instance_key")
        if key in keys:
            raise ValueError(f"target_inventory instance_key 重复：{key}")
        keys.add(key)
        _nonempty_text(item.get("description"), f"target_inventory[{index}].description")
        _nonempty_text(item.get("location"), f"target_inventory[{index}].location")

    for index, item in enumerate(exclusions, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"exclusions[{index}] 必须是对象")
        _nonempty_text(item.get("description"), f"exclusions[{index}].description")
        _nonempty_text(item.get("reason"), f"exclusions[{index}].reason")

    if status == "eligible":
        if not separated or not connected:
            raise ValueError("eligible 要求所有实例彼此分离且各自可见区域连通")
        if blocking:
            raise ValueError("eligible 的 blocking_relations 必须为空")
        if len(inventory) != count:
            raise ValueError(
                "eligible 的 target_inventory 数量必须等于 observed_instance_count"
            )
    else:
        if separated and connected:
            raise ValueError("skipped 必须至少有一个门禁布尔值为 false")
        if not blocking:
            raise ValueError("skipped 必须提供 blocking_relations 证据")


def main() -> int:
    parser = argparse.ArgumentParser(description="校验类别级可标注性报告")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    validate(load(args.report))
    print("[通过] eligibility report 结构及门禁结论有效")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
