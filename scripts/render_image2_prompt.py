#!/usr/bin/env python3
"""Render an image2 prompt template without paraphrasing its contents."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _obj_lib.annotation_colors import ANNOTATION_COLORS

PLACEHOLDER = re.compile(r"{{([A-Z][A-Z0-9_]*)}}")


def render(template: str, values: Dict[str, str]) -> str:
    required = set(PLACEHOLDER.findall(template))
    missing = sorted(key for key in required if not values.get(key, "").strip())
    if missing:
        raise ValueError("缺少占位符值：" + ", ".join(missing))
    rendered = PLACEHOLDER.sub(lambda match: values[match.group(1)].strip(), template)
    remaining = sorted(set(PLACEHOLDER.findall(rendered)))
    if remaining:
        raise ValueError("仍有未填充占位符：" + ", ".join(remaining))
    return rendered


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="完整填充参考模板，并输出可原样发送给 image2 的提示词"
    )
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--eligibility-report", type=Path, required=True)
    parser.add_argument("--size", required=True)
    parser.add_argument("--ratio", required=True)
    parser.add_argument(
        "--annotation-color", choices=tuple(ANNOTATION_COLORS), default="red"
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    template = args.template.read_text(encoding="utf-8")
    report = json.loads(args.eligibility_report.read_text(encoding="utf-8"))
    from validate_eligibility_report import validate
    validate(report)
    if report["status"] != "eligible":
        raise ValueError("仅 eligible 类别可以渲染 image2 prompt")

    color_spec = ANNOTATION_COLORS[args.annotation_color]
    inventory = "\n".join(
        f"- {item['instance_key']}：{item['description']}；位置：{item['location']}"
        for item in report["target_inventory"]
    )
    exclusions = "\n".join(
        f"- {item['description']}：{item['reason']}"
        for item in report["exclusions"]
    ) or "- 无额外排除项；仍然只能填充目标类别。"
    values = {
        "TARGET_OBJECTS": str(report["target_category"]),
        "EXPECTED_COUNT": str(report["observed_instance_count"]),
        "TARGET_INVENTORY": inventory,
        "EXCLUSIONS": exclusions,
        "SIZE": args.size,
        "RATIO": args.ratio,
        "ANNOTATION_COLOR_NAME": str(color_spec["display"]),
        "ANNOTATION_COLOR_RGB": str(color_spec["rgb_text"]),
    }
    rendered = render(template, values)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(f"[完成] image2 prompt -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
