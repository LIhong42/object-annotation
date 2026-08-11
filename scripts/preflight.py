#!/usr/bin/env python3
"""Fail-fast runtime and input checks before any image2 call."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


DEPENDENCIES = {
    "numpy": "numpy",
    "opencv-python-headless": "cv2",
    "Pillow": "PIL",
    "scipy": "scipy",
}


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    missing: list[str] = []
    for distribution, module_name in DEPENDENCIES.items():
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            missing.append(distribution)
    if missing:
        raise RuntimeError(
            "缺少运行依赖：" + ", ".join(missing)
            + "；请先安装 scripts/requirements.txt，禁止在 image2 调用后补装"
        )
    smoke = subprocess.run(
        [
            sys.executable,
            "-c",
            "import numpy, cv2, PIL, scipy; "
            "print(numpy.__version__, cv2.__version__, PIL.__version__, scipy.__version__)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if smoke.returncode != 0:
        detail = (smoke.stderr or smoke.stdout).strip()[-1000:]
        raise RuntimeError(
            f"依赖导入自检失败（exit={smoke.returncode}）：{detail}；"
            "请在 image2 调用前重建锁定依赖环境"
        )
    return versions


def run(image: Path, output_dir: Path, report: Path) -> dict:
    if not (3, 10) <= sys.version_info[:2] <= (3, 12):
        raise RuntimeError(
            f"仅验证 Python 3.10-3.12，当前为 {platform.python_version()}"
        )
    versions = _dependency_versions()

    from PIL import Image

    image = image.resolve()
    if not image.is_file():
        raise FileNotFoundError(f"原图不存在：{image}")
    with Image.open(image) as pil_image:
        pil_image.verify()
    with Image.open(image) as pil_image:
        pil_image.load()
        width, height = pil_image.size
    if width < 2 or height < 2:
        raise ValueError(f"原图尺寸无效：{width}x{height}")

    output_dir = output_dir.resolve()
    for relative in ("image2-labels", "reports", "visualizations"):
        (output_dir / relative).mkdir(parents=True, exist_ok=True)

    payload = {
        "schema_version": "1.0",
        "status": "passed",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "dependencies": versions,
        "image": {
            "path": str(image),
            "width": int(width),
            "height": int(height),
            "ratio": float(width / height),
        },
        "output_dir": str(output_dir),
    }
    _atomic_json(report.resolve(), payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="image2 调用前的强制预检")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run(args.image, args.output_dir, args.report)
    print(
        f"[通过] preflight: {payload['image']['width']}x"
        f"{payload['image']['height']} -> {args.report}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
