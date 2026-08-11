#!/usr/bin/env python3
"""Persist a cached image data URL without terminal streaming or model retries."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Optional


BASE64_CHUNK = re.compile(r"^[A-Za-z0-9+/=]+$")
MAX_CHUNK_CHARS = 96_000


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def initialize(state_path: Path, output: Path) -> dict:
    state_path = state_path.resolve()
    output = output.resolve()
    if state_path.exists():
        state = _load(state_path)
        if Path(state["output"]) != output:
            raise RuntimeError("已有传输状态指向不同输出文件")
        return state
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "status": "receiving",
        "output": str(output),
        "base64_part": str(output.with_suffix(output.suffix + ".b64.part")),
        "next_index": 0,
        "received_chars": 0,
    }
    Path(payload["base64_part"]).write_text("", encoding="ascii")
    _save(state_path, payload)
    return payload


def append_chunk(state_path: Path, index: int, chunk: str) -> dict:
    state_path = state_path.resolve()
    state = _load(state_path)
    if state.get("status") != "receiving":
        raise RuntimeError("传输已结束，禁止继续追加")
    if int(index) != int(state["next_index"]):
        raise RuntimeError(
            f"分块序号错误：期望 {state['next_index']}，实际 {index}"
        )
    if not chunk or len(chunk) > MAX_CHUNK_CHARS:
        raise ValueError(f"每块必须为 1-{MAX_CHUNK_CHARS} 个 base64 字符")
    if not BASE64_CHUNK.fullmatch(chunk):
        raise ValueError("分块包含非法 base64 字符")
    with Path(state["base64_part"]).open("a", encoding="ascii") as fh:
        fh.write(chunk)
        fh.flush()
        os.fsync(fh.fileno())
    state["next_index"] = int(state["next_index"]) + 1
    state["received_chars"] = int(state["received_chars"]) + len(chunk)
    _save(state_path, state)
    return state


def _decode_stream(source: Path, destination: Path) -> None:
    remainder = ""
    with source.open("r", encoding="ascii") as src, destination.open("wb") as dst:
        while True:
            text = src.read(4 * 1024 * 1024)
            if not text:
                break
            text = remainder + text
            usable = len(text) - (len(text) % 4)
            if usable:
                dst.write(base64.b64decode(text[:usable], validate=True))
            remainder = text[usable:]
        if remainder:
            dst.write(base64.b64decode(remainder, validate=True))
        dst.flush()
        os.fsync(dst.fileno())


def finalize(
    state_path: Path,
    source_image: Optional[Path],
    ratio_tolerance: float,
) -> dict:
    from PIL import Image

    state_path = state_path.resolve()
    state = _load(state_path)
    if state.get("status") == "verified":
        return state
    if state.get("status") != "receiving" or int(state["received_chars"]) == 0:
        raise RuntimeError("没有可完成的 base64 数据")
    output = Path(state["output"])
    binary_part = output.with_suffix(output.suffix + ".binary.part")
    try:
        _decode_stream(Path(state["base64_part"]), binary_part)
        with Image.open(binary_part) as image:
            image.verify()
        with Image.open(binary_part) as image:
            image.load()
            width, height = image.size
            image_format = image.format
        if image_format != "PNG":
            raise ValueError(f"期望 PNG，实际为 {image_format}")

        ratio_error = None
        if source_image is not None:
            with Image.open(source_image.resolve()) as original:
                source_width, source_height = original.size
            source_ratio = source_width / source_height
            output_ratio = width / height
            ratio_error = abs(output_ratio - source_ratio) / source_ratio
            if ratio_error > float(ratio_tolerance):
                raise ValueError(
                    f"输出宽高比误差 {ratio_error:.6f} 超过上限 {ratio_tolerance:.6f}"
                )

        digest = hashlib.sha256(binary_part.read_bytes()).hexdigest()
        os.replace(binary_part, output)
        Path(state["base64_part"]).unlink(missing_ok=True)
        state.update({
            "status": "verified",
            "width": int(width),
            "height": int(height),
            "format": image_format,
            "sha256": digest,
            "ratio_error": ratio_error,
        })
        _save(state_path, state)
        return state
    except Exception:
        binary_part.unlink(missing_ok=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="缓存 image2 data URL 的可靠落盘工具")
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("--state", type=Path, required=True)
    init.add_argument("--output", type=Path, required=True)
    append = sub.add_parser("append")
    append.add_argument("--state", type=Path, required=True)
    append.add_argument("--index", type=int, required=True)
    append.add_argument("--chunk", required=True)
    finish = sub.add_parser("finalize")
    finish.add_argument("--state", type=Path, required=True)
    finish.add_argument("--source-image", type=Path)
    finish.add_argument("--ratio-tolerance", type=float, default=0.02)
    show = sub.add_parser("show")
    show.add_argument("--state", type=Path, required=True)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "init":
        state = initialize(args.state, args.output)
    elif args.command == "append":
        state = append_chunk(args.state, args.index, args.chunk)
    elif args.command == "finalize":
        state = finalize(args.state, args.source_image, args.ratio_tolerance)
    else:
        state = _load(args.state.resolve())
    print(json.dumps(state, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
