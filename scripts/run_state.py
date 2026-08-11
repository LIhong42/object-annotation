#!/usr/bin/env python3
"""Atomic per-category state machine that prevents duplicate image2 calls."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _save(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


@contextmanager
def _locked(state_path: Path) -> Iterator[None]:
    lock_path = state_path.with_suffix(state_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        yield


def _event(state: dict, action: str, **details: object) -> None:
    state.setdefault("events", []).append(
        {"at": _now(), "action": action, **details}
    )
    state["updated_at"] = _now()


def initialize(
    state_path: Path,
    category: str,
    image: Path,
    prompt: Optional[Path],
) -> dict:
    state_path = state_path.resolve()
    image = image.resolve()
    prompt = prompt.resolve() if prompt is not None else None
    if not image.is_file():
        raise FileNotFoundError("初始化状态前必须存在原图")
    if prompt is not None and not prompt.is_file():
        raise FileNotFoundError("指定的 prompt.txt 不存在")
    with _locked(state_path):
        if state_path.exists():
            existing = _load(state_path)
            if (
                existing.get("category") != category
                or existing.get("image_sha256") != _sha256(image)
                or existing.get("prompt_sha256")
                != (_sha256(prompt) if prompt is not None else None)
            ):
                raise RuntimeError("已有 run-state 与当前类别、原图或提示词不一致")
            return existing
        state = {
            "schema_version": "1.0",
            "category": category,
            "status": "prepared",
            "generation_calls": 0,
            "max_generation_calls": 1,
            "image": str(image),
            "image_sha256": _sha256(image),
            "prompt": str(prompt) if prompt is not None else None,
            "prompt_sha256": _sha256(prompt) if prompt is not None else None,
            "cache_key": None,
            "created_at": _now(),
            "updated_at": _now(),
            "events": [],
        }
        _event(state, "initialized")
        _save(state_path, state)
        return state


def transition(state_path: Path, action: str, args: argparse.Namespace) -> dict:
    state_path = state_path.resolve()
    with _locked(state_path):
        if not state_path.is_file():
            raise FileNotFoundError(f"run-state 不存在：{state_path}")
        state = _load(state_path)
        status = state.get("status")

        if action == "reserve-generation":
            if not state.get("prompt") or not state.get("prompt_sha256"):
                raise RuntimeError("没有已锁定的 prompt，禁止调用 image2")
            if status != "prepared" or int(state.get("generation_calls", 0)) >= 1:
                raise RuntimeError(
                    "image2 调用已被占用或完成；禁止第二次调用。"
                    "持久化失败时必须复用 cache_key 对应的原结果"
                )
            state["generation_calls"] = 1
            state["status"] = "generation_reserved"
            _event(state, action)
        elif action == "mark-received":
            if status != "generation_reserved":
                raise RuntimeError("仅 generation_reserved 状态可记录模型返回")
            if not args.cache_key:
                raise ValueError("cache_key 不能为空")
            state["cache_key"] = args.cache_key
            state["status"] = "generation_received"
            _event(state, action, cache_key=args.cache_key)
        elif action == "mark-persisted":
            if status != "generation_received":
                raise RuntimeError("仅 generation_received 状态可确认落盘")
            artifact = args.artifact.resolve()
            if not artifact.is_file():
                raise FileNotFoundError(f"标注图不存在：{artifact}")
            state["artifact"] = str(artifact)
            state["artifact_sha256"] = _sha256(artifact)
            state["status"] = "image_persisted"
            _event(state, action, artifact=str(artifact))
        elif action == "mark-extracted":
            if status != "image_persisted":
                raise RuntimeError("仅 image_persisted 状态可确认 COCO 提取")
            artifact = args.artifact.resolve()
            if not artifact.is_file():
                raise FileNotFoundError(f"COCO 不存在：{artifact}")
            state["annotations"] = str(artifact)
            state["annotations_sha256"] = _sha256(artifact)
            state["status"] = "coco_extracted"
            _event(state, action, artifact=str(artifact))
        elif action == "mark-audited":
            if status != "coco_extracted":
                raise RuntimeError("仅 coco_extracted 状态可确认质量审计")
            artifact = args.artifact.resolve()
            if not artifact.is_file():
                raise FileNotFoundError(f"质量报告不存在：{artifact}")
            state["quality_report"] = str(artifact)
            state["quality_report_sha256"] = _sha256(artifact)
            state["status"] = "audited"
            _event(state, action, artifact=str(artifact))
        elif action == "mark-skipped":
            if status != "prepared" or int(state.get("generation_calls", 0)) != 0:
                raise RuntimeError("仅未调用 image2 的 prepared 任务可标记 skipped")
            artifact = args.artifact.resolve()
            if not artifact.is_file():
                raise FileNotFoundError(f"门禁报告不存在：{artifact}")
            state["eligibility_report"] = str(artifact)
            state["status"] = "skipped"
            _event(state, action, artifact=str(artifact))
        else:
            raise ValueError(f"未知动作：{action}")

        _save(state_path, state)
        return state


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="对象标注任务状态机")
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("--state", type=Path, required=True)
    init.add_argument("--category", required=True)
    init.add_argument("--image", type=Path, required=True)
    init.add_argument("--prompt", type=Path)
    for name in (
        "reserve-generation", "mark-received", "mark-persisted",
        "mark-extracted", "mark-audited", "mark-skipped",
    ):
        child = sub.add_parser(name)
        child.add_argument("--state", type=Path, required=True)
        if name == "mark-received":
            child.add_argument("--cache-key", required=True)
        if name in {"mark-persisted", "mark-extracted", "mark-audited", "mark-skipped"}:
            child.add_argument("--artifact", type=Path, required=True)
    show = sub.add_parser("show")
    show.add_argument("--state", type=Path, required=True)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "init":
        state = initialize(args.state, args.category, args.image, args.prompt)
    elif args.command == "show":
        state = _load(args.state.resolve())
    else:
        state = transition(args.state, args.command, args)
    print(json.dumps(state, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
