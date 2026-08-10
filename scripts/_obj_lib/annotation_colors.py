"""Single source of truth for supported image2 annotation colors."""

from __future__ import annotations

from typing import Dict, Tuple


ANNOTATION_COLORS: Dict[str, Dict[str, object]] = {
    "red": {"bgr": (0, 0, 255), "display": "纯红色", "rgb_text": "RGB（255, 0, 0）"},
    "green": {"bgr": (0, 255, 0), "display": "纯绿色", "rgb_text": "RGB（0, 255, 0）"},
    "black": {"bgr": (0, 0, 0), "display": "纯黑色", "rgb_text": "RGB（0, 0, 0）"},
    "white": {"bgr": (255, 255, 255), "display": "纯白色", "rgb_text": "RGB（255, 255, 255）"},
    "blue": {"bgr": (255, 0, 0), "display": "纯蓝色", "rgb_text": "RGB（0, 0, 255）"},
}


def normalize_annotation_color(annotation_color: str = "red") -> str:
    color = str(annotation_color or "red").strip().lower()
    if color not in ANNOTATION_COLORS:
        allowed = ", ".join(ANNOTATION_COLORS)
        raise ValueError(f"不支持的标注颜色 {annotation_color!r}；可选：{allowed}")
    return color


def color_bgr(annotation_color: str = "red") -> Tuple[int, int, int]:
    value = ANNOTATION_COLORS[normalize_annotation_color(annotation_color)]["bgr"]
    return tuple(value)  # type: ignore[arg-type,return-value]
