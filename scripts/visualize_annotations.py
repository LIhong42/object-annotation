#!/usr/bin/env python3
"""标注文件可视化脚本。

输入原始图片 + COCO 风格标注 JSON，输出两张可视化图：

1. bbox 可视化：用红色矩形边框框出所有标注对象（附类别名与 id 标签）。
2. mask 可视化：默认以 45% 不透明度叠加红色掩码，并绘制纯红细轮廓，
   使原图纹理、真实对象边界和掩码偏差可以同时观察。

用法示例::

    python visualize_annotations.py \
        --image ori.png \
        --annotations annotations.json \
        --output-bbox bbox_visualization.png \
        --output-mask mask_visualization.png

不带参数时，默认在脚本所在目录读取 ori.png / annotations.json，
并输出 bbox_visualization.png / mask_visualization.png。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# BGR 红色（cv2 默认色彩空间）。
RED_BGR: Tuple[int, int, int] = (0, 0, 255)

# 脚本所在目录，作为输入/输出默认目录。
_DEFAULT_DIR = Path(__file__).resolve().parent


# ----------------------------------------------------------------------------
# IO（兼容中文路径）
# ----------------------------------------------------------------------------

def _imread_bgr(path: Path) -> np.ndarray:
    """读取图片为 BGR ndarray，兼容中文路径。"""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        raise FileNotFoundError(f"图片为空或不存在：{path}")
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"无法解码图片：{path}")
    return img


def _imwrite_bgr(path: Path, image_bgr: np.ndarray) -> None:
    """保存 BGR 图片，兼容中文路径。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix or ".png"
    ok, buf = cv2.imencode(ext, image_bgr)
    if not ok:
        raise ValueError(f"无法编码图片：{path}")
    buf.tofile(str(path))


# ----------------------------------------------------------------------------
# COCO 辅助
# ----------------------------------------------------------------------------

def _load_coco(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    data.setdefault("images", [])
    data.setdefault("categories", [])
    data.setdefault("annotations", [])
    return data


def _category_names(coco: Dict[str, Any]) -> Dict[int, str]:
    return {int(c["id"]): str(c.get("name", c["id"])) for c in coco["categories"]}


def _filter_annotations(
    coco: Dict[str, Any], image_id: Optional[int]
) -> List[Dict[str, Any]]:
    anns = coco.get("annotations", [])
    if image_id is None:
        return anns
    return [a for a in anns if int(a.get("image_id", -1)) == int(image_id)]


def _polygon_to_int_array(flat: Sequence[float]) -> Optional[np.ndarray]:
    """把 COCO 扁平多边形 [x1,y1,x2,y2,...] 转为 (N,2) int32 点数组。

    顶点数 < 3 视为非法多边形，返回 None。
    """
    pts = np.asarray(flat, dtype=np.float32).reshape(-1, 2)
    if len(pts) < 3:
        return None
    return np.round(pts).astype(np.int32)


def _annotation_polygons(ann: Dict[str, Any]) -> List[np.ndarray]:
    """从 annotation.segmentation 提取多边形点数组（仅支持 polygon 格式）。"""
    seg = ann.get("segmentation", [])
    polygons: List[np.ndarray] = []
    if not isinstance(seg, list):
        # RLE（dict）等格式本项目不产生，跳过。
        return polygons
    for poly in seg:
        if isinstance(poly, list) and len(poly) >= 6:
            pts = _polygon_to_int_array(poly)
            if pts is not None:
                polygons.append(pts)
    return polygons


# ----------------------------------------------------------------------------
# 绘制
# ----------------------------------------------------------------------------

def _draw_label(
    image_bgr: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    color: Tuple[int, int, int],
    font_scale: float = 0.5,
    thickness: int = 1,
) -> None:
    """在 origin(左上角) 绘制带背景的文字标签，自动避开画布上边界。"""
    (tw, th), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    x, y = origin
    # 标签默认放在框上方；贴顶时回落到框内顶部。
    top = y - th - baseline
    if top < 0:
        top = y + 1
    cv2.rectangle(
        image_bgr,
        (x, top),
        (x + tw, top + th + baseline),
        color,
        -1,
    )
    cv2.putText(
        image_bgr,
        text,
        (x, top + th),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),  # 白字配红底
        thickness,
        cv2.LINE_AA,
    )


def draw_bbox_image(
    image_bgr: np.ndarray,
    coco: Dict[str, Any],
    *,
    image_id: Optional[int],
    color: Tuple[int, int, int] = RED_BGR,
    thickness: int = 2,
    show_label: bool = True,
) -> np.ndarray:
    """在图片副本上用红色边框框出所有标注对象的 bbox。"""
    canvas = image_bgr.copy()
    cat_name = _category_names(coco)
    for ann in _filter_annotations(coco, image_id):
        bbox = ann.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x, y, w, h = (int(round(v)) for v in bbox)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, thickness)
        if show_label:
            name = cat_name.get(int(ann.get("category_id", -1)), "?")
            label = f"{name}#{ann.get('id', '?')}"
            _draw_label(canvas, label, (x, y), color)
    return canvas


def draw_mask_image(
    image_bgr: np.ndarray,
    coco: Dict[str, Any],
    *,
    image_id: Optional[int],
    color: Tuple[int, int, int] = RED_BGR,
    alpha: float = 0.45,
    draw_outline: bool = True,
) -> np.ndarray:
    """在图片副本上用红色像素填充每个对象的 segmentation 多边形区域。

    ``alpha`` 控制红色与原图的混合比例：默认 0.45，既能辨认掩码，
    又能看见原图对象边缘；1.0 为纯红填充。
    """
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError(f"alpha 必须位于 [0, 1]，实际为 {alpha}")
    overlay = image_bgr.copy()
    all_polygons: List[np.ndarray] = []
    for ann in _filter_annotations(coco, image_id):
        all_polygons.extend(_annotation_polygons(ann))

    if all_polygons:
        cv2.fillPoly(overlay, all_polygons, color)

    if alpha >= 1.0:
        canvas = overlay
    else:
        canvas = cv2.addWeighted(overlay, alpha, image_bgr, 1.0 - alpha, 0)

    if draw_outline and all_polygons:
        cv2.polylines(canvas, all_polygons, isClosed=True, color=color, thickness=1)
    return canvas


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "标注可视化：原始图 + COCO JSON -> bbox 红框图 / mask 红色填充图。"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image", "-i", type=Path, default=_DEFAULT_DIR / "ori.png",
        help="原始输入图片路径",
    )
    parser.add_argument(
        "--annotations", "-a", type=Path, default=_DEFAULT_DIR / "annotations.json",
        help="COCO 标注 JSON 路径",
    )
    parser.add_argument(
        "--output-bbox", type=Path,
        default=_DEFAULT_DIR / "bbox_visualization.png",
        help="bbox 红框可视化输出路径",
    )
    parser.add_argument(
        "--output-mask", type=Path,
        default=_DEFAULT_DIR / "mask_visualization.png",
        help="mask 红色填充可视化输出路径",
    )
    parser.add_argument(
        "--image-id", type=int, default=None,
        help="只可视化指定 image id 的标注；默认处理全部",
    )
    parser.add_argument(
        "--thickness", type=int, default=2,
        help="bbox 边框线宽（像素）",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.45,
        help="mask 红色填充不透明度，默认半透明；1.0=纯红填充",
    )
    parser.add_argument(
        "--no-label", action="store_true",
        help="bbox 图不绘制类别/id 标签",
    )
    parser.add_argument(
        "--no-mask-outline", action="store_true",
        help="关闭mask图默认绘制的纯红色多边形细轮廓",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    image_path = args.image.resolve()
    ann_path = args.annotations.resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"原始图片不存在：{image_path}")
    if not ann_path.is_file():
        raise FileNotFoundError(f"标注文件不存在：{ann_path}")

    image_bgr = _imread_bgr(image_path)
    coco = _load_coco(ann_path)

    # 尺寸一致性校验（仅 warning，不阻断）。
    for img in coco.get("images", []):
        if int(img.get("id", -1)) == int(args.image_id) if args.image_id else True:
            w, h = int(img.get("width", -1)), int(img.get("height", -1))
            if w > 0 and h > 0 and (w, h) != (image_bgr.shape[1], image_bgr.shape[0]):
                print(
                    f"[警告] 图片实际尺寸 {image_bgr.shape[1]}x{image_bgr.shape[0]}"
                    f" 与 JSON 记录 {w}x{h} 不一致",
                    file=sys.stderr,
                )

    ann_count = len(_filter_annotations(coco, args.image_id))
    print(f"[可视化] 加载 {ann_count} 条标注（来源 {ann_path.name}）")

    bbox_img = draw_bbox_image(
        image_bgr, coco,
        image_id=args.image_id,
        thickness=args.thickness,
        show_label=not args.no_label,
    )
    _imwrite_bgr(args.output_bbox.resolve(), bbox_img)
    print(f"[完成] bbox 红框图 -> {args.output_bbox}")

    mask_img = draw_mask_image(
        image_bgr, coco,
        image_id=args.image_id,
        alpha=args.alpha,
        draw_outline=not args.no_mask_outline,
    )
    _imwrite_bgr(args.output_mask.resolve(), mask_img)
    print(f"[完成] mask 红色填充图 -> {args.output_mask}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
