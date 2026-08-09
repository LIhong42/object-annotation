---
name: object-annotation
description: 使用 image2 将图片中的指定对象类别标注为高质量 COCO 检测/实例分割数据。用于对象标注、COCO JSON 交付、目标检测和实例分割数据集构建。
---

## 强制工作流

1. 读取原图宽高，按用户顺序建立稳定类别 ID。
2. 先在原图上判断各类别的可见实例数，以及实例是否接触、重叠或被遮挡。
3. 选择生成模式：
   - 类别级模式：待标注类别的所有实例彼此明显分离，读取 `references/object.md`，每个类别调用一次 image2。
   - 实例级模式：待标注类别任意实例接触、重叠，或一次生成后红色区域粘连，读取 `references/object-instance.md`，以同一张未标注原图为输入，每次只标注一个实例。
4. 每次 image2 只能标注一种类别；实例级模式下只能标注该类别的一个可辨识实例。每次都使用同一张未标注原图，不能使用上一张标注图。
5. 保存所有类别/实例标注图，创建 manifest。单文件和同类别多实例格式见 `references/multiclass-manifest.example.json`。
6. 构建 COCO 类似标注：

```bash
python scripts/build_multiclass_coco.py \
  --manifest <manifest.json> \
  --output <annotations.json> \
  --summary <annotation-summary.json>
```

7. 生成可视化：

```bash
python scripts/visualize_annotations.py \
  --image <原图> \
  --annotations <annotations.json> \
  --output-bbox <bbox-visualization.png> \
  --output-mask <mask-visualization.png>
```

8. 同时审视原图、`mask-visualization.png` 和 `bbox-visualization.png`，先做漏标扫描，再逐实例评价。按 `references/quality-gates.md` 创建 `quality-report.initial.json`；不得只给自然语言结论。
9. 运行报告校验并生成重标清单：

```bash
python scripts/quality_cycle.py validate \
  --annotations <annotations.initial.json> \
  --report <quality-report.initial.json>
python scripts/quality_cycle.py plan \
  --report <quality-report.initial.json> \
  --output <retry-plan.json>
```

10. 只对 `retry-plan.json` 中不合格或漏标的实例重新调用 image2。每张重标图只能含一个实例，输入仍为未标注原图。合格实例严禁重做；任一实例 `retry_count` 上限为 1。
11. 把每个重标图写入报告中的 `retry.labeled`，再定向替换旧 annotation 或补入漏标实例：

```bash
python scripts/quality_cycle.py apply \
  --image <原图> \
  --annotations <annotations.initial.json> \
  --report <quality-report.initial.json> \
  --output <annotations.final.json>
```

12. 重新生成可视化，仅复核重标实例，同时确认合格实例未变化。填写 `quality-report.final.json` 的 `final_evaluation`；即使一次重标后仍失败也停止重标并如实保留问题。最后运行 `quality_cycle.py validate --final`。
13. 返回 `annotations.final.json`、`quality-report.final.json`、`retry-plan.json`、两张最终可视化和摘要。报告类别数、各类别实例数、总实例数、初检失败数、实际重标数、重标后仍失败数、配准方式和局部修正量。

## Manifest规则

- `labeled` 为字符串：一张图可包含该类别的多个互不接触实例。
- `labeled` 为数组：每张图表示同一类别的一个实例；默认启用 `require_single_instance=true`。
- 接触实例必须使用数组模式。
- 默认启用 `refine_edges=true`，局部搜索上限为原图坐标中的 `max_local_shift=8` 像素。
- 小对象可将 `max_local_shift` 调低到 3–5；大幅偏移不应扩大搜索范围掩盖配准失败，应重新生成标注图。
