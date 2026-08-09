# Object Annotation Skill

`object-annotation` 使用 image2 将原图中的指定对象类别转换为 COCO 风格的目标检测与实例分割标注，并通过视觉质量评估、定向重标和逐实例评价报告提高交付质量。

## 主要能力

- 每次只处理一种对象类别，避免不同类别之间的红色掩码相互干扰。
- 支持同一类别的多个独立实例。
- 当实例接触、重叠或遮挡时，切换为单实例标注模式。
- 将 image2 输出的纯红色区域配准、映射回原图坐标。
- 输出 COCO `bbox`、`area` 和 `segmentation`。
- 生成半透明 mask 与 bbox 可视化，便于检查掩码偏移。
- 对原图和可视化结果执行整体漏标扫描与逐实例质量评价。
- 只重新标注漏标或质量不合格的实例，合格实例保持不变。
- 每个实例最多重新标注一次；重标后仍不合格时如实记录。
- 输出机器可读的质量报告和重标清单。

## 适用场景

- 为单张图片生成目标检测标注。
- 为单张图片生成实例分割标注。
- 按类别逐次构建多类别 COCO 数据。
- 检查生成式标注中的漏标、误标、粘连、偏移、欠分割和过分割。
- 为后续训练数据集构建提供图片级标注单元。

## 工作流程

1. 读取原图尺寸，确定本次唯一目标类别。
2. 视觉检查可见实例数量、遮挡关系及是否接触。
3. 选择标注模式：
   - 实例彼此明显分离：类别级模式，一次标注该类别的全部实例。
   - 任意实例接触、重叠或粘连：实例级模式，每次只标注一个实例。
4. 每次都以同一张未标注原图作为 image2 输入。
5. 从纯红色区域提取掩码，并映射到原图坐标。
6. 构建初始 COCO 标注和半透明可视化。
7. 对照原图执行整体漏标扫描，然后逐实例评价。
8. 生成结构化初检报告和重标清单。
9. 仅对不合格或漏标实例重新调用 image2，且每个实例最多一次。
10. 定向替换失败 annotation 或补入漏标实例，不修改合格 annotation。
11. 复核重标实例并生成最终质量报告。
12. 交付最终 COCO、评价报告、重标清单、摘要和可视化。

## 标注图要求

- 目标对象的完整可见区域必须使用纯红色 `RGB(255, 0, 0)` 实心填充。
- 每个实例应沿真实可见轮廓填充，不能使用矩形框或轮廓线代替。
- 不能覆盖背景、阴影、相邻对象或对象之间的空隙。
- 不补全被遮挡的不可见区域。
- 除目标像素外，不得修改、移动、裁剪或重绘原图。
- 输出优先保持原始像素尺寸，至少必须保持相同宽高比。

## 安装依赖

建议使用 Python 3.10 或更高版本：

```bash
python -m pip install -r scripts/requirements.txt
```

主要依赖包括 NumPy、OpenCV、Pillow 和 SciPy。

## Manifest

多类别或多实例输入由 manifest 描述。完整示例见：

```text
references/multiclass-manifest.example.json
```

类别级模式：

```json
{
  "id": 1,
  "name": "chair",
  "labeled": "chair-labeling.png",
  "refine_edges": true,
  "max_local_shift": 8
}
```

实例级模式：

```json
{
  "id": 1,
  "name": "chair",
  "labeled": [
    "chair-01-labeling.png",
    "chair-02-labeling.png"
  ],
  "require_single_instance": true,
  "refine_edges": true,
  "max_local_shift": 8
}
```

数组中的每张图必须且只能包含一个红色实例。接触或重叠实例必须使用这种模式。

## 构建 COCO 标注

```bash
python scripts/build_multiclass_coco.py \
  --manifest manifest.json \
  --output annotations.initial.json \
  --summary annotation-summary.initial.json
```

也可以处理单个类别标注图：

```bash
python scripts/extract_object_annotations.py \
  --image ori.png \
  --labeled chair-labeling.png \
  --object-name chair \
  --output annotations.json
```

追加另一个类别时使用 `--append`。重新构建同一类别时，可配合 `--replace-category` 避免产生重复 annotation。

## 生成可视化

```bash
python scripts/visualize_annotations.py \
  --image ori.png \
  --annotations annotations.initial.json \
  --output-bbox bbox-visualization.initial.png \
  --output-mask mask-visualization.initial.png
```

mask 可视化默认采用半透明红色叠加，因此可以同时观察原图真实轮廓和标注边缘。

## 质量评价

初检必须同时检查原图、mask 可视化、bbox 可视化和 COCO 数据。质量标签包括：

| 标签 | 含义 |
| --- | --- |
| 标注合格 | 完整贴合实例的真实可见边界 |
| 漏标 | 原图中存在目标实例，但 COCO 中没有对应标注 |
| 多标 | 标入了其他对象或把多个实例错误连接 |
| 边界偏移 | 掩码整体平移或缩放，与对象位置不一致 |
| 掩码不足 | 对象边缘、腿、把手等可见部分缺失 |
| 掩码溢出 | 掩码越过对象边界，覆盖背景或邻近区域 |

报告格式见 `references/quality-report.example.json`，详细判定规则见 `references/quality-gates.md`。

验证报告并生成重标清单：

```bash
python scripts/quality_cycle.py validate \
  --annotations annotations.initial.json \
  --report quality-report.initial.json

python scripts/quality_cycle.py plan \
  --report quality-report.initial.json \
  --output retry-plan.json
```

## 定向重标与替换

重标时必须遵守以下限制：

- 只重标 `retry-plan.json` 中的失败或漏标实例。
- 每张重标图只包含一个实例。
- 仍然使用未标注原图作为输入。
- 合格实例不能重新生成。
- 每个实例的 `retry_count` 只能为 `0` 或 `1`。

将重标图路径写入质量报告的 `retry.labeled` 后执行：

```bash
python scripts/quality_cycle.py apply \
  --image ori.png \
  --annotations annotations.initial.json \
  --report quality-report.initial.json \
  --output annotations.final.json
```

重新生成可视化并填写 `final_evaluation`，最后验证：

```bash
python scripts/quality_cycle.py validate \
  --annotations annotations.final.json \
  --report quality-report.final.json \
  --final
```

一次重标后仍不合格的实例不能再次重标，必须在最终报告中保留失败标签、问题描述、证据和大致位置。

## 最终交付

一次完整任务至少应交付：

```text
annotations.final.json
quality-report.final.json
retry-plan.json
annotation-summary.final.json
bbox-visualization.final.png
mask-visualization.final.png
```

其中 `quality-report.final.json` 必须覆盖每个可见实例。未重标的合格实例沿用 `initial_evaluation`；重标实例使用 `final_evaluation` 记录最终状态。

## 目录结构

```text
object-annotation/
├── SKILL.md
├── README.md
├── agents/
│   └── openai.yaml
├── assets/
│   └── icon.svg
├── references/
│   ├── multiclass-manifest.example.json
│   ├── object.md
│   ├── object-instance.md
│   ├── quality-gates.md
│   └── quality-report.example.json
└── scripts/
    ├── build_multiclass_coco.py
    ├── extract_object_annotations.py
    ├── quality_cycle.py
    ├── visualize_annotations.py
    ├── requirements.txt
    └── _obj_lib/
```

## 关键原则

- image2 一次只能标注一个类别。
- 接触实例使用单实例标注图，不能通过侵蚀或人为留缝伪分离。
- 不扩大局部搜索范围来掩盖严重配准失败。
- 不用手绘粗略多边形代替 image2 标注。
- bbox 必须是 mask 的最小外接矩形。
- `area` 必须等于二值 mask 的像素面积。
- 质量评价必须逐实例记录，不能只给总体结论。
- 定向修复不能改变已经通过评价的实例。
