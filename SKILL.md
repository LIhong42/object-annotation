---
name: object-annotation
description: 使用 image2 对单张图片中的一个对象类别进行一次类别级纯色填充标注，并生成 COCO 目标检测/实例分割数据、可视化和如实质量报告。仅用于目标类别的全部可见实例彼此明确分离、互不接触且互不重叠的图片；若任意同类实例接触、相交、重叠、粘连或无法确认边界分离，则整类跳过且不调用 image2。不支持单实例定位、多类别合并、重标或掩码修复。
---

# 强制工作流

## 1. 建立单类别任务

- 每次只接受一张原图和一个目标类别。
- 不拆分为单实例任务，不生成单实例描述，不使用多类别追加流程。
- 读取原图宽高并固定 `category_id=1`。

## 2. 执行类别级可标注性门禁

调用 image2 前观察原图中的全部目标实例，并优先判断同类实例之间的空间关系。

只有同时满足以下条件时才继续：

- 所有可见目标实例均可辨识；
- 任意两个同类实例的可见区域之间都有明确背景间隔；
- 同类实例不存在接触、相交、重叠、堆叠、粘连或边界归属不清；
- 每个目标实例的待填充可见区域是单一连通区域。

出现以下任一情况时，整类判定为 `skipped`：

- 任意两个同类实例接触或边界相连，即使只在一个点或一小段边缘接触；
- 任意两个同类实例前后重叠或互相遮挡；
- 无法可靠判断两个同类实例之间是否存在背景间隔；
- 任一实例因遮挡、出画或其他原因形成多个互不连通的可见区域。

跳过时不得调用 image2。创建 `eligibility-report.json`，记录 `status: "skipped"`、观察到的实例数、触发门禁的关系、位置与证据；随后直接返回该报告。不得改用单实例标注规避门禁。

通过时创建 `eligibility-report.json`，记录 `status: "eligible"`、观察到的实例数，以及“全部同类实例具有明确背景间隔”的判断证据，然后继续。

报告结构见 `references/eligibility-report.example.json`。

## 3. 选择唯一标注颜色

读取 `references/color-selection.md`。从 `red`、`green`、`black`、`white`、`blue` 中选择与目标对象、紧邻背景和原图已有颜色混淆最少的颜色。记录参数值和理由；无法确定时使用 `red`。

## 4. 渲染并调用一次 image2

使用固定类别级模板生成提示词：

```bash
python scripts/render_image2_prompt.py \
  --template references/object.md \
  --target-objects "<类别>" \
  --size "<宽>x<高>" \
  --ratio "<宽高比>" \
  --annotation-color "<red|green|black|white|blue>" \
  --output prompt.txt
```

把 `prompt.txt` 全文原样传给 image2，并以未标注原图作为唯一参考图。一个任务最多调用一次 image2；不得重标、补标或再次调用。

将 image2 原始输出保存为 `image2-labels/<类别>.png`。不得将它与 COCO、报告或可视化混放。

## 5. 提取单类别 COCO

```bash
python scripts/extract_object_annotations.py \
  --image <原图> \
  --labeled image2-labels/<类别>.png \
  --annotation-color <颜色> \
  --object-name "<类别>" \
  --output annotations.json
```

脚本只创建一个类别，不支持追加第二个类别。颜色区域的每个连通域映射为一个实例；不强制提取数量，不执行仿射后的 mask 修复。

## 6. 可视化与质量审计

```bash
python scripts/visualize_annotations.py \
  --image <原图> \
  --annotations annotations.json \
  --output-bbox bbox-visualization.png \
  --output-mask mask-visualization.png
```

同时审视原图、两张可视化和 COCO。读取 `references/quality-gates.md` 并创建 `quality-report.json`。误标、漏标、粘连、偏移、掩码不足或溢出只记录，不触发修复或新一次 image2 调用。

```bash
python scripts/validate_quality_report.py \
  --annotations annotations.json \
  --report quality-report.json
```

## 7. 交付

通过门禁并完成标注时返回：

- `eligibility-report.json`
- `prompt.txt`
- `image2-labels/` 原始输出
- `annotations.json`
- `quality-report.json`
- `bbox-visualization.png`
- `mask-visualization.png`

跳过时只返回 `eligibility-report.json`，并明确说明未调用 image2、未生成标注数据。

# 禁止事项

- 禁止单实例级标注、唯一实例定位描述或逐实例 image2 调用。
- 禁止一次任务处理多个类别或向已有 COCO 追加类别。
- 禁止在同类实例接触、相交、重叠或边界不确定时继续标注。
- 禁止用单实例模式、裁剪图片或多次调用绕过类别级门禁。
- 禁止根据质量评估重标、替换或修复 mask。
