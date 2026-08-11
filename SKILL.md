---
name: object-annotation
description: 使用 image2 为单张图片中的一个或多个指定类别构建高质量 COCO 检测/实例分割标注。适用于类别级对象标注、可恢复的单次 image2 调用、原始输出持久化、质量审计，以及将多个单类别结果合并为一份 COCO；同类实例接触、重叠、边界不清或单实例可见区域不连通时，整类跳过且不调用 image2。
---

# 对象标注

严格按以下顺序执行。每个类别是独立、最多一次 image2 调用的任务；最后可确定性
合并。质量问题如实记录，不重标、不修 mask。

## 1. 在任何 image2 调用前完成预检

使用 Python 3.10-3.12，并先安装锁定依赖：

```bash
python -m pip install -r scripts/requirements.txt
python scripts/preflight.py \
  --image <原图> --output-dir <任务目录> \
  --report <任务目录>/reports/preflight.json
```

预检必须成功。它隔离测试 NumPy、OpenCV、Pillow 和 SciPy 导入，验证原图并创建
标准目录。依赖缺失、ABI 崩溃、图片解码失败或版本不符时立即停止；禁止先调用
image2 再补装依赖。

## 2. 建立类别清单

- 保持用户类别顺序，为其分配稳定且唯一的正整数 `category_id`。
- 每个类别建立独立目录、报告、prompt、run-state、image2 原始输出、COCO 和可视化。
- 多类别任务先创建 `multiclass-manifest.json`。结构见
  `references/multiclass-manifest.example.json`。

## 3. 执行类别级可标注性门禁

对原图中的目标类别做一次完整视觉盘点。对每个类别创建 `eligibility-report.json`，并运行：

```bash
python scripts/validate_eligibility_report.py --report <eligibility-report.json>
```

只有以下条件全部满足才可标记 `eligible`：

- 至少有一个可辨识实例；
- 所有同类实例之间有明确背景间隔，互不接触、相交、重叠或遮挡；
- 每个实例的实际可见像素是单一连通区域；
- `target_inventory` 逐一列出全部实例，数量等于观察数量；
- `exclusions` 列出画面中易混淆但不属于目标类别的对象；
- `blocking_relations` 为空。

任一条件失败则标记 `skipped`，提供非空的 `blocking_relations`，整类不调用
image2。不得裁剪、改成逐实例任务或用多次调用绕过门禁。合格与跳过示例分别见
`references/eligibility-report.example.json` 和
`references/eligibility-report.skipped.example.json`。

## 4. 选色并渲染无歧义提示词

读取 `references/color-selection.md`，从 `red`、`green`、`black`、`white`、`blue`
中选与目标及背景混淆最少的唯一颜色。只对 `eligible` 类别执行：

```bash
python scripts/render_image2_prompt.py \
  --template references/object.md \
  --eligibility-report <eligibility-report.json> \
  --size <宽>x<高> --ratio <宽高比> \
  --annotation-color <颜色> --output <prompt.txt>
```

渲染结果包含目标数量、逐实例位置、完整性要求与明确排除项。将 `prompt.txt` 全文
原样发送，不再自由改写。

## 5. 锁定次数、调用一次并可靠落盘

先完整读取 `references/runtime-persistence.md` 并照做。核心不可省略：

1. `run_state.py init` 锁定原图与 prompt 的 SHA-256；
2. `reserve-generation` 在调用前把次数原子增加到 1；
3. 以原图作为唯一参考图调用一次 image2；
4. 工具返回后立即用 code-mode `store(cache_key, result)` 缓存原始结果；
5. 从缓存按不超过 60000 字符分块，经 `persist_image_result.py` 写入
   `image2-labels/<类别>.png`；
6. `finalize` 校验 PNG、宽高比和 SHA-256，再 `mark-persisted`。

持久化或传输失败时，从 `load(cache_key)` 恢复并继续传输。禁止第二次 image2 调用，
禁止删除 run-state，禁止 TTY/base64 流式交互，禁止把完整 data URL 放进单个命令参数。

## 6. 提取单类别 COCO 与诊断

```bash
python scripts/extract_object_annotations.py \
  --image <原图> --labeled <image2-labels/类别.png> \
  --annotation-color <颜色> --object-name <类别> \
  --category-id <稳定ID> --output <annotations.json> \
  --diagnostics <reports/extraction-diagnostics.json>
python scripts/run_state.py mark-extracted \
  --state <run-state.json> --artifact <annotations.json>
```

诊断文件保留配准、仿射矩阵、输入输出尺寸、映射 mask 数和接受 annotation 数，便于
定位缩放、错位或提取异常。不得基于诊断修补 mask。

## 7. 可视化并只记录质量

```bash
python scripts/visualize_annotations.py \
  --image <原图> --annotations <annotations.json> \
  --output-bbox <bbox.png> --output-mask <mask.png>
python scripts/validate_quality_report.py \
  --annotations <annotations.json> --report <quality-report.json>
python scripts/run_state.py mark-audited \
  --state <run-state.json> --artifact <quality-report.json>
```

读取 `references/quality-gates.md`，同时审视原图、image2 原始输出、两张可视化和
COCO。逐 annotation 记录合格、误标、多标、边界偏移、掩码不足、掩码溢出、粘连
或实例拆分，并记录漏标和多余标注。验证器必须通过；任何质量结论都不得触发重标、
替换或修复。

## 8. 合并多类别并终检

所有类别完成或跳过后，使用 manifest 合并；不要手工拼 JSON：

```bash
python scripts/build_multiclass_coco.py \
  --manifest <multiclass-manifest.json> \
  --output <annotations.coco.json> \
  --quality-index <quality-index.json>
python scripts/validate_multiclass_coco.py \
  --annotations <annotations.coco.json> \
  --quality-index <quality-index.json>
```

合并器保留用户类别顺序和 ID、重编号 annotation ID、统一 image ID，并让 skipped
类别以零 annotation 留在 categories 与质量索引中。

## 9. 交付

多类别任务至少交付：原图、`annotations.coco.json`、`quality-index.json`、manifest，
以及每个类别的 eligibility report。合格类别另交付 prompt、run-state、image2 原始
输出、单类别 COCO、提取诊断、质量报告和两张可视化；跳过类别明确说明未调用
image2。

# 禁止事项

- 同一类别逐实例调用、质量重试或任何第二次 image2 调用。
- 在同类实例接触、重叠、边界不清或可见区域不连通时继续。
- 把相似物、相邻柜体、背景、阴影或反光当成目标。
- 修改生成 mask 来掩盖漏标、溢出、错位、粘连或实例拆分。
- 在模型调用后安装依赖，或混用不同版本的 skill 实现。
