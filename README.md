# Object Annotation Skill

使用 image2 对单张图片中的指定对象类别进行纯色填充标注，再通过图像配准与连通域分析生成可训练、可审计的 COCO 检测/实例分割数据。

## 工作流程

```mermaid
flowchart TD
    A[原图与类别列表] --> B[环境预检]
    B --> C[逐类别可标注性门禁]
    C -->|不满足| D[跳过并记录原因]
    C -->|满足| E[选色并渲染 Prompt]
    E --> F[锁定并调用一次 image2]
    F --> G[持久化原始标注图]
    G --> H[配准和 Mask 提取]
    H --> I[单类别 COCO 与可视化]
    I --> J[结构化质量审计]
    D --> K[多类别合并]
    J --> K
    K --> L[COCO 校验与最终可视化]
```

## 安装与使用

### 1. 作为 Skill 使用

将整个 `object-annotation` 目录放入 Codex 可发现的 Skills 目录，或在当前项目中保留该目录并明确要求智能体使用其中的 `$object-annotation`。

示例指令：

```text
请使用 $object-annotation 标注这张图片中的 chair、vase、tv、person，
分别生成单类别结果，最后合并为一份完整的 COCO 实例分割数据。
```

智能体必须读取并遵循 [`SKILL.md`](SKILL.md) 中的强制工作流。

### 2. 准备 Python 环境

支持 Python 3.10–3.12。建议使用独立虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r scripts/requirements.txt
```

依赖已锁定为：

- NumPy 1.26.4
- OpenCV Headless 4.10.0.84
- Pillow 10.4.0
- SciPy 1.13.1

### 3. 执行预检

任何 image2 调用之前都必须先运行：

```bash
python scripts/preflight.py \
  --image images/ori.png \
  --output-dir output \
  --report output/reports/preflight.json
```

预检会验证依赖组合、图片解码、尺寸，并建立标准输出目录。预检失败时必须停止，不能先调用 image2 再补装依赖。

### 4. 建立可标注性报告

为每个类别建立 `eligibility-report.json`，随后校验：

```bash
python scripts/validate_eligibility_report.py \
  --report output/chair/eligibility-report.json
```

只有以下条件全部满足时，类别才能进入 image2 标注：

1. 至少存在一个可辨识实例；
2. 所有同类实例间存在明确背景间隔；
3. 同类实例不接触、不相交、不重叠、不互相遮挡；
4. 每个实例的实际可见区域为单一连通区域；
5. 目标盘点数量与 `observed_instance_count` 一致。

不满足条件时标记为 `skipped`，记录阻塞关系，且不得调用 image2。

### 5. 选色并渲染 image2 Prompt

```bash
python scripts/render_image2_prompt.py \
  --template references/object.md \
  --eligibility-report output/chair/eligibility-report.json \
  --size 640x426 \
  --ratio 1.5023 \
  --annotation-color blue \
  --output output/chair/prompt.txt
```

必须将渲染后的 `prompt.txt` 原样发送给 image2，不再自由改写。颜色选择规则见 [`references/color-selection.md`](references/color-selection.md)。

### 6. 锁定并持久化单次调用

在调用前初始化并占用唯一一次生成机会：

```bash
python scripts/run_state.py init \
  --state output/reports/chair.run-state.json \
  --category chair \
  --image images/ori.png \
  --prompt output/chair/prompt.txt

python scripts/run_state.py reserve-generation \
  --state output/reports/chair.run-state.json
```

智能体随后以原图为唯一参考图调用一次 image2。完整的缓存与分块持久化要求见 [`references/runtime-persistence.md`](references/runtime-persistence.md)。

### 7. 提取单类别 COCO

```bash
python scripts/extract_object_annotations.py \
  --image images/ori.png \
  --labeled output/image2-labels/chair.png \
  --annotation-color blue \
  --object-name chair \
  --category-id 1 \
  --output output/chair/annotations.json \
  --diagnostics output/chair/extraction-diagnostics.json
```

### 8. 生成可视化并审计质量

```bash
python scripts/visualize_annotations.py \
  --image images/ori.png \
  --annotations output/chair/annotations.json \
  --output-bbox output/chair/bbox.png \
  --output-mask output/chair/mask.png

python scripts/validate_quality_report.py \
  --annotations output/chair/annotations.json \
  --report output/chair/quality-report.json
```

质量报告需逐 annotation 记录：标注合格、误标、多标、边界偏移、掩码不足、掩码溢出、粘连或实例拆分；还需单独记录漏标和无法对应真实目标的 annotation。

### 9. 合并多类别 COCO

```bash
python scripts/build_multiclass_coco.py \
  --manifest output/multiclass-manifest.json \
  --output output/annotations.coco.json \
  --quality-index output/quality-index.json

python scripts/validate_multiclass_coco.py \
  --annotations output/annotations.coco.json \
  --quality-index output/quality-index.json
```

Manifest 示例见 [`references/multiclass-manifest.example.json`](references/multiclass-manifest.example.json)。

## 运行效果展示

以下展示来自一次真实运行。输入为一张 `640 × 426` 的室内场景，要求依次标注：

```text
chair、vase、tv、person、book、potted plant、microwave、
refrigerator、clock、dining table
```

### 原始输入

![示例原图](assets/readme-demo/input.jpg)

### image2 原始填充输出

8 个通过门禁的类别均使用纯蓝色进行一次类别级标注；下图是 image2 返回结果的缩略汇总。

![各类别 image2 原始输出](assets/readme-demo/image2-labels.jpg)

### 映射到原图坐标后的单类别 Mask

脚本从蓝色填充图中提取连通域，通过特征匹配估计变换，并映射回 `640 × 426` 原图坐标。可视化以红色半透明 Mask 展示最终 COCO polygon。

![各类别 COCO Mask](assets/readme-demo/category-masks.jpg)

### 多类别合并结果

最终合并文件保留 10 个类别，共生成 18 条 annotation。

| 合并实例分割可视化 | 合并检测框可视化 |
| --- | --- |
| ![合并 Mask](assets/readme-demo/merged-mask.jpg) | ![合并 BBox](assets/readme-demo/merged-bbox.jpg) |

完整示例数据：

- [`examples/demo-output/annotations.coco.json`](examples/demo-output/annotations.coco.json)
- [`examples/demo-output/quality-index.json`](examples/demo-output/quality-index.json)
- [`examples/demo-output/multiclass-manifest.json`](examples/demo-output/multiclass-manifest.json)

### 类别统计与质量结论

| ID | 类别 | 门禁状态 | COCO annotation 数 | 运行结论 |
| ---: | --- | --- | ---: | --- |
| 1 | chair | eligible | 2 | 1 条基本合格；1 条将三把相邻椅子粘连为一个实例 |
| 2 | vase | eligible | 5 | 2 条基本合格；1 条粘连；2 条误标 |
| 3 | tv | eligible | 1 | 电视主体被覆盖，但底部支架/底座存在掩码不足 |
| 4 | person | eligible | 2 | 两个可见人物均被提取，质量审计通过 |
| 5 | book | skipped | 0 | 多本书紧贴，实例间没有稳定背景间隔 |
| 6 | potted plant | eligible | 4 | 四个盘点实例均被提取，质量审计通过 |
| 7 | microwave | eligible | 1 | 单实例被提取，质量审计通过 |
| 8 | refrigerator | eligible | 2 | 同一台冰箱被拆分为上下两个 annotation |
| 9 | clock | eligible | 1 | 单实例被提取，质量审计通过 |
| 10 | dining table | skipped | 0 | 桌体被椅子和人物遮挡，实际可见区域不连通 |

这一示例刻意保留模型真实错误。Skill 不会通过二次生成或手工修改 Mask 来美化结果，而是将问题写入质量报告，方便训练前过滤或人工复核。

## 示例配准诊断

8 个合格类别均成功使用 SIFT + KNN + RANSAC 完成全局配准：

- 方法根据场景在 `full_affine` 与 `similarity` 中自动选择；
- RANSAC 内点比例约为 `0.710–0.832`；
- 中位配准误差约为 `0.413–0.480` 像素；
- image2 输出约为 `1540 × 1021`，最终 Mask 映射回 `640 × 426` 原图。

每个类别的完整仿射矩阵、匹配数、内点数、颜色支持像素和连通域数量都保存在 `extraction-diagnostics.json` 中。

## 输出目录结构

```text
output/
├── images/
│   └── ori.png
├── image2-labels/
│   ├── chair.png
│   └── ...
├── reports/
│   ├── preflight.json
│   ├── chair.run-state.json
│   └── chair.transfer.json
├── chair/
│   ├── eligibility-report.json
│   ├── prompt.txt
│   ├── annotations.json
│   ├── extraction-diagnostics.json
│   ├── quality-report.json
│   ├── bbox.png
│   └── mask.png
├── book/
│   └── eligibility-report.json
├── multiclass-manifest.json
├── annotations.coco.json
└── quality-index.json
```

## COCO 输出说明

最终 `annotations.coco.json` 使用标准的 COCO 核心结构：

```json
{
  "images": [
    {"id": 1, "file_name": "ori.png", "width": 640, "height": 426}
  ],
  "categories": [
    {"id": 1, "name": "chair", "supercategory": "chair"}
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "segmentation": [[0, 0, 1, 0, 1, 1]],
      "bbox": [0, 0, 1, 1],
      "area": 1,
      "iscrowd": 0
    }
  ]
}
```

其中：

- `segmentation` 为原图坐标系中的 polygon；
- `bbox` 是 Mask 的最小轴对齐外接矩形；
- `area` 是最终二值 Mask 的像素数；
- `quality-index.json` 将最终 annotation 与各类别质量报告关联起来。

## 重要限制

1. 当前流程只接受**类别级标注**，不对同一类别逐实例多次调用 image2。
2. 同类实例接触、重叠、边界不清，或可见区域不连通时，整类必须跳过。
3. 一个类别最多调用一次 image2，质量问题不会触发第二次生成。
4. 不允许手工修复、形态学修补或替换生成 Mask 来掩盖问题。
5. 图像生成模型可能改变输出分辨率；脚本通过配准映射回原图，但严重构图变化仍会导致失败。
6. `quality-report.json` 是训练前过滤和人工复核的重要依据；通过 COCO 结构校验不代表所有 annotation 都具有高标注质量。

## 仓库结构

```text
object-annotation/
├── SKILL.md
├── README.md
├── agents/
│   └── openai.yaml
├── assets/
│   ├── icon.svg
│   └── readme-demo/
├── examples/
│   └── demo-output/
├── references/
│   ├── object.md
│   ├── color-selection.md
│   ├── runtime-persistence.md
│   ├── quality-gates.md
│   └── *.example.json
└── scripts/
    ├── preflight.py
    ├── render_image2_prompt.py
    ├── run_state.py
    ├── persist_image_result.py
    ├── extract_object_annotations.py
    ├── visualize_annotations.py
    ├── build_multiclass_coco.py
    └── validate_*.py
```

## 设计原则

- **失败关闭**：不确定是否满足门禁时选择跳过，而不是生成看似完整但不可审计的标注。
- **一次调用可证明**：调用次数、原图哈希、Prompt 哈希与产物哈希进入状态文件。
- **原始结果优先**：保留 image2 原始标注图和提取诊断，便于复盘配准与 Mask 问题。
- **质量透明**：标注质量问题是数据的一部分，应被记录、索引和显式交付。
- **确定性合并**：多类别 COCO 不手工拼接，统一由合并器重编号并校验。

完整的智能体执行约束以 [`SKILL.md`](SKILL.md) 为准。
