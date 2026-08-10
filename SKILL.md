---
name: object-annotation
description: 使用 image2 对图片中的指定对象类别进行单次标注，生成 COCO 目标检测/实例分割数据、可视化和如实质量报告。用于对象标注、单实例消歧、多类别 COCO 合并及标注质量审计；每个类别或实例最多调用一次 image2，不执行重标或掩码修复。
---

## 强制工作流

1. 读取原图宽高，按用户给出的顺序建立稳定类别 ID。
2. 观察原图，记录各类别可辨识实例及接触、重叠和遮挡关系。先执行“可标注性过滤”：若一个对象因其他对象遮挡、出画或其他原因，其可见像素被分成两个或更多彼此不连通的区域，则把它写入排除清单，不得调用 image2 标注该实例。观察数量只用于规划，不作为输出数量约束。
3. 读取 `references/color-selection.md`。模型分析目标对象、紧邻背景和整张原图，从 `red`、`green`、`black`、`white`、`blue` 中为每次调用选择对比最强且原图混淆最少的标注颜色，并记录参数值和理由。不得使用其它颜色；无法确定时使用 `red`。
4. 只对过滤后仍合格的实例选择生成模式：
   - 类别级：同类实例明显分离时，读取 `references/object.md`，该类别调用一次 image2。
   - 单实例级：同类实例接触、重叠或容易混淆时，读取 `references/instance-localization.md` 和 `references/object-instance.md`，每个计划实例分别调用一次 image2。
5. 单实例描述必须通过唯一性检查：使用固定图像坐标系、同类排序序号、图像区域、稳定参照物、外观特征和遮挡关系，使描述只能指向一个实例。若仍可能匹配多个实例，先改写描述，不能直接调用 image2。
6. 调用 image2 前必须使用脚本填充参考文件占位符：

```bash
python scripts/render_image2_prompt.py \
  --template references/object.md \
  --target-objects "<类别>" \
  --size "<宽>x<高>" \
  --ratio "<宽高比>" \
  --excluded-instances "<排除实例说明；无则写 无>" \
  --annotation-color "<red|green|black|white|blue>" \
  --output <prompt.txt>
```

单实例模式改用 `references/object-instance.md` 并额外传入 `--instance-description`。脚本必须确认无未填充占位符。把生成的 prompt 文件全文原样传给 image2；不得概括、改写、删减、添加前后缀或把模板内容重新组织成临时提示词。
7. 每个类别级任务或计划单实例最多调用一次 image2。质量评价后不得重标、补标或再次调用 image2；误标、漏标、粘连和偏移均如实记录。
8. 每次都以同一张未标注原图为输入，不得把上一张标注图作为下一次输入。每次只能标注一种类别；单实例级每次只能标注一个描述明确的实例。类别级提示词必须列出排除实例，要求 image2 不标注它们。
9. 将 image2 原始输出全部保存到交付目录的 `image2-labels/` 子目录，不得与 COCO、报告或可视化混放。
10. 创建 manifest。格式见 `references/multiclass-manifest.example.json`。每个类别写入与对应调用一致的 `annotation_color` 和 `annotation_color_reason`，`labeled` 路径必须指向 `image2-labels/`；若某类别没有合格实例，使用空数组并记录 `excluded_instances`。旧 manifest 缺省颜色时按 `red` 处理。
11. 在标注图坐标系提取 mask：
    - 对所有 image2 标注图统一使用同一套参数化算法，不区分纯色填充或纹理化覆盖。
    - 每次把 manifest 的 `annotation_color` 传给提取器；不传参数时默认 `red`。
    - 结合相对原图的颜色变化、所选颜色优势和编辑方向识别完整区域。
    - 从高置信同色种子仅在连续的同色变化支持区域内执行迟滞生长，并填充具有编辑证据的封闭纹理孔洞。
    - 直接以最终同色变化区域的连通域区分实例；不得跨越非标注色间隔做闭合、膨胀或碎片合并。天然分离的多个标注色区域必须输出为多个实例。

    具体算法和摘要字段见 `references/color-mask-extraction.md`。

随后仅使用首次全局仿射变换映射 mask，不做仿射后的平移、边缘吸附或轮廓对齐：

```bash
python scripts/build_multiclass_coco.py \
  --manifest <manifest.json> \
  --output <annotations.json> \
  --summary <annotation-summary.json>
```

12. 不强制任何类别或单实例标注图产生指定数量的 annotation。提取到 0 个、1 个或多个实例均继续构建 COCO，并在摘要和质量报告中记录。
13. 生成可视化：

```bash
python scripts/visualize_annotations.py \
  --image <原图> \
  --annotations <annotations.json> \
  --output-bbox <bbox-visualization.png> \
  --output-mask <mask-visualization.png>
```

14. 同时审视原图、mask 可视化、bbox 可视化和 COCO。读取 `references/quality-gates.md`，创建 `quality-report.json`。报告必须区分“规划时排除”与“意外漏标”，并如实保留误标、粘连、偏移、掩码不足或溢出，不触发任何修复。
15. 校验报告结构：

```bash
python scripts/validate_quality_report.py \
  --annotations <annotations.json> \
  --report <quality-report.json>
```

16. 返回 `annotations.json`、`annotation-summary.json`、`quality-report.json`、两张可视化、manifest、渲染后的 prompt 文件和完整 `image2-labels/` 文件夹。报告类别数、各类别提取实例数、总实例数、排除实例数、质量问题数和仿射方法。

## Manifest 规则

- `labeled` 为字符串：一张 image2 输出对应一次类别级调用。
- `labeled` 为数组：每张 image2 输出对应一次单实例级调用。
- `labeled` 为空数组：该类别不存在通过可标注性过滤的实例，不调用 image2，但仍在 COCO 中保留类别。
- 数组长度表示计划调用数，不代表强制实例数。
- 所有路径必须位于 `image2-labels/`。
- `excluded_instances` 记录因可见区域不连通而禁止标注的实例；不得把这些实例计为意外漏标。
- `annotation_color` 只能是 `red`、`green`、`black`、`white`、`blue`；缺省为 `red`。提示词、标注图和 mask 提取必须一致。
- 不支持 `require_single_instance`、`refine_edges`、`max_local_shift`、`fail_on_local_limit`、`expected_instances` 或任何掩码修复字段。
