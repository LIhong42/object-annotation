---
name: object-annotation
description: 使用 image2 对图片中的指定对象类别进行单次标注，生成 COCO 目标检测/实例分割数据、可视化和如实质量报告。用于对象标注、单实例消歧、多类别 COCO 合并及标注质量审计；每个类别或实例最多调用一次 image2，不执行重标或掩码修复。
---

## 强制工作流

1. 读取原图宽高，按用户给出的顺序建立稳定类别 ID。
2. 观察原图，记录各类别可辨识实例及接触、重叠和遮挡关系。观察数量只用于选择生成模式，不作为输出数量约束。
3. 选择生成模式：
   - 类别级：同类实例明显分离时，读取 `references/object.md`，该类别调用一次 image2。
   - 单实例级：同类实例接触、重叠或容易混淆时，读取 `references/instance-localization.md` 和 `references/object-instance.md`，每个计划实例分别调用一次 image2。
4. 单实例描述必须通过唯一性检查：使用固定图像坐标系、同类排序序号、图像区域、稳定参照物、外观特征和遮挡关系，使描述只能指向一个实例。若仍可能匹配多个实例，先改写描述，不能直接调用 image2。
5. 每个类别级任务或计划单实例最多调用一次 image2。质量评价后不得重标、补标或再次调用 image2；误标、漏标、粘连和偏移均如实记录。
6. 每次都以同一张未标注原图为输入，不得把上一张标注图作为下一次输入。每次只能标注一种类别；单实例级每次只能标注一个描述明确的实例。
7. 将 image2 原始输出全部保存到交付目录的 `image2-labels/` 子目录，不得与 COCO、报告或可视化混放。
8. 创建 manifest。格式见 `references/multiclass-manifest.example.json`。其中 `labeled` 路径必须指向 `image2-labels/`。
9. 仅使用首次全局仿射变换映射红色 mask，不做仿射后的平移、边缘吸附、裁剪、连通块合并、颜色放宽或数量修正：

```bash
python scripts/build_multiclass_coco.py \
  --manifest <manifest.json> \
  --output <annotations.json> \
  --summary <annotation-summary.json>
```

10. 不强制任何类别或单实例标注图产生指定数量的 annotation。提取到 0 个、1 个或多个实例均继续构建 COCO，并在摘要和质量报告中记录。
11. 生成可视化：

```bash
python scripts/visualize_annotations.py \
  --image <原图> \
  --annotations <annotations.json> \
  --output-bbox <bbox-visualization.png> \
  --output-mask <mask-visualization.png>
```

12. 同时审视原图、mask 可视化、bbox 可视化和 COCO。读取 `references/quality-gates.md`，创建 `quality-report.json`。报告必须如实保留漏标、误标、粘连、偏移、掩码不足或溢出，不触发任何修复。
13. 校验报告结构：

```bash
python scripts/validate_quality_report.py \
  --annotations <annotations.json> \
  --report <quality-report.json>
```

14. 返回 `annotations.json`、`annotation-summary.json`、`quality-report.json`、两张可视化、manifest 和完整 `image2-labels/` 文件夹。报告类别数、各类别提取实例数、总实例数、质量问题数和仿射方法。

## Manifest 规则

- `labeled` 为字符串：一张 image2 输出对应一次类别级调用。
- `labeled` 为数组：每张 image2 输出对应一次单实例级调用。
- 数组长度表示计划调用数，不代表强制实例数。
- 所有路径必须位于 `image2-labels/`。
- 不支持 `require_single_instance`、`refine_edges`、`max_local_shift`、`fail_on_local_limit`、`expected_instances` 或任何掩码修复字段。
