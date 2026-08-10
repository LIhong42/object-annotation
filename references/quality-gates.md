# 标注质量审计

对比原图、mask 可视化、bbox 可视化和 COCO，只评价并记录，不修复、不补标、不重做 image2。

## 整体扫描

- 按类别记录原图中视觉上可辨识的实例，以及 COCO 中实际提取的实例数。
- 单独记录规划阶段因可见区域不连通而排除的实例；它们属于 `excluded_instances`，不属于漏标，也不得触发 image2 调用。
- 两者不必相等。漏标、误标或数量不确定均可交付，但必须写入报告。
- 对漏标实例记录类别、外观、大致区域、近似中心和参照物。
- 不因为漏标再次调用 image2。

## 已提取实例评价

| 标签 | 定义 |
| --- | --- |
| 标注合格 | 掩码与实例可见轮廓基本贴合 |
| 误标 | annotation 对应错误对象或错误类别 |
| 多标 | 一个 annotation 纳入多个独立对象 |
| 边界偏移 | 掩码整体相对目标平移、旋转或缩放 |
| 掩码不足 | 目标可见区域部分缺失 |
| 掩码溢出 | 掩码侵入背景、阴影或邻近对象 |
| 粘连 | 本应独立的同类实例被连为一个 annotation |

允许同一实例存在多个问题；label 写最严重问题，issues 列出其余问题。

## 数据一致性

- bbox 必须是最终 mask 的最小外接轴对齐矩形。
- area 必须等于最终二值 mask 像素数。
- polygon 不得为空或退化。
- 数据结构错误必须修正，因为它不属于标注质量修复；不得改变 mask 本身。

## 报告

使用 quality-report.example.json 的结构：

- annotations：逐条评价实际 COCO annotation。
- missing_instances：记录视觉上发现但未进入 COCO 的实例；允许为空。
- excluded_instances：从 manifest 复制规划阶段排除的实例及原因；允许为空，不得与 missing_instances 重复。
- unexpected_instances：记录无法对应真实目标的多余 annotation ID；允许为空。
- summary：如实填写可辨识数量、提取数量和各质量标签计数，不把数量差异视为构建失败。
- 报告不得包含 retry、replacement、final_evaluation 或任何修复计划字段。
