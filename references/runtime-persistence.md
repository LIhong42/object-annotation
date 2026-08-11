# image2 单次调用与可靠持久化

必须在调用前锁定次数，在工具返回后先缓存原始返回值，再进行任何解析或写盘。
持久化失败只允许从缓存恢复，绝不能再次调用 image2。

## 调用前

```bash
python scripts/run_state.py init \
  --state reports/<类别>.run-state.json --category "<类别>" \
  --image <原图> --prompt <prompt.txt>
python scripts/run_state.py reserve-generation \
  --state reports/<类别>.run-state.json
```

只有 `reserve-generation` 成功后才调用一次 image2。若命令拒绝，停止该类别。

## 工具返回后立即缓存

在 code mode 的同一个 `functions.exec` 调用中执行如下顺序。缓存键必须唯一且与
run-state 一致：

```javascript
const result = await tools.image_gen__imagegen({
  prompt,
  referenced_image_paths: [sourceImage]
});
store(cacheKey, result);
text(JSON.stringify({cached: true, cacheKey}));
```

随后运行 `mark-received --cache-key <cacheKey>`。如果工具返回的是 data URL，使用
`load(cacheKey)` 读取缓存，只取 base64 负载，按不超过 60000 字符切块。每个分块
通过参数传给以下工具；不要把整张图放入一个命令参数，不要通过 TTY 传输：

```bash
python scripts/persist_image_result.py init \
  --state reports/<类别>.transfer.json \
  --output image2-labels/<类别>.png
python scripts/persist_image_result.py append \
  --state reports/<类别>.transfer.json --index <从0递增> --chunk '<分块>'
python scripts/persist_image_result.py finalize \
  --state reports/<类别>.transfer.json --source-image <原图>
python scripts/run_state.py mark-persisted \
  --state reports/<类别>.run-state.json \
  --artifact image2-labels/<类别>.png
```

`finalize` 会流式解码、使用 Pillow 两次校验 PNG、核对宽高比、计算 SHA-256，并
原子替换最终文件。若任一步失败，保留 transfer state 与缓存；修复传输步骤并从
`load(cacheKey)` 续传。不要删除 run-state 来规避单次调用限制。
