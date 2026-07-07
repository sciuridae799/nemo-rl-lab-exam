# qa_rl — 技术培训考题（考试任务数据集）

完整训练 / 验证集在集群共享盘 **`/data/datasets/qa_rl`**（`train.jsonl` / `val.jsonl`），提交作业时平台自动注入 `QA_RL_DATA_DIR`，**无需在本机准备**。

本目录仅入库 **格式示例**（`examples.jsonl`），供了解字段结构与各题型写法。

## 字段格式

每行一条 JSON：

```json
{"query": "题目与作答说明（含选项）", "expected_answer": "[type] 标准答案"}
```

| `expected_answer` 前缀 | 题型 | 模型 `\boxed{}` 写法 |
| --- | --- | --- |
| `[single]` | 单选 | `\boxed{B}` |
| `[multiple]` | 多选 | `\boxed{A,C,D}`（字母逗号分隔） |
| `[bool]` | 判断 | `\boxed{A}`（A=对，B=错） |
| `[fill]` | 填空 | `\boxed{空1; 空2}`（按空顺序，`;` 分隔） |
| `[short]` | 简答 | `\boxed{要点1; 要点2}`（关键词；标准答案用 `\|\|\|` 分隔多个可接受写法） |

## 示例文件

[`examples.jsonl`](examples.jsonl) — 5 条样本，覆盖上述五种题型（客观题来自验证集，简答来自训练集）。
[`train.jsonl`](train.jsonl) — 真实训练样本示例。
[`val.jsonl`](val.jsonl) — 真是验证样本示例。

`split` 字段仅作说明（`train` / `val`），正式数据文件中不含此字段。
