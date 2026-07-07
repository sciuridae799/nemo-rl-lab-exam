# experiments/ — 练习 / 探索性实验

放调参、试错、复现、学习类的微调实验。每个子目录是一个独立实验，命名见 `docs/naming-convention.md`。

新建：

```bash
cp -r templates/experiment-template experiments/grpo_qwen3.5-9b_gsm8k_v1
```

要求（即使是练习）：每个实验都要有 `README.md`，记录目标、关键超参、结论、SwanLab 链接。

## 考生示例实验

| 实验 | 说明 |
| --- | --- |
| `grpo_qwen3.5-9b_gsm8k_v1` | 单轮 GRPO + 官方 `ResponseDataset`（数学） |
| `grpo_qwen3.5-9b_qa-rl_v1` | 单轮 GRPO + 自定义 QA 判分环境 + `run.py`（考试数据集格式） |
| `agent-grpo_qwen3.5-9b_sliding-puzzle_v1` | 多轮 Agent GRPO（无外部依赖，先跑通多轮链路） |
