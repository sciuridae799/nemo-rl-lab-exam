# common/environments — 自定义环境（奖励来源）

NeMo-RL 里 GRPO 的奖励由 **Environment** 产生（而非独立 reward 函数）。把跨实验复用的
自定义环境放这里：

- 数学/通用单轮任务：通常用内置环境（配置 `data.default.env_name=math` 等），无需自写。
- 多轮 Agent / 工具调用：实现自定义 Environment + 自定义 run 脚本。
- 单轮自定义判分（如考试 QA）：`common/environments/qa_env.py` 的 `QARewardEnv` + `common/rewards/`。
- 多轮本地检索 QA：`qa_search_core.py` 提供可测试的检索与状态机，`qa_search_env.py` 提供 Ray 环境包装。

参考实验：

- `agent-grpo_qwen3.5-9b_sliding-puzzle_v1`（多轮 Agent，NeMo-RL 自带拼图环境）
- `grpo_qwen3.5-9b_qa-rl_v1`（单轮 QA 判分环境 + 自定义 `run.py`）
- `grpo_qwen3.5-9b_qa-rl-agent_v1`（多轮 Markdown 检索 + QA 判分）
