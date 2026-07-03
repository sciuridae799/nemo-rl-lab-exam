# common/environments — 自定义环境（奖励来源）

NeMo-RL 里 GRPO 的奖励由 **Environment** 产生（而非独立 reward 函数）。把跨实验复用的
自定义环境放这里：

- 数学/通用单轮任务：通常用内置环境（配置 `data.default.env_name=math` 等），无需自写。
- 多轮 Agent / 工具调用：实现自定义 Environment + 自定义 run 脚本。

> 实现要点：环境是 Ray actor；多轮训练需要**自定义 run 脚本**来喂数据和环境（纯改配置不够）；
> `step()` 的返回结构、`DatumSpec` 字段以你装的 0.6.0 源码为准（参考 `nemo_rl/environments/` 内置环境）。

参考实验：`agent-grpo_qwen3.5-9b_sliding-puzzle_v1`（多轮 Agent，NeMo-RL 自带拼图环境）。
