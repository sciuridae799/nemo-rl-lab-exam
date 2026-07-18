# grpo_qwen3.5-9b_qa-rl-agent_v1

在单轮 QA 基线之上增加本地 Markdown 多轮检索，模型每轮输出：

```text
<search>关键词</search>
```

环境从 `/data/docs` 返回相关片段；模型最多检索两次，最终输出：

```text
<answer>简短依据；\boxed{答案}</answer>
```

## 设计要点

- 数据和最终判分复用 `grpo_qwen3.5-9b_qa-rl_v1`。
- 中文字符二元组 + 英文词项的 BM25 风格检索，无外部模型和网络依赖。
- 检索轮奖励为 0，防止污染 `validation/accuracy`。
- 每次最多返回 3 个片段、1200 个正文字符，控制多轮上下文长度。

## 提交

```bash
lab validate grpo_qwen3.5-9b_qa-rl-agent_v1
lab submit grpo_qwen3.5-9b_qa-rl-agent_v1
```

真实 run、截图与复盘统一在仓库外维护，不随训练代码打包。

当前 R5 行为门从完整证据、最终答案监督的 SFT step 4 只加载权重，重建
GRPO optimizer/scheduler，并只运行 128 题 step-0 validation。开放题至少达到
`3/19` 正分、至少一条填空正分，且封闭题与格式保护门通过后，才关闭
validation-only 运行最多 3-step GRPO。
