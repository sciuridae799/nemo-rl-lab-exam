# sft_qwen3.5-9b_qa-direct_v1

用 QA 训练集的全部五类题做直接答案 SFT，解决 GRPO 在开放题上奖励稀疏、同组零方差导致没有有效梯度的问题。

数据只来自 `/data/datasets/qa_rl/train.jsonl`：按规范化题干去重，同题答案冲突时整组丢弃，用稳定 SHA-1 分组留出约 5% 监控集。不读取官方验证答案，不构造搜索轨迹，每条样本只监督 `<answer>\boxed{...}</answer>`。

训练从当前最佳 F4 step 30 只加载模型权重，重建 optimizer/scheduler。配置为 LoRA rank 8/alpha 16、global/micro batch `64/2`、恒定 LR `5e-7`、最多 16 step，每 4 step 记录 holdout loss 并保存 checkpoint。最终是否改善仍以官方 QA Agent validation 为准，不用 SFT loss 代替真实指标。
