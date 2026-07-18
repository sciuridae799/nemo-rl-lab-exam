# QA Agent cold-start SFT

只使用训练集和 `/data/docs` 构造可在线复现的多轮正轨迹：预渲染题目、一次搜索、原始环境回灌、正确 `\boxed{}` 作答。

首轮配置只验证两件事，不执行 optimizer step：

1. 固定预算内能否得到足够的证据可答开放题轨迹；
2. F4 step 30 能否只加载模型权重，并以全新的 SFT optimizer/scheduler 完成 holdout 前向验证。

通过后才把 `data.weights_load_probe_only` 设为 `false`，运行 8-step SFT。

