# QA Agent cold-start SFT

只使用训练集和 `/data/docs` 构造可在线复现的多轮正轨迹：预渲染题目、一次搜索、原始环境回灌、正确 `\boxed{}` 作答。

无更新探针已验证两件事：

1. 固定预算内能否得到足够的证据可答开放题轨迹；
2. F4 step 30 能否只加载模型权重，并以全新的 SFT optimizer/scheduler 完成 holdout 前向验证。

当前配置进入正式短跑：开放/封闭轨迹各最多 128 条，按规范化题干 90/10
切分；只加载 F4 step 30 模型权重并重建 optimizer/scheduler，最多运行
8-step SFT。训练数据不足门槛、加载失败、holdout 前向异常时均不继续。
