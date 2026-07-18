# QA Agent cold-start SFT

只使用训练集和 `/data/docs` 构造可在线复现的多轮正轨迹：预渲染题目、一次搜索、原始环境回灌、正确 `\boxed{}` 作答。

前序无更新探针已验证两件事：

1. 固定预算内能否得到足够的证据可答开放题轨迹；
2. F4 step 30 能否只加载模型权重，并以全新的 SFT optimizer/scheduler 完成 holdout 前向验证。

R4 的 8-step SFT 虽降低 holdout loss，但没有改善开放题行为。复盘发现它允许
部分证据轨迹监督完整答案，且搜索动作也参与 loss，因而 loss 下降不能识别证据
抽取能力。

当前 R5 配置只保留 `evidence_coverage=1.0` 的开放题轨迹，混入 96 条封闭题
能力回放，并只监督最终 assistant 回合。数据仍按规范化题干 90/10 切分；只加载
F4 step 30 模型权重并重建 optimizer/scheduler，最多运行 4-step SFT。开放题少于
80 条、加载失败或 holdout 前向异常时均不继续。
