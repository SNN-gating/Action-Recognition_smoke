# Action-Recognition smoke

此为一个真实事件相机动作识别smoke实验。输入来自 Zenodo 记录
`10.5281/zenodo.3228846` 的四个 DVS128 会话。模型为轻量卷积+LIF+SNN，并在第一层
脉冲之后加入可学习的时序通道门控。
共有两个轻量模型：

1. `baseline`：门恒为 1。
2. `gated`：门由当前事件密度与上一时刻通道脉冲率产生。

## smoke结构说明

Action-Recognition_smoke/
├─ data/                 # 原始少量数据、处理缓存
├─ src/                  # 数据、模型、训练代码
├─ results/              # 日志/训练曲线
├─ reports/              # 架构说明、实验结果报告、数学建模
