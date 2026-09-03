# 迭代路线

总原则：**测绘先行，不盲赌**。训练前快速预测——以小规模代理实验验证数据 / 架构 / insight，再决定大规模投入。

## Phase 1：管线跑通（组集群）

- 基座：SmolVLA / DM0.5 base
- 任务：LIBERO 基线复现
- 数据管线：LeRobot v3 格式统一打通（LIBERO / OpenX / DROID）
- 出口标准：LIBERO 基线成功率复现至参考水平，数据 → 训练 → 评测全链路可跑

## Phase 2：数据 know-how + 配比测绘

- 小规模代理实验：验证数据来源、配比、架构 insight
- 性能~规模测绘（scaling 曲线），确定 3–5B 预训练配置
- 数据配比 know-how（参照 Isaac 0.5：视频 ↔ 遥操作 210× 量级）
- 出口标准：给出大规模预训练的模型规模 / 数据配比 / 训练量决策依据

## Phase 3：SuperPOD 64 卡预训练

- 规模：3–5B
- 时长：2–3 周（单轮预算：64 卡 × ≤1 个月）
- 数据：仅真实物理交互数据（人类穿戴 / 真机），不使用仿真数据
- 出口标准：基模产出，下游微调可行

## Phase 4：微调 + Benchmark 迭代 → 开源发布

- 评测：LIBERO → RoboDojo 能力诊断
- 目标：指标超越 DM0.5
- 发布物：模型权重 + Benchmark + Tech Report（Apache-2.0）
- 不发布：Data / Training Pipeline 代码（仅以技术报告公开方法）

## 保密

- 组会不公开，与 HiveServe 同级保密
- 仓库在 Phase 4 发布前保持 private
