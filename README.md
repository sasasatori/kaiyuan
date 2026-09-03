# 开元 Kaiyuan

**面向灵巧操作的具身智能基础模型 · 自训开源具身基模**

> 「开元」——谐音「开源」，取年号「开元盛世」之意，亦有开天辟地之志。

## 项目定位

国内「开天辟地」级工作：自训并开源面向灵巧操作的具身智能基础模型，建立开源影响力。

## 核心目标

1. **开源**：模型 + Benchmark + Tech Report 对外开源（Data / Training Pipeline 不开源，仅以技术报告公开）
2. **性能**：泛用 + 指标超越 π0.5；当前标杆已升级为 **DM0.5**（RoboDojo 第一 + 全栈开源）
3. **影响力**：成为国内具身基模的代表性开源工作

## 关键策略

| 维度 | 决策 |
| --- | --- |
| 预训练数据 | 仅用**实际采集的物理交互数据**（人类穿戴 / 真机），**不用仿真数据**（Generalist 路线；仿真仅用于提示 / 演示 / 评测） |
| 模型规模 | 3–7B 区间（对标 π0.5 3.3B / DM0.5 4B / GR00T N1.7 3B） |
| 算力 | HKUST SuperPOD，**64 卡/轮，至多 1 个月** |
| Benchmark | LIBERO 起步（复现基线）+ RoboDojo 能力诊断 |
| 方法论内核 | **训练前快速预测**：先做性能~规模测绘 + 数据配比 know-how（小规模代理验证数据 / 架构 / insight），再上大规模——不盲目堆算力 |

## 管线四要素

| 要素 | 内容 | 目录 |
| --- | --- | --- |
| 数据 | LIBERO / OpenX / DROID 等，LeRobot v3 格式统一 | [`data/`](data/) |
| 模型 | 参考 GR00T / π0.5 / OpenDM（DM0.5 微调生态可起步） | [`model/`](model/) |
| Infra | SuperPOD 64 卡预训练 + 组集群微调 | [`infra/`](infra/) |
| Benchmark | LIBERO → RoboDojo | [`benchmark/`](benchmark/) |

## 迭代节奏

- **Phase 1**：管线跑通（组集群，SmolVLA / DM0.5 base + LIBERO 复现）
- **Phase 2**：数据 know-how + 配比测绘（快速预测方法验证）
- **Phase 3**：SuperPOD 64 卡预训练（3–5B，2–3 周）
- **Phase 4**：微调 + Benchmark 迭代 → 开源发布

详见 [docs/roadmap.md](docs/roadmap.md)。

## 参考标杆

- **DM0.5**（原力灵机）：RoboDojo 第一 + 全开源（Apache-2.0）——当前最直接对标
- **Isaac 0.5**：36B MoE + 数据配比定律（视频 ↔ 遥操作 210×）
- **Generalist**：纯物理交互数据预训练 + scaling 定律 + ICL 涌现——数据路线与本项目一致

详见 [docs/baselines.md](docs/baselines.md)。

## 一句话

开源、超 DM0.5、用真实物理交互数据、64 卡一个月级训练、测绘先行不盲赌——做一个国内开天辟地的具身基模。

## License

[Apache-2.0](LICENSE)
