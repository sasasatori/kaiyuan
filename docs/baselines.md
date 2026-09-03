# 参考标杆调研结论

| 标杆 | 规模 | 要点 | 对本项目的意义 |
| --- | --- | --- | --- |
| **DM0.5**（原力灵机） | 4B | RoboDojo 第一；全栈开源（Apache-2.0） | 当前最直接对标：性能要超它，开源形态对齐它 |
| **π0.5**（Physical Intelligence） | 3.3B | flow-matching VLA，开放泛化标杆 | 性能 baseline 的原始参照 |
| **GR00T N1.7**（NVIDIA） | 3B | 开源 VLA，双系统设计 | 架构参考之一 |
| **Isaac 0.5** | 36B MoE | 数据配比定律：视频 ↔ 遥操作 210× | Phase 2 配比测绘的参照系 |
| **Generalist** | — | 纯物理交互数据预训练 + scaling 定律 + ICL 涌现 | 数据路线与本项目一致：预训练不用仿真数据 |

结论：性能标杆看 DM0.5 / π0.5，数据路线学 Generalist，配比方法论学 Isaac 0.5。
