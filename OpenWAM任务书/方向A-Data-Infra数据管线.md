# 方向 A：Data-Infra 数据管线

> 一句话定位：把 OpenWAM 的 13 种异构机器人数据如何统一成「一种训练样本」讲清楚，产出数据管线搭建指南与数据契约权威文档，并评估预训练数据混合的可复现边界。读者：第一次接触该模块的同学。
> 前置：先读 [README.md](README.md) 的总览。**本方向第一阶段（W1–W2）纯阅读+文档化，不需要任何 GPU/环境**。

约定：所有路径为仓库相对路径，仓库根记作 `OpenWAM/`。行号以 `main @ 90e94ae` 为准，行号漂移时以符号锚点为准。

## 0. 你的环境与资源路径

本方向自带最小环境路径，**不依赖任何其他方向的环境搭建成果**：

- **W1–W2（A1/A2/A4/A5/A7）**：零依赖。只需要一份 `OpenWAM/` 源码 checkout + 一个能 `import omegaconf/numpy` 的普通 Python（A5 跑 `build_dataset` 用合成数据，CPU 即可，不碰真实数据集、不需要 GPU）。不依赖方向 B 的模型权重、方向 C 的部署环境。
- **A3（归一化复现）**：CPU + 下载 LIBERO（1.9GB，最小的可下载数据集）：`python scripts/download_assets/download_benchmark_data.py --name LIBERO --yes`（下载器会自动建 stats 并回写 yaml）。无需 GPU。
- **A4（下载器对账）**：CPU + 任选 1–2 个已下载数据集抽验即可，可复用 A3 的 LIBERO。
- **A6（预训练混合）**：才需要大存储（AgiBotWorld-Beta / RoboCOIN_clean / InternData-A1 / OXE-DROID 合计数百 GB）与 HF 授权申请；建议 W2 就发起申请，等待期间继续做其他任务。仍然不需要 GPU——跑通 `mixture` 的一个 step 是 CPU 任务。
- 你**不依赖**：统一 Docker 镜像（方向 C）、模型权重下载（方向 B）、评测环境（方向 D）。本方向所有任务的验收都不需要 GPU。

## 1. 这个方向做什么、为什么值得做

OpenWAM-α 的卖点是「跨本体预训练」：同一模型在 518.5M 帧、来自十几种机器人/人类视角的数据上联合训练。这件事能成立，全靠数据层把千差万别的原始格式（HDF5、LeRobot v2.1/v3 parquet、不同臂数、不同动作语义）压成**同一个 canonical sample**：视频 PIL 列表 + `(T,80)` 动作张量 + 逐维 bool mask + prompt 字符串。80 维动作空间（80-D unified action space）和逐维 mask 是让单臂 LIBERO 和双臂 AgileX 能在同一个 batch 里训练的关键发明。

本方向不写模型代码，产出**文档、最小实现与验证记录**：管线搭建全景指南（A1）、80-D 映射总表（A2）、归一化统计链路复现（A3）、下载器对账（A4）、合成数据最小新数据集（A5，本方向唯一的动手搭建任务）、预训练混合复现攻坚（A6）、egocentric human 源定位（A7）。A6/A7 触及本仓库最大的复现缺口（预训练数据无下载脚本、EgoDex/Ego4D reader 未随仓库发布）。

你能学到：多源异构数据对齐的工程范式（hook 体系 + 双射映射 + 统计制品校验）、分布式训练下统计量的 rank0 构建/轮询协议、以及「如何给一个开源项目的数据管线做考古」。

## 2. 最小必要背景

**① registry 机制**——所有数据集类型用名字注册，训练时按 `cfg.dataloader.type` 查表实例化。
锚点：`openwam/dataloader/registry.py:75 _register_builtins`（`:91-106` 共 13 个注册名：robotwin, robodojo, agibotworld, mixture, robocasa_gr1, robocoin, ebench, libero, muka_franka, oxe_droid, robocasa365, interndata_a1, vlabench）；`registry.py:38 build_dataset` 强制每个类实现 `from_config(config, split)`（无此方法直接 TypeError，无 fallback）。

**② canonical sample**——每个 reader 的 `__getitem__` 必须吐出同一形状的字典。
锚点：`openwam/dataloader/bases/dataset.py:8 BaseDataset`；字段规范写在 `openwam/dataloader/bases/lerobot_v3_reader.py:12-24` docstring：

```python
{"video": [PIL.Image x num_video_frames], "vace_video": None,
 "first_frame_image": [video[0]],
 "action": (T-1, action_dim) float32, "action_mask": (T-1, action_dim) bool,
 "video_mask": (num_video_frames,) bool,
 "proprio": (1, action_dim) float32, "proprio_mask": (1, action_dim) bool,
 "prompt": str}
```

**③ LeRobotV3Reader hook 体系**——13 个 reader 里 8 个是 `LeRobotV3Reader` 的直接子类，差异全部通过类属性 + hook 表达，不设中间基类。
锚点：`openwam/dataloader/bases/lerobot_v3_reader.py:1140 CONFIG_KEYS`（yaml 可配字段白名单，`:1158 from_config` 逐 key 消费）；hook：`_resolve_cameras`（`:547`）、`_load_stats`（`:671`，reader 内置统计或 None）、`_action_20d`/`_proprio_20d`（`:788`/`:792`，返回归一化后的 20 维中间表示，返回 None 则发零张量=视频-only 样本）、`_finalize_action`/`_finalize_proprio`（`:912`/`:965`，20 维 → 80 维 scatter + 生成 2D mask）。

**④ 80-D unify 双射**——每个数据集的原始动作经 spec 字符串（如 `["0-9","34-43"]`）散落到 80 维槽位，可逆。
锚点：`openwam/dataloader/utils/unify_action.py:48 UNIFY_DIM = 80`；`:118 parse_unify_spec`；`:183 map_to_unify`（raw→unify）；`:215 unmap_from_unify`（unify→raw，部署侧用）。

**⑤ 归一化与 rot6d 钉住**——三模式 min-max / z-score / quantile；min-max 与 quantile clip 到 [-1,1]，z-score 不 clip；rot6d 维度统计量被钉为恒等，因为旋转基分量本就有界。
锚点：`openwam/dataloader/utils/normalization.py:82 apply_normalization`；`:58 pin_rot6d_identity`、`:43 ROT6D_DIMS_EEF20`、`:47 _ROT6D_IDENTITY`（`{min:-1, max:1, q01:-1, q99:1, mean:0, std:1}`）；`openwam/dataloader/transforms/normalize.py:135 normalize` / `:156 unnormalize`（部署侧反归一化同链路）。

**⑥ 多视角 L-shape 拼图**——384×320 画布：上方一整块 head 相机（256×320），底部左右两个半块腕部相机（128×160），缺省视角填黑。
锚点：`openwam/dataloader/transforms/multiview.py:23 crop_and_resize`、`:48 assemble_multiview_layout`；布局参数见 `configs/dataloader/robocasa365.yaml` 的 `camera_layout` 注释。

**⑦ MixtureDataset**——预训练混合的顶层数据集，本身无数据，按 `weight_strategy` 对各子源虚拟索引加权采样。
锚点：`openwam/dataloader/mixture.py:21 MixtureDataset`、`:257 from_config`、`:348` 起四种策略 `proportional`（默认，按窗口数比例）/ `uniform` / `inverse_size` / `manual`；配置文件 `configs/dataloader/pretrain_data/mixture.yaml`。

## 3. 代码地图

```
openwam/dataloader/
├── registry.py                 # ★ 入口：注册表 + build_dataset（:38）
├── bases/
│   ├── dataset.py              # BaseDataset ABC（canonical sample 契约源头）
│   ├── lerobot_v3_reader.py    # ★ LeRobot v3 单桶 reader 基类（hook 体系，1218 行）
│   └── multi_lerobot_v3_reader.py  # 多桶聚合基类
├── transforms/                 # normalize.py / multiview.py / video.py / rotation.py / builder.py
├── utils/
│   ├── unify_action.py         # 80-D 双射
│   ├── normalization.py        # apply_normalization + rot6d 钉住 + stats 物化
│   ├── lerobotv3.py            # 无状态共享 helper（DataContractError :94）
│   ├── eef.py / poses.py / oxe_schema.py / exclusion_io.py / video_io.py
│   └── stats_computation/      # 12 个按数据集的 stats 构建脚本
└── <13 个 reader 文件，见下表>
```

13 个 reader 一览（格式与动作维度契约；raw→unify 槽位）：

| 文件 | 格式 | raw 维度 → unify 槽位 | 下载 |
|---|---|---|---|
| `libero.py` | LeRobot v3 + `meta/conversion.json` | native delta EEF10 → [0-9] | 下载器一键 |
| `vlabench.py` | LeRobot v3 | EEF7→EEF10 → [0-9] | 下载器一键 |
| `robocasa365.py` | LeRobot v3 | state19/action15 非对称 → action [0-9],[68-72]；state [0-9],[68-76] | 下载器一键 |
| `robocasa_gr1.py` | LeRobot v3 | EEF33 → [0-8],[10-15],[34-42],[44-49],[68-70] | 下载器一键 |
| `robotwin.py` | HDF5（episode*.hdf5，JPEG 流） | joint14 / eef20 → [0-9],[34-43] | 下载器一键 |
| `robodojo.py` | HDF5 | EEF20 → [0-9],[34-43] | 下载器一键 |
| `ebench.py` | LeRobot v2.1 parquet | raw23（含 base3）→ [0-9],[34-43],[68-70] | 下载器一键 |
| `robocoin.py` | LeRobot v3 | EEF20 / dex(18+kL+kR) → [0-9],[34-43] | 手工 |
| `agibotworld.py` | LeRobot v3 | EEF20(+move2)/dex30 → [0-9],[34-43],[68/70] | 手工 |
| `interndata_a1.py` | LeRobot v3 | EEF20 → [0-9],[34-43] | 手工（`${oc.env:INTERNDATA_A1_ROOT}`） |
| `oxe_droid.py` | LeRobot v3 | 单臂 EEF10 → [0-9]（右侧 mask） | 手工（需 excluded_episodes.json） |
| `muka_franka.py` | LeRobot v3（15fps） | EEF7→EEF10 → [0-9] | 手工（HF 小仓） |
| `mixture.py` | 聚合（无自有数据） | 继承子源（强制同 action_dim/num_frames） | — |

**入口文件**：训练侧入口是 `scripts/train.py` 调 `build_dataset(cfg.dataloader, split)`；读代码从 `openwam/dataloader/registry.py` 进，沿 `libero.py`（最小 reader）→ `bases/lerobot_v3_reader.py` 主线走。

## 4. 任务书

### 任务 A1：数据管线搭建全景指南（registry → reader → canonical sample → transforms → train 入口）

- **目标**：写一份「数据如何流进训练」的主线文档，把整条装配链讲清楚：`scripts/train.py` → Hydra 组合 `configs/train.yaml` + `configs/dataloader/<name>.yaml` → `build_dataset(cfg.dataloader, split)`（`registry.py:38`）→ `from_config` 逐 key 消费 `CONFIG_KEYS` → reader `__init__`（读 `meta/info.json`、episode 拼接、`_filter_episodes`、`_load_stats`、`_resolve_cameras`）→ `__getitem__`（parquet 窗口 → `_action_20d` 归一化 → `_finalize_action` unify+mask → `_decode_window_video` L 形拼图）→ `collate_fn=list` 进 batch。同时涵盖 canonical sample + 2D mask 契约（旧 D1 内容：mask 的 per-dim 有效性语义、2D 掩码迁移的历史原因）。
- **前置**：读完 §2 全部 7 条背景；`assets/openwam_usage_docs/benchmark-integration.md` 与 `assets/openwam_usage_docs/train-and-deploy.md`。
- **步骤**：
  1. 沿 §3 代码地图把装配链每一环的符号+行号抄进笔记，画出调用链图；
  2. 用最小 fixture 打印一个真实 sample 的各字段 shape/dtype（复用 `tests/dataloader/conftest.py` 的 fixture），对照 `benchmark-integration.md` 的字段表逐字段注释；
  3. 画一页示意图：80 维向量按臂/手/base 分段，标注每段的 mask 取值规则（与 `_finalize_action` 实现逐维对上）；
  4. 补一节「如何新增一个数据集类型」的 10 行速览（细节留给 A5 实操）。
- **验收标准**：全景指南（含调用链图 + 字段契约表 + 80-D mask 示意图）经组内评审；示意图与 `bases/lerobot_v3_reader.py:912 _finalize_action` 实现逐维对得上。
- **产出**：管线搭建全景指南 1 份（作为本方向后续所有任务的公共底座，也供方向 B/F 引用）。
- **难度（估计）**：低-中。**工作量（估计）**：2–3 天。
- **代码锚点**：`registry.py:38,75,91-106`；`bases/lerobot_v3_reader.py:12-24,547,671,912,965,1140,1158`；`tests/dataloader/test_2d_mask_migration.py`。

### 任务 A2：80-D unified action space 语义总表

- **目标**：汇总全部 13 个 reader 的 `unify_action_map`/`unify_state_map` 为一张统一槽位语义总表，标注跨数据集冲突（同一槽位不同语义）与 gripper 极性差异。
- **前置**：A1；`openwam/dataloader/utils/unify_action.py:118 parse_unify_spec`；各 `configs/dataloader/*.yaml`。
- **步骤**：
  1. 逐文件提取 map：`grep -n "unify_action_map\|unify_state_map" configs/dataloader/*.yaml configs/dataloader/pretrain_data/*.yaml`；
  2. 重点核对非对称样例 `configs/dataloader/robocasa365.yaml`（state19/action15）与 gripper 反转注释 `openwam/dataloader/agibotworld.py:10`（AgiBot `0=open,1=closed`，reader 内反转）、`openwam/dataloader/vlabench.py:26-41`（state/action 极性相反是上游 bug，**保持原样**，eval 客户端读同一个有 bug 的 accessor，rollout 时反转抵消）；
  3. 写一个小脚本，对每个 spec 调 `parse_unify_spec` 展开成 80 维布尔占用图，汇总成表。
- **验收标准**：映射总表（数据集 × 槽位段 × 语义 × 极性）+ 跨数据集冲突清单；总表与 `tests/dataloader/test_unify_action.py` 行为一致。
- **产出**：80-D 语义总表 1 份 + 冲突清单。
- **难度（估计）**：中。**工作量（估计）**：2–3 天。
- **代码锚点**：`utils/unify_action.py:48,118,183,215`；`configs/dataloader/robocasa365.yaml`；`vlabench.py:26-41`；`agibotworld.py:10`。

### 任务 A3：归一化统计链路复现（LIBERO 样本）

- **目标**：以 LIBERO（1.9GB，最小数据集）端到端跑通 stats 计算，验证 rot6d 恒等钉住与 train/deploy 两侧归一化一致（parity）。
- **前置**：CPU + 下载 LIBERO：`python scripts/download_assets/download_benchmark_data.py --name LIBERO --yes`（下载器自动建 stats 并回写 yaml）。**无 GPU 需求**。
- **步骤**：
  1. 下载后删除 `meta/libero_normalization_stats.npy`，手动重跑 `openwam/dataloader/utils/stats_computation/libero_stats_computation.py` 重建；
  2. 检查 rot6d 维度（`utils/normalization.py:43 ROT6D_DIMS_EEF20`）的统计量是否等于恒等值（`:47 _ROT6D_IDENTITY`）；
  3. 取同一 raw 向量，分别走 `utils/normalization.py:82 apply_normalization`（train 侧）与 `transforms/normalize.py:135 normalize` + `:156 unnormalize`（deploy 侧往返），断言 round-trip 误差 < 1e-5；
  4. 阅读 rank0 构建/其余 rank 轮询协议：stats 缺失时非 0 rank 等待，超时由 `OPENWAM_STATS_WAIT_TIMEOUT_S` 控制（默认 12h，见 `robocasa_gr1.py:184`、`ebench.py:590`、`libero.py:187`）。
- **验收标准**：重建的 stats 与下载器产物数值一致；rot6d 恒等钉住验证记录；train/deploy parity 冒烟脚本通过。
- **产出**：stats 链路验证记录 + parity 冒烟脚本。
- **难度（估计）**：中。**工作量（估计）**：2–3 天（含下载）。
- **代码锚点**：`utils/normalization.py:43,47,58,82`；`transforms/normalize.py:135,156`；`configs/dataloader/libero.yaml`（`normalize_mode: min-max`）。

### 任务 A4：下载器 ↔ reader ↔ config 对账

- **目标**：把 `scripts/download_assets/download_benchmark_data.py` 的下载条目与各 reader、各 yaml 逐条对账，产出「数据集 × 下载条目 × reader × config × stats 步骤」对应表。
- **前置**：A1 的代码地图；CPU + 1–2 个已下载数据集用于抽验（可复用 A3 的 LIBERO）。
- **步骤**：
  1. 列出 `BENCHMARKS`（`download_benchmark_data.py:81`，编号 1–8：RoboTwin2.0 / RoboDojo / RoboDojo-Real / LIBERO / VLABench / EBench / RoboCasa365 / RoboCasa_GR1）每条目的 HF repo、`dataset_subpath`、`allow_patterns`、目标 yaml；同时记录体量估算（LIBERO 1.9GB → RoboDojo 488GB）；
  2. 对照 §3 的 13 reader 表，标出无下载条目的 5 个（robocoin / agibotworld / interndata_a1 / oxe_droid / muka_franka）；
  3. 抽 1–2 个已下载数据集核对 `_POST_DOWNLOAD`（`:360`，解压/重定位）与 `_STATS_STEPS`（`:588`）的产物落点、回写字段（`dataset_dir`）是否和 reader 的 `_load_stats` 预期一致。
- **验收标准**：对应关系表 + 缺口报告（哪些 reader 无下载路径、哪些 yaml 字段依赖回写）。
- **产出**：对账表 1 份。
- **难度（估计）**：低。**工作量（估计）**：1 天。
- **代码锚点**：`download_benchmark_data.py:81,360,588`；`configs/dataloader/`。

### 任务 A5：合成数据最小新数据集注册（动手搭建）

- **目标**：按 `assets/openwam_usage_docs/benchmark-integration.md` §1 的扩展约定，用**合成数据**实现并注册一个最小新数据集类型，跑通 `build_dataset`——亲手把 A1 文档化的装配链走一遍。
- **前置**：A1 完成（知道 canonical sample 每个字段的语义）；CPU Python 环境。
- **步骤**：
  1. 造合成数据：一个临时目录下写 2 个 episode 的最小 LeRobot v3 布局（或直接 subclass `BaseDataset` 走更轻的路线，见 `benchmark-integration.md:18-30` 的 `MyBenchmarkDataset` 模板）；
  2. 实现 `from_config` / `__len__` / `__getitem__`，`__getitem__` 吐出全字段 canonical sample（视频-only 路径：零 action + 全 False `action_mask`，注意零值本身不关闭 loss，mask 才关闭）；
  3. `@register_dataset('synthetic_demo')` 注册，写 `configs/dataloader/synthetic_demo.yaml`（`type: synthetic_demo`、`num_frames: 33`、`multiview: false`、`normalize_mode: null`）；
  4. 用 registry 检验一个样本：
     ```bash
     cd OpenWAM && python -c "
     from omegaconf import OmegaConf
     from openwam.dataloader.registry import build_dataset
     cfg = OmegaConf.load('configs/dataloader/synthetic_demo.yaml')
     ds = build_dataset(cfg, split='train')
     s = ds[0]
     print(len(ds), s['action'].shape, s['action'].dtype, s['action_mask'].sum())"
     ```
  5.（进阶）改用 `LeRobotV3Reader` 子类 + `unify_action_map`，验证 `map_to_unify`/`unmap_from_unify` round-trip。
- **验收标准**：`build_dataset` 成功实例化并吐出形状/类型全对的 canonical sample；字段与 `benchmark-integration.md:38-55` 的契约表逐条对上。
- **产出**：可运行的最小 reader 示例（放组内共享目录或 playground 分支，**不入主仓库**）+ 一页踩坑记录。
- **难度（估计）**：中。**工作量（估计）**：2–3 天。
- **代码锚点**：`assets/openwam_usage_docs/benchmark-integration.md:10-115`；`registry.py:38,58-63`；`bases/dataset.py:8`；`utils/unify_action.py:183,215`。

### 任务 A6：预训练混合可复现性攻坚（reader/mixture 机制侧）

- **目标**：补齐 AgiBotWorld / RoboCOIN / InternData-A1 / OXE-DROID 四个预训练源的获取路径，在本组存储上跑起 `mixture` 数据集的一个 step，并把 `MixtureDataset` 的加权采样**机制**（四种 `weight_strategy`、虚拟索引、strict 模式同 action_dim/num_frames 强制）验证清楚。
- **前置**：A4 完成；大容量存储（各源合计数百 GB）；`configs/dataloader/pretrain_data/mixture.yaml`（当前仅 4 个机器人源，`weight_strategy: proportional`）。**与方向 F 的分工**：本任务只管「数据能不能进来、按什么权重采样」的机制层；OpenWAM-α 的实际数据配方（各源配比、训练时长安排）归 [方向F-OpenWAM-Alpha模型实现.md]。
- **步骤**：
  1. 申请/下载四个 HF 数据集：AgiBotWorld-Beta、RoboCOIN_clean、InternData-A1、OXE-DROID（DROID 还需按 `openwam/dataloader/oxe_droid.py` 要求准备 `meta/excluded_episodes.json`；InternData-A1 用环境变量 `INTERNDATA_A1_ROOT` 指向）；
  2. 把各 yaml 的 `/path/to/pretrain_dataset/...` 占位（如 `configs/dataloader/pretrain_data/agibotworld.yaml:5`、`oxe_droid.yaml:5`、`robocoin.yaml:6`）改成本组实际路径；
  3. 逐源单测：`python -m pytest -q tests/dataloader/test_robocoin_trim.py tests/dataloader/test_interndata_a1.py`（无真实数据时先用 fixture 验证逻辑）；
  4. 机制验证：分别用四种 `weight_strategy` 构建 mixture，打印各子源的有效采样比例，与 `mixture.py:348` 起的实现对照；
  5. 最小冒烟（CPU）：
     ```bash
     cd OpenWAM && python -c "
     from openwam.dataloader.registry import build_dataset
     from omegaconf import OmegaConf
     cfg = OmegaConf.load('configs/dataloader/pretrain_data/mixture.yaml')
     ds = build_dataset(cfg, split='train')
     print(len(ds), ds[0]['action'].shape, ds[0]['action_mask'].sum())"
     ```
  6. 核算各源帧数/小时数（`total_hours` 旋钮），估算与论文 518.5M 帧的差距（结论交给方向 F 的配方任务消费）。
- **验收标准**：可运行的混合配置 + 一个 step 的真实样本输出 + 四种 weight_strategy 的机制验证记录 + 体量核算表；跑不起来的源在报告中注明卡点（授权/体积/制品）。
- **产出**：混合配置 + 机制验证记录 + 体量核算 + 可复现性结论。
- **难度（估计）**：高。**工作量（估计）**：1–2 周（主要在数据申请与传输）。
- **代码锚点**：`mixture.py:21,257,348`；`configs/dataloader/pretrain_data/`；`utils/exclusion_io.py`。

### 任务 A7：egocentric human 源定位结论

- **目标**：确认 EgoDex/Ego4D reader 缺席的证据链，评估自实现一个 EgoDex-style reader 的工作量。
- **前置**：A2 的槽位语义理解。**与方向 F 的分工**：本任务只出「是否存在/要不要补」的定位结论；若决定补齐并用于 alpha 训练配方，实现与接入归 [方向F-OpenWAM-Alpha模型实现.md]。
- **证据链（已核验，行号以 90e94ae 为准）**：
  1. `openwam/dataloader/bases/lerobot_v3_reader.py:4-7` docstring 宣称「the 4 OXE readers (BC-Z / Bridge / Fractal / DROID), RoboCOIN, and **EgoDex** are all direct, equal subclasses of `LeRobotV3Reader`」；
  2. `openwam/dataloader/utils/lerobotv3.py:3` docstring 称「Both `RoboCOINDataset` and `EgoDexDataset` follow the same setup recipe」；`bases/multi_lerobot_v3_reader.py:75` 注释提及 `MultiBucketEgoDex`；
  3. 但 `openwam/dataloader/registry.py:91-106` 的 13 个注册名中**没有** egodex/ego4d；
  4. 全仓 grep 无任何 `class EgoDex*`/`class Ego4D*` 定义——reader 未随仓库发布；
  5. `tests/dataloader/test_oxe_mixture_integration.py:68-69` 显式把 `{ego4d, egodex, oxe_bcz, oxe_bridge, oxe_fractal}` 列为 deprecated 并断言未注册。
- **步骤**：
  1. 复核上述 5 条证据并记录；
  2. 以 `robocoin.py` 为模板起草 EgoDex-style reader 实现方案：需要哪些 hook（`_resolve_cameras`、`_action_20d`、stats 路径）、EgoDex 原始格式到 80-D 的映射设计、统计制品要求；
  3. 给出工作量评估（hook 覆写数量、测试 fixture 构造、风险点），形成「做/不做」建议。
- **验收标准**：结论备忘录（含完整证据链）+ reader 实现草案；若决定实现，移交给方向 F 立项。
- **产出**：备忘录 1 份（+ 可选实现草案）。
- **难度（估计）**：中-高。**工作量（估计）**：2–4 天（仅定位+评估）。
- **代码锚点**：见上证据链 5 条。

## 5. 建议推进顺序与里程碑

1. **W1**：A1 → A4 → A7（纯阅读+文档化，零外部依赖）；里程碑：全景指南 + 对账表 + EgoDex 备忘录证据链复核。
2. **W2**：A2 → A5（动手搭建）→ 发起 A6 的 HF 数据申请（周期长，越早越好）；里程碑：80-D 总表 + synthetic_demo 跑通。
3. **W3**：A3（依赖 LIBERO 下载，CPU 即可）；里程碑：stats 链路验证记录 + parity 脚本。
4. **W3–W4+**：A6（数据到位即推进，等待期间可支援其他方向）；里程碑：混合可复现性报告。
里程碑汇总：全景指南（W1）→ 映射总表 + 最小 reader（W2）→ parity 记录（W3）→ 混合可复现性报告（W4+，受数据申请制约）。

## 6. 风险与坑

- **测试 fixture 缓存不自动失效**：`tests/dataloader/conftest.py:18` 把生成数据缓存在 `tests/dataloader/.cache/`；schema/代码变更后旧 fixture 会导致假失败。规避：改 reader 后先 `rm -rf tests/dataloader/.cache` 再跑测试。
- **改 trim/exclusion 后必须重算 stats**：AgiBotWorld 的 `trim_csv`、DROID 的 `excluded_episodes.json`、RoboCOIN 的整桶 exclusion 改变数据分布后，旧 stats 会在加载时抛 `DataContractError`（`openwam/dataloader/utils/lerobotv3.py:94`；stats 带 population/provenance 校验）。规避：任何 trim/exclusion/分片变更后删除旧 stats 重建。
- **VLABench 上游 meta/episodes 分片索引错误**：上游把 `dataset_from_index = length * episode_index` 写错（3953 分片 / 5000 episode），reader 已用真实 parquet 行数重建 episode→分片映射（`openwam/dataloader/vlabench.py:97-127`）。规避：不要「修复」回信任上游 meta 的版本；升级数据版本时需重新验证该假设。
- **VLABench state/action 极性相反是上游 bug，保持原样**：`vlabench.py:26-41` 说明 eval 客户端读同一个有 bug 的 accessor，rollout 时反转恰好抵消。规避：绝不在 reader 里「修正」极性，否则训练/评测分布错位。
- **`configs/dataloader/pretrain_data/` 全是 `/path/to` 占位**：预训练混合无下载脚本，OpenWAM-α 从头预训练不可完整复现。规避：把 A6 定位为准入评估而非承诺；以官方 checkpoint 微调为主路径（权重下载归 [方向B-Model-Infra模型管线.md]）。
- **stats 轮询假死**：多卡训练时 stats 缺失会让非 0 rank 静默轮询最长 12h（`OPENWAM_STATS_WAIT_TIMEOUT_S`，见 `robocasa_gr1.py:184`、`ebench.py:590`、`libero.py:187`）。规避：单卡先跑通建 stats，再上多卡；必要时调小该环境变量（配 `OPENWAM_STATS_POLL_INTERVAL_S`）快速失败。

## 7. 推荐阅读顺序

1. 本目录 `README.md`——知道 13 数据集全景与本方向在 6 管线中的位置。
2. `openwam/dataloader/registry.py`（全文很短）——注册机制与 13 个名字。
3. `openwam/dataloader/bases/lerobot_v3_reader.py:1-52` docstring——canonical sample 与 hook 总览（本方向最重要的 52 行）。
4. `openwam/dataloader/libero.py` + `configs/dataloader/libero.yaml`——最小具体 reader，看 hook 如何被实例化。
5. `openwam/dataloader/utils/unify_action.py`——80-D 双射全部实现（约 230 行，可全读）。
6. `openwam/dataloader/utils/normalization.py:42-114` + `transforms/normalize.py:135-160`——归一化正反链路。
7. `openwam/dataloader/transforms/multiview.py`——L-shape 拼图。
8. `assets/openwam_usage_docs/benchmark-integration.md` §1——A5 的扩展约定原文。
9. `openwam/dataloader/mixture.py` + `configs/dataloader/pretrain_data/mixture.yaml`——混合采样（A6 前必读）。
10. `openwam/dataloader/vlabench.py:1-60` 与 `agibotworld.py:1-30` 的文件头注释——上游数据坑的一手记录（A2/A7 的考古范本）。

相关方向：[方向B-Model-Infra模型管线.md]（消费 canonical sample 的模型侧）；[方向F-OpenWAM-Alpha模型实现.md]（A6/A7 的配方与实现承接方）。
