# 方向 F：OpenWAM-α 模型实现解剖

> 一句话定位：把 OpenWAM-α 当作一个具体的 artifact 解剖——它的 checkpoint 目录里有什么、config.yaml 声明了什么架构、80 维 action/state 契约如何约束上下游、预训练配方实际是什么、如何微调与部署它。读者：第一次接触该模块的同学。
> 前置：先读 [README.md](README.md) 的总览。**本方向第一阶段（W1–W2）纯阅读+文档化，不需要任何 GPU/环境**。
> 边界：骨干/架构机制调研归 [方向B-Model-Infra模型管线.md](方向B-Model-Infra模型管线.md)；reader 与数据管线机制归 [方向A-Data-Infra数据管线.md](方向A-Data-Infra数据管线.md)；alpha 是各 study 消融结论的产物，结论本身见 [方向E-Study消融实验实现.md](方向E-Study消融实验实现.md)。本方向只管"alpha 这个具体模型长什么样"。

仓库根 `$R = OpenWAM`，下文所有路径均为仓库相对路径。行号以 main @ 90e94ae 为准；行号漂移时以符号名为准。

## 0. 你的环境与资源路径

| 阶段 | 任务 | 资源需求 |
|---|---|---|
| W1–W2（纯阅读） | F2、F3、F6 | 零依赖：只需仓库源码 + 官方文档，CPU 笔记本即可 |
| 解剖阶段 | F1 | 下载 1 个 checkpoint（OpenWAM-Alpha-Pretrain-Foundation-Model，约 24.8 GB，`download_openwam_checkpoints.py:103`）；**CPU 即可**解剖 config.yaml 与 safetensors 键结构，不需要 GPU |
| 微调阶段 | F4 | 4×80GB 级 GPU（官方实测 2×H100 在首个 Adam step 即 OOM，见 `assets/openwam_usage_docs/docker-validation.md:68`）；全方向唯一重资源任务，排最后 |
| 冒烟阶段 | F5 | 1 张 ≥24GB GPU 起部署 server + LIBERO benchmark client |

你不依赖别人的工作：F1/F2/F3/F6 完全独立；F4 只依赖 checkpoint 下载与 LIBERO 数据（下载器一键）；F5 复用 [方向C-Deployment-Infra部署管线.md](方向C-Deployment-Infra部署管线.md) 的部署流程，但不依赖其产出物——你可以直接用官方 Sim-LIBERO 微调 checkpoint 冒烟。磁盘规划：每个 checkpoint 约 24.8GB，若做 F6 全谱系下载需预留 ≥350GB。

## 1. 这个方向做什么、为什么值得做

OpenWAM-α 是仓库的实际交付物：一个在 518.5M 帧（约 6400 小时）ego-centric human + robot 数据上预训练的 WAM（`README.md:23`），外加 13 个下游微调变体。但"alpha"不是代码里的一个类——它是**一组配置选择 + 一组权重 + 一套数据契约**的组合物：dual_system/joint_self_attn 架构 + wan22_ti2v_5b 骨干 + mutual 掩码 + 80 维 unify 动作空间 + min-max 归一化 + 4 源预训练混合。

为什么值得做：全组所有下游工作（部署、评测、微调）都在消费 alpha checkpoint，但"它到底宣称了什么、实际是什么"需要一份权威解剖。你把 config.yaml 的每个关键键、checkpoint 的权重键结构、80 维契约的文档与代码双侧实现钉死，其他人就不必各自考古。同时 F3 的配方审计回答一个尖锐问题：论文宣称的 518.5M 帧，有多少是代码可验证的？

你能学到：自包含 checkpoint 的工程范式（config+权重+统计制品+部署资产一个目录打包）、checkpoint 考古方法论（从 safetensors 键反推架构）、以及"文档宣称 vs 代码事实"的对账技术。

## 2. 最小必要背景

**① 自包含 checkpoint 目录**。下载器 docstring 明示：每个 checkpoint 目录自包含 config.yaml + 权重 + tokenizer + 归一化统计（`scripts/download_assets/download_openwam_checkpoints.py:7-8`）。部署直接 `bash scripts/deploy.sh <ckpt_dir_path>`，不需要原始训练配置。

**② finetune_ckpt_path 重建机制**。`openwam/train/utils/ckpt_model_loader.py`：`:216 build_architecture_from_ckpt_dir` 从 checkpoint 目录的 config.yaml 重建模型（不需原始 `model_path`）；`:149 merge_ckpt_model_cfg` 以 ckpt 的 model 配置为基底、本 run 的 `cfg.model` 覆盖之；`:60 _ARCH_IDENTITY_KEYS` 只有两个键——`video_backbone.name` 与 `architecture.framework`——不一致直接拒绝启动。这就是为什么微调时可以不传 `model=...`：架构身份从 ckpt 继承。

**③ 80 维 unify 契约的代码侧实现**。`openwam/dataloader/utils/unify_action.py:48 UNIFY_DIM = 80`；`:118 parse_unify_spec` 把 `["0-9","34-43"]` 这类 spec 解析为索引数组；`:183 map_to_unify`（raw→80 维 scatter，训练侧）；`:215 unmap_from_unify`（80 维→raw，部署侧）。契约语义（槽位布局、gripper 极性、rot6d 钉住）的权威来源是 `assets/openwam_usage_docs/openwam-alpha-finetuning.md:46-89`。

**④ alpha 默认架构链**。README Quick Start 表（`README.md:287-290`）：`dual_system / joint_self_attn` + `wan22_ti2v_5b` + `attention_mask_mode=mutual` + LIBERO。对应配置 `configs/model/dual_system.yaml:21 variant: joint_self_attn`、`:29 attention_mask_mode: mutual`、`:23 action_dim: 80`、`:25 state_dim: 80`。

**⑤ 采样-时序耦合**。`README.md:381-383`：`num_frames=33` + `video_stride=4` 产生 sampler 期望的 32 步 action horizon；无独立 `action_chunk` 键（官方文档 `openwam-alpha-finetuning.md:44` 同样强调）。改采样参数会破坏 sampler 预期，见 §6。

## 3. 代码地图

```
scripts/download_assets/
  download_openwam_checkpoints.py   # 【入口①】ALPHA 组 14 个 checkpoint 注册表 :99-118；Ckpt dataclass :77
configs/
  model/dual_system.yaml            # alpha 架构身份：:21 variant、:29 attention_mask_mode、:23/:25 action/state_dim
  dataloader/pretrain_data/
    mixture.yaml                    # 【F3 主目标】4 源 defaults 链 :9-14；weight_strategy=proportional :22
    agibotworld.yaml / robocoin.yaml / oxe_droid.yaml / interndata_a1.yaml  # 各源配置（路径全是占位）
openwam/train/utils/
  ckpt_model_loader.py              # 【F1/F4 核心】:60 _ARCH_IDENTITY_KEYS；:149 merge_ckpt_model_cfg；:216 build_architecture_from_ckpt_dir
openwam/dataloader/utils/
  unify_action.py                   # 80 维双射：:48/:118/:183/:215
  normalization.py                  # apply_normalization :82；rot6d 恒等钉住 :58
assets/openwam_usage_docs/
  openwam-alpha-finetuning.md       # 【权威文档】80 维契约 :46-89；微调命令 :107-140；部署契约 :142-150
README.md                           # alpha 宣称 :23；Quick Start 表 :287-290；采样耦合 :381-383
```

入口文件：F1/F6 从 `download_openwam_checkpoints.py` 进；F2/F3 从 `openwam-alpha-finetuning.md` 进；F4 从 `ckpt_model_loader.py` 进。

## 4. 任务书

### 任务 F1：alpha checkpoint 考古（主线）

**目标**：下载 OpenWAM-Alpha-Pretrain-Foundation-Model，解剖其 config.yaml 与 safetensors 键结构，产出"alpha 到底长什么样"解剖报告。
**前置**：约 25GB 磁盘；CPU 即可，无需 GPU。
**步骤**：
1. 下载（选 ALPHA → OpenWAM-Alpha-Pretrain-Foundation-Model；非交互模式）：
   ```bash
   cd $R
   python scripts/download_assets/download_openwam_checkpoints.py \
     --family alpha --name OpenWAM-Alpha-Pretrain-Foundation-Model --yes
   ```
2. 列出目录结构，对照"自包含"宣称（`download_openwam_checkpoints.py:7-8`）：config.yaml、`checkpoint_step_*.safetensors`、tokenizer/组件资产、`normalization_stats.npy` 是否齐全；注意该条目带 `finetune_only=True` 标记（`:103`，`:81` 注释：预训练 ckpt 供微调、不直接部署）。
3. 解剖 config.yaml：提取并记录 `architecture.framework` / `architecture.variant` / `attention_mask_mode` / `video_backbone.name` / `action_dim` / `state_dim` / `normalize_mode` / `unify_action` 的实际取值，与 §2④ 的 Quick Start 宣称逐项对账。
4. 解剖 safetensors 键结构（CPU 可读元数据，不加载权重）：
   ```bash
   python -c "
   from safetensors import safe_open
   import glob
   f = sorted(glob.glob('<ckpt_dir>/checkpoint_step_*.safetensors'))[-1]
   with safe_open(f, framework='pt') as h:
       keys = list(h.keys())
   print(len(keys)); [print(k) for k in keys if not k.startswith(('video_backbone.dit','action_backbone'))][:50]
   "
   ```
   按前缀聚类（video_backbone / action_backbone / 有无 understanding_expert），反推架构组成，与 config.yaml 互证。
**验收标准**：解剖报告含 ① 目录清单与自包含性核对结果；② config.yaml 关键键取值表（含与宣称的对账结论）；③ safetensors 键前缀聚类表与架构反推结论。
**产出**：alpha 解剖报告 1 份。
**难度（估计）**：低-中。**工作量（估计）**：1-2 天（主要是下载等待）。
**代码锚点**：`download_openwam_checkpoints.py:99-118`、`ckpt_model_loader.py:60`。

### 任务 F2：80 维 action/state 契约与归一化规则速查卡

**目标**：把 `openwam-alpha-finetuning.md` 全文与代码侧实现对照，产出一张契约速查卡（cheat sheet）。
**前置**：无（纯阅读）。
**步骤**：
1. 精读 `assets/openwam_usage_docs/openwam-alpha-finetuning.md`：样本字段规范（`:9-25`）、槽位布局（`:50-65`：左臂 0:9 + 左手 10:34、右臂 34:43 + 右手 44:68、保留 68:80）、gripper 约定（-1 闭合 / +1 张开，`:67-72`）、时间契约（num_frames=33 / video_stride=4 / 32 步 action chunk，`:38-44`）、min-max 归一化公式与 rot6d 恒等规则（`:76-87`）。
2. 代码侧逐项对照：`unify_action.py:48,118,183,215`（80 维双射）、`openwam/dataloader/utils/normalization.py:82 apply_normalization` 与 `:58 pin_rot6d_identity`（rot6d 钉住）、训练侧 mask 生成 `openwam/dataloader/bases/lerobot_v3_reader.py:912 _finalize_action` / `:965 _finalize_proprio`、部署侧反归一化 `openwam/dataloader/transforms/normalize.py:156 unnormalize`。
3. 标注文档与代码的差异/互补点（例如：归一化先于 scatter 时钉 native rot6d 维、后于 scatter 时钉两段 α 槽位——文档 `:87` 的规则在代码里如何体现）。
**验收标准**：速查卡一页纸覆盖：槽位表、mask 语义、归一化公式、rot6d 规则、gripper 极性、时间契约、部署侧 gather/反归一化义务（文档 `:150`）；每条标注文档行号 + 代码锚点双引用。
**产出**：契约速查卡 1 份（markdown）。
**难度（估计）**：低。**工作量（估计）**：1 天。
**代码锚点**：`unify_action.py:48-215`、`normalization.py:58,82`、`openwam-alpha-finetuning.md:46-89`。

### 任务 F3：alpha 预训练配方审计

**目标**：审计 `configs/dataloader/pretrain_data/mixture.yaml` 的实际配方，回答"518.5M 帧宣称有多少代码可验证"。与方向A的分工：A 管 reader 机制（MixtureDataset 采样策略、hook 体系），你管 alpha 实际配方的结论。
**前置**：无（纯阅读+核算）；F2 的契约理解有帮助。
**步骤**：
1. 读 `configs/dataloader/pretrain_data/mixture.yaml`：defaults 链 `:9-14` 只有 **4 个机器人源**（agibotworld / robocoin / oxe_droid / interndata_a1），`:22 weight_strategy: proportional`（按窗口数比例采样，weight 字段惰性）。
2. 对照 README 宣称 `README.md:23`（518.5M 帧含 egocentric human + robot）：human 部分（EgoDex/Ego4D）不在 mixture.yaml 里。
3. EgoDex 缺口证据链复核（证据已核验）：`openwam/dataloader/bases/lerobot_v3_reader.py:4-7` docstring 提及 EgoDex 为同等子类；`openwam/dataloader/utils/lerobotv3.py:3` docstring 提及 EgoDexDataset；但 `openwam/dataloader/registry.py:91-106` 的 13 个注册名中无 egodex/ego4d；全仓 grep 无 `class EgoDex*` 定义；`tests/dataloader/test_oxe_mixture_integration.py:68-69` 显式断言这 5 个名字 deprecated 未注册。
4. 核算 4 源各自的 `total_hours` 旋钮与体量（读各源 yaml 注释与 reader 的窗口统计逻辑），估算与 518.5M 帧的差距量级。
**验收标准**：配方审计报告含 ① mixture.yaml 实际 4 源与配比策略；② 518.5M 帧宣称 vs 代码可验证部分的逐项对账（哪些帧来自未发布的 EgoDex/human 源）；③ EgoDex 缺口证据链 5 条复核记录；④ 明确结论："alpha 预训练配方不可完整复现"的边界表述。
**产出**：配方审计报告 1 份。
**难度（估计）**：中。**工作量（估计）**：2-3 天。
**代码锚点**：`mixture.yaml:9-22`、`README.md:23`、`registry.py:91-106`、`test_oxe_mixture_integration.py:68-69`。

### 任务 F4：alpha 微调复现（finetune_ckpt_path 微调 LIBERO）

**目标**：按官方文档路径用 foundation checkpoint 热启动微调 LIBERO，产出可部署 checkpoint。
**前置**：F1 的 checkpoint 已下载；LIBERO 数据已下载（`python scripts/download_assets/download_benchmark_data.py`）；**4×80GB 级 GPU**（2×H100 在首个 Adam step OOM，`docker-validation.md:68`）。
**步骤**：
```bash
cd $R
NPROC_PER_NODE=8 bash scripts/train.sh \
  dataloader=libero \
  training.finetune_ckpt_path=<foundation_ckpt_dir_path> \
  training.num_epochs=5 \
  training.output_path=<output_dir_path>
```
注意：官方文档 `openwam-alpha-finetuning.md:117` 写的是 `project.output_dir`，但 trainer 实际读 `training.output_path`（`openwam/train/openwam_trainer.py:448`；`configs/train.yaml:51`），命令里必须用后者（见 §6）。卡数不足 8 时用 `NPROC_PER_NODE=4` 并视 OOM 叠加 `training.batch_size=8 training.offload_optimizer_device=cpu`。不需要传 `model=...`——架构身份由 ckpt 的 config.yaml 继承，`_ARCH_IDENTITY_KEYS`（`ckpt_model_loader.py:60`）会拒绝不一致的覆盖。
**验收标准**：微调 run 从 step 0 起步、写入新输出目录（不覆盖 foundation ckpt）；完成后目录含最终 `checkpoint_step_*.safetensors`、config.yaml、`normalization_stats.npy`，可直接 `bash scripts/deploy.sh <dir>` 部署。
**产出**：微调 run 记录 + 可部署 checkpoint（供 F5 或方向D 评测使用）。
**难度（估计）**：中。**工作量（估计）**：2-3 天（主要是训练时长）。
**代码锚点**：`ckpt_model_loader.py:149,216`、`configs/train.yaml:58`、`openwam-alpha-finetuning.md:107-140`。

### 任务 F5：alpha 部署与基准冒烟

**目标**：用方向C 的部署流程起 server，跑 LIBERO benchmark 冒烟，验证 alpha 能力下限。
**前置**：1 张 ≥24GB GPU；一个可部署 checkpoint（F4 产物，或直接下载 `OpenWAM-Alpha-Sim-LIBERO`）；LIBERO client 环境（`benchmarks/libero/README.md`）。**依赖声明**：部署流程（deploy.sh/server 语义）归 [方向C-Deployment-Infra部署管线.md](方向C-Deployment-Infra部署管线.md)，本任务只消费其结论；benchmark 评测口径归方向D，本任务只做"能不能跑起来、成功率是否非零"的冒烟。
**步骤**：
1. 起服务：`bash scripts/deploy.sh <ckpt_dir_path>`（部署资产要求见 `openwam-alpha-finetuning.md:142-150`：client 须把 80 维输出 gather 回 native 维度、应用保存的反归一化、保持槽位映射与 gripper 约定）。
2. 按 `benchmarks/libero/README.md` 起 LIBERO client，跑一个 task suite 的少量 episode（如 10 个）。
3. 记录：server 启动日志、显存占用、episode 成功率（非精确评测，只看量级）。
**验收标准**：server 正常加载 checkpoint；client 完成 ≥10 个 episode 且成功率非零（官方微调 ckpt 应有显著成功率；F4 自微调产物允许较低但须非零）；冒烟记录含显存与吞吐数字。
**产出**：冒烟报告 1 份。
**难度（估计）**：中。**工作量（估计）**：1-2 天。
**代码锚点**：`openwam-alpha-finetuning.md:142-150`、`README.md:386-391`（部署+LIBERO client 指引段）。

### 任务 F6：alpha 家族谱系整理

**目标**：把 `download_openwam_checkpoints.py` ALPHA 组（`:99-118`）的 14 个 checkpoint 整理成谱系表。
**前置**：无（纯阅读）；不必真的下载全部 14 个（各 24.8GB）。
**步骤**：
1. 从 `:103-116` 提取全部条目：1 个 foundation（`finetune_only=True`）+ 13 个微调变体（Real 系 5 个：Dexterous-Hand-Wuji / RoboDojo-ARX-X5 / RoboDojo-Piper / RoboDojo-Piper-X / Single-Arm-Franka；Sim 系 8 个：EBench / LIBERO / RoboCasa-GR1 / RoboCasa365 / RoboDojo / RoboTwin-Clean2Random / RoboTwin-Full / VLABench）。
2. 每个条目填表：名称 / 训练数据来源（从命名与 §3 各 reader 契约推断，标注推断依据）/ 本体与动作维度契约差异（对照方向A 的 80 维映射总表，如 Franka 单臂 vs Wuji 灵巧手）/ 适用场景（部署目标本体 or 仿真评测）。
3. 标注 `finetune_only` 语义（`:81`：foundation 供微调不供直接部署）。
**验收标准**：谱系表 14 行，每行含上述 4 列；契约差异列与 reader 代码锚点互链；全表可指导"我该下载哪个 checkpoint"的决策。
**产出**：alpha 家族谱系表 1 份。
**难度（估计）**：低。**工作量（估计）**：1 天。
**代码锚点**：`download_openwam_checkpoints.py:77-81,99-118`。

## 5. 建议推进顺序与里程碑

1. **W1**：F2（契约速查卡）+ F6（谱系表）→ F3（配方审计）。全部纯阅读，零环境依赖。里程碑：契约速查卡 + 谱系表 + 配方审计报告入库。
2. **W2**：F1（checkpoint 考古，下载可与 W1 并行发起）。里程碑：alpha 解剖报告——本方向核心交付。
3. **W3 起（有卡后）**：F4（微调，训练时长最长，拿到 4 卡资源即启动）→ F5（冒烟，可直接用官方 Sim-LIBERO ckpt 先行，不等 F4）。
里程碑汇总：W1 三份文档 → W2 解剖报告 → F4 可部署 ckpt → F5 能力下限验证。

## 6. 风险与坑

1. **预训练混合不可完整复现**：mixture.yaml 只有 4 个机器人源（`mixture.yaml:9-14`），EgoDex/human 源 reader 未随仓库发布（`registry.py:91-106` 无注册；`test_oxe_mixture_integration.py:68-69` 断言 deprecated）；各源 yaml 路径全是 `/path/to/pretrain_dataset/...` 占位，无下载脚本。规避：F3 定位为准入审计而非复现承诺；以官方 checkpoint 微调为主路径。
2. **num_frames=33 / video_stride=4 与 32 步 action horizon 耦合**：`README.md:381-383` 与 `openwam-alpha-finetuning.md:44` 均明示该组合产生 sampler 期望的 32 步 action chunk，且无独立 `action_chunk` 键。规避：微调/部署时绝不动这两个采样参数；`inference_horizon` 只控制每次生成消费多少步，不改 chunk 长度。
3. **project.output_dir 文档/代码不一致**：官方微调文档 `openwam-alpha-finetuning.md:117` 让用户设 `project.output_dir`，但 trainer 实际只读 `training.output_path`（`openwam/train/openwam_trainer.py:448`；`configs/train.yaml:51` vs `:66` 疑似遗留键）。规避：所有命令一律用 `training.output_path=...`；F4 报告中记录该勘误。
4. **每个 checkpoint ~24.8GB 的磁盘规划**：`download_openwam_checkpoints.py:103-116` 每条 24.8GB；F4 微调还会产生自己的输出 checkpoint（每个 safetensors 约 23GB，`save_steps` 默认 2000）。规避：磁盘预留 ≥150GB（单 ckpt + 微调产物）；做 F6 全谱系下载另需 ≥350GB；微调 run 靠 `training.keep_last_k_ckpts` 滚动清理。
5. **foundation checkpoint 不可直接部署**：`finetune_only=True` 标记（`download_openwam_checkpoints.py:81,103`）——foundation 缺少针对具体本体的归一化/部署语境，应微调后部署。规避：F5 冒烟若要直接部署，用 `OpenWAM-Alpha-Sim-LIBERO` 而非 foundation。
6. **架构身份键不可覆盖**：微调时若误传 `model=single_system` 或换 video_backbone 名，`_ARCH_IDENTITY_KEYS`（`ckpt_model_loader.py:60-63`）会直接拒绝启动。规避：微调命令不传 model 组参数，让 ckpt config 继承身份；报错时按 CLI 提示对齐。

## 7. 推荐阅读顺序

1. 本目录 [README.md](README.md) —— 全局定位与方向间分工。
2. `assets/openwam_usage_docs/openwam-alpha-finetuning.md` 全文 —— alpha 契约的权威来源，F2/F4 的蓝本。
3. `README.md:277-300`（Quick Start 表）+ `:370-391`（微调与部署段）—— alpha 默认架构链与官方推荐路径。
4. `scripts/download_assets/download_openwam_checkpoints.py:76-118` —— ALPHA 组注册表，F1/F6 直接依据。
5. `openwam/train/utils/ckpt_model_loader.py` —— finetune 重建语义（`:60`/`:149`/`:216`），理解"为什么微调不用传 model 参数"。
6. `openwam/dataloader/utils/unify_action.py` + `normalization.py:42-114` —— 80 维双射与归一化的代码侧实现（约 230 行，可全读）。
7. `configs/dataloader/pretrain_data/mixture.yaml` + `openwam/dataloader/registry.py:91-106` —— F3 配方审计的直接材料；reader 机制细节去 [方向A-Data-Infra数据管线.md](方向A-Data-Infra数据管线.md)。
8. [方向E-Study消融实验实现.md](方向E-Study消融实验实现.md) —— 理解 alpha 的每个配置选择（joint_self_attn/mutual/双系统）分别是哪组消融的结论。
