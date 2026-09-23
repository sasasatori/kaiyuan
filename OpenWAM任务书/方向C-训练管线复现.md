# 方向 C：训练管线复现

> 一句话定位：把 OpenWAM 的「torchrun + Accelerate + DeepSpeed ZeRO-2」双流 flow-matching 训练管线从 20 步 debug 冒烟一路复现到 LIBERO 微调，并摸清资源-旋钮关系。读者：第一次接触该模块的同学。
> 前置：先读本目录 README.md 的总览，了解本方向在全局中的位置。环境搭建依赖 [方向A-环境搭建与验证矩阵.md](方向A-环境搭建与验证矩阵.md)；训练产物的部署见 [方向B-部署与推理复现.md](方向B-部署与推理复现.md)。

仓库根 `$R = /fact_home/yiyangyuan/workspace/projects/kaiyuan/OpenWAM`，下文所有路径均为仓库相对路径。行号以 main @ 90e94ae 为准；行号漂移时以符号名为准（下文引用格式 `path:line 符号`，符号均已逐一核对）。

## 1. 这个方向做什么、为什么值得做

OpenWAM 是一个 World-Action Model（WAM）：同一个模型既做视频世界预测（video DiT 流），又做机器人动作生成（action DiT 流），两路各吃一个 flow-matching 损失，用 `lambda_video` / `lambda_action` 加权相加。训练管线的作用就是把这条「双流联合训练」跑起来：从 Hydra 配置组装 → torchrun 多进程启动 → Accelerate 包一层 DeepSpeed ZeRO-2 → 数据加载 → 主循环（前向、加噪、算 loss、反传、存 checkpoint）。

为什么值得做：这是整个调研的「生产资料」环节。方向 B 的部署、方向 E 的评测、方向 F 的架构消融，全部依赖本方向产出的 checkpoint（尤其是 C3 的 OpenWAM-α 微调结果）。同时，这套代码的 checkpoint 语义比一般开源项目讲究得多：finetune（热启动新 run）与 resume（续训同一 run）互斥、resume 有 grad_accum 对齐和归一化统计强校验，搞清楚它们是理解「大模型训练工程」的绝佳样本。

你能学到什么：Hydra 配置组合与 CLI override 的实战用法；torchrun/Accelerate/DeepSpeed 三层是怎么叠起来的；flow-matching 损失长什么样；ZeRO 分片下 checkpoint 为什么是「双轨制」；以及一个 6B 级模型在真实 80GB 卡上的显存/磁盘账本。

## 2. 最小必要背景

**Hydra 配置组合**：训练配置不是单个 yaml，而是「defaults 链 + CLI override」拼出来的。入口 `configs/train.yaml:1-4` 声明默认组：

```yaml
defaults:
  - model: dual_system            # single_system | tri_system
  - dataloader: robotwin          # 可换 libero 等
  - _self_
```

命令行 `key=value` 覆盖任意叶子，如 `training.debug=true`、`dataloader=libero`、`model.architecture.variant=joint_self_attn`。全部训练旋钮集中在 `configs/train.yaml` 的 `training:` 段（lr、batch、save_steps 等），模型结构旋钮在 `configs/model/dual_system.yaml`（如 `:29 attention_mask_mode: mutual`）。

**torchrun + Accelerate + DeepSpeed ZeRO-2**：`scripts/train.sh` 只是 torchrun 启动器（`:36-41` 解析 `NPROC_PER_NODE`/`NNODES`/`NODE_RANK`/`MASTER_ADDR`），真正干活的是 `scripts/train.py`：`:61 _build_accelerator()` 用 cfg.training 构造 `accelerate.DeepSpeedPlugin`（`:77`，吃 `zero_stage`、`gradient_accumulation_steps`、`max_grad_norm`、`offload_optimizer_device`），torchrun 已初始化分布式，所以不需要 `accelerate launch`。

**OpenWAMTrainer 主循环**：`openwam/train/openwam_trainer.py:58 class OpenWAMTrainer` 只负责编排：构建架构 → 冻结模块 → AdamW（可分模块 LR）→ `train()` 循环。debug 模式在 `:190-193`：`max_steps=20`、`save_steps=10`、恒定 LR。输出目录在 `:448 setup_output_dir` 用 `training.output_path`（注意：不是 `project.output_dir`，见 §6 风险）。

**flow-matching 双流损失**：`openwam/model/architectures/base.py:1044 compute_loss`。流程：`:868 prepare_inputs`（`@torch.no_grad` 编码视频 latent 与文本）→ 采 video/action 各自的 timestep → 线性插值加噪 → 一次联合前向 → `:1198 _compute_video_loss`（per-frame 加权 MSE，跳过条件帧）+ `:1254 _compute_action_loss`（per-element MSE，吃 (B,T)/(B,T,D) mask）→ `total = lambda_video*loss_video + lambda_action*loss_action`（权重在 `configs/train.yaml:43-44`）。

**checkpoint 双轨制**：`openwam/train/utils/checkpointing.py` 里有两条互不相干的产物线——
- 权重轨（部署用）：`:195 save_weights`，rank0 写 `checkpoint_step_N.safetensors`（约 23 GiB/个）；
- 状态轨（续训用）：`:220 save_full_state`，全 rank 写 `accel_state_step_N/`（优化器/scheduler/RNG，>100 GiB/个），以 `trainer_state.json` 为原子完成标记。
  配套：`:340 manage_checkpoints` 按 `keep_last_k_ckpts` 滚动清理；`:368 finalize_keep_weights_only` 训练正常结束后删掉所有 resume state。

**finetune / resume 重建**：`openwam/train/utils/ckpt_model_loader.py`。`:216 build_architecture_from_ckpt_dir` 从 checkpoint 目录自包含重建模型（不需要原始 `model_path`）；`:149 merge_ckpt_model_cfg` 以 ckpt 的 model 配置为基底、本 run 的 `cfg.model` 覆盖之；`:60 _ARCH_IDENTITY_KEYS`（`video_backbone.name` 与 `architecture.framework`）不一致时直接拒绝启动。二者语义对比见任务 C5。

## 3. 代码地图

```
scripts/
  train.sh                       # 【入口】torchrun 启动器；NPROC_PER_NODE/NNODES 解析 :36-41
  train.py                       # Hydra 入口 main()；:61 _build_accelerator；_train_openwam
configs/
  train.yaml                     # 训练旋钮总表；:1-4 defaults 链；:51 output_path；:57-59 finetune/resume
  model/dual_system.yaml         # :21 variant=joint_self_attn；:29 attention_mask_mode=mutual
  dataloader/libero.yaml         # LIBERO 数据配置（任务 C1 起使用）
openwam/train/
  openwam_trainer.py             # 【核心】:58 OpenWAMTrainer；:190-193 debug；:448 setup_output_dir
  utils/checkpointing.py         # 双轨 checkpoint：:195/:220/:320/:340/:368/:136
  utils/ckpt_model_loader.py     # finetune/resume 重建：:60/:149/:216
  utils/training_utils.py        # :88 init_wandb；:150 write_debug_loss_row（写 debug_loss_history.csv）
  utils/optimizer_groups.py      # build_trainable_parameters：video/action/other 三分桶
  utils/seeding.py               # seed_everything / per_step_seed（rank 隔离窗口）
openwam/model/architectures/
  base.py                        # :868 prepare_inputs；:1044 compute_loss；:1198/:1254 两路 MSE
  dual_system/                   # joint_self_attn 等三个变体的实现（方向 F 重点，本方向只消费）
assets/openwam_usage_docs/
  openwam-alpha-finetuning.md    # 官方微调文档（C3 的蓝本；:117 有误，见 C5）
  docker-validation.md           # 官方硬件验证记录（:68 2×H100 OOM；:28 磁盘实测）
```

入口文件：`scripts/train.sh`（ shell 层）→ `scripts/train.py`（进程内入口）→ `openwam/train/openwam_trainer.py`（循环编排）。

## 4. 任务书

### 任务 C1：20 步 debug 冒烟（LIBERO + Wan2.2-5B + dual_system/joint_self_attn + mutual）

**目标**：用 `training.debug=true` 跑通 20 步最小训练，验证环境、数据、模型权重、损失链路全部就位。
**前置**：方向 A 环境就绪；已下载 LIBERO 数据与 Wan2.2-TI2V-5B 权重（下载脚本会回写 yaml 路径）。
**步骤**：

```bash
cd $R
NPROC_PER_NODE=4 bash scripts/train.sh \
  dataloader=libero \
  model=dual_system \
  model/video_backbone=wan22_ti2v_5b \
  model.architecture.variant=joint_self_attn \
  model.architecture.attention_mask_mode=mutual \
  training.debug=true
```

`NPROC_PER_NODE` 不设时自动取全部可见 GPU（`scripts/train.sh:36`）。多机时每个节点分别执行（rank 0 节点写 `MASTER_ADDR` 为自己的 IP）：

```bash
# 节点 0
NNODES=2 NODE_RANK=0 MASTER_ADDR=<node0_ip> NPROC_PER_NODE=4 bash scripts/train.sh <同上参数>
# 节点 1
NNODES=2 NODE_RANK=1 MASTER_ADDR=<node0_ip> NPROC_PER_NODE=4 bash scripts/train.sh <同上参数>
```

**验收标准**：进程退出码 0；输出目录（`outputs/openwam_checkpoints/<时间戳>_debug/`）下出现 `debug_loss_history.csv`（写入逻辑在 `openwam/train/utils/training_utils.py:165`）与 step 10、20 两个 `checkpoint_step_*.safetensors`；20 行 loss 均为有限值。
**产出**：可复现的冒烟日志 + `debug_loss_history.csv`。
**难度（估计）**：低-中。**工作量（估计）**：0.5 天（含权重下载等待）。
**代码锚点**：`openwam/train/openwam_trainer.py:190-193`（debug 语义）、`scripts/train.sh:36-41`、`configs/model/dual_system.yaml:21,29`。

### 任务 C2：正式小规模训练（debug=false，验证 checkpoint 保存/清理与 wandb）

**目标**：关掉 debug 跑一个完整 epoch 级 run，亲眼验证 `keep_last_k_ckpts` 滚动清理与 wandb 日志。
**前置**：C1 通过。
**步骤**：为缩短周期，把 save 粒度调小、步数限定：

```bash
cd $R
NPROC_PER_NODE=4 bash scripts/train.sh \
  dataloader=libero \
  model=dual_system model/video_backbone=wan22_ti2v_5b \
  training.debug=false \
  training.max_steps=60 training.save_steps=20 \
  training.keep_last_k_ckpts=2 \
  project.wandb.run_name=c2_small_run
```

**验收标准**：loss 曲线正常下降；step 20/40/60 均产出 `checkpoint_step_*.safetensors`，且 step 60 落盘后 step 20 的 checkpoint 被 `manage_checkpoints` 清掉（`openwam/train/utils/checkpointing.py:340`）；目录结构与 §3 双轨描述一致（默认 `save_full_states_for_resume=false`，无 `accel_state_step_*`）。
**产出**：run 记录（wandb 截图或 offline run 目录 + 目录树快照）。
**难度（估计）**：中。**工作量（估计）**：1 天。
**代码锚点**：`openwam/train/openwam_trainer.py:362-367`（保存+清理调用点）、`openwam/train/utils/training_utils.py:88 init_wandb`。

### 任务 C3：OpenWAM-α 微调复现（finetune_ckpt_path 微调 LIBERO）

**目标**：按官方文档 `assets/openwam_usage_docs/openwam-alpha-finetuning.md` 的路径，用 foundation checkpoint 热启动微调 LIBERO，产出可部署 checkpoint（交付方向 B/E 使用）。
**前置**：下载 OpenWAM-Alpha foundation checkpoint 目录；C1 通过。
**步骤**（官方文档 §3 的命令模板，**注意把文档里的 `project.output_dir` 换成 `training.output_path`**，见 C5 与 §6 风险）：

```bash
cd $R
NPROC_PER_NODE=8 bash scripts/train.sh \
  dataloader=libero \
  training.finetune_ckpt_path=<foundation_ckpt_dir_path> \
  training.num_epochs=5 \
  training.output_path=<output_dir_path>
```

卡数不足 8 时用 `NPROC_PER_NODE=4` 并视 OOM 情况叠加 `training.batch_size=8 training.offload_optimizer_device=cpu`。
**验收标准**：微调 run 从 step 0 起步、写入**新**输出目录（不覆盖 foundation ckpt）；完成 `num_epochs` 后目录下含最终 `checkpoint_step_*.safetensors`、`config.yaml`、`normalization_stats.npy`，可直接交给方向 B 部署。
**产出**：微调 run 记录 + 可部署 checkpoint。
**难度（估计）**：中。**工作量（估计）**：2-3 天（主要是训练时长）。
**代码锚点**：`openwam/train/utils/ckpt_model_loader.py:216 build_architecture_from_ckpt_dir`、`configs/train.yaml:58`、`assets/openwam_usage_docs/openwam-alpha-finetuning.md:105-140`。

### 任务 C4：checkpoint 恢复语义验证（resume 中断续训、grad_accum 对齐、normalization stats 校验）

**目标**：做一组中断-续训对照实验，验证 `resume_ckpt_path` 的三层语义：续上 optimizer/scheduler/RNG、skip 边界对齐到 grad_accum、归一化统计不一致即拒绝。
**前置**：C2 完成。
**步骤**：
1. 启动带状态保存的 run，并在 step 40 的 `accel_state_step_40/trainer_state.json` 出现后手动 `kill`（模拟中断）：

```bash
NPROC_PER_NODE=4 bash scripts/train.sh \
  dataloader=libero training.max_steps=100 training.save_steps=20 \
  training.save_full_states_for_resume=true training.gradient_accumulation_steps=4 \
  training.output_path=outputs/c4_resume_test
```

2. 续训（注意 `finetune_ckpt_path` 必须保持 null，二者互斥）：

```bash
NPROC_PER_NODE=4 bash scripts/train.sh \
  dataloader=libero training.resume_ckpt_path=outputs/c4_resume_test/<run_dir>
```

3. 对照实验：(a) 把 `gradient_accumulation_steps` 改成 2 再 resume，观察 `compute_resume_position`（`openwam/train/utils/checkpointing.py:320`）如何 floor 对齐；(b) 人为改坏 run 目录里的 `normalization_stats.npy`，验证 `verify_resume_normalization_stats`（`checkpointing.py:136`）是否 fail fast。
**验收标准**：续训从 step 40 之后继续（日志 `[resume] reusing run dir`；trainer 侧逻辑见 `openwam/train/openwam_trainer.py:458-465`）；loss 与「不中断一口气跑完」的对照曲线基本重合；(a)(b) 两个边界行为与代码语义一致并形成文字结论。
**产出**：中断-续训对照实验报告。
**难度（估计）**：中。**工作量（估计）**：1.5-2 天。
**代码锚点**：`openwam/train/utils/checkpointing.py:244 load_full_state`、`:293 find_latest_accel_state`、`:320 compute_resume_position`、`:136 verify_resume_normalization_stats`。

### 任务 C5：finetune vs resume 差异审计（含 project.output_dir 文档不一致修正）

**目标**：产出一份 finetune/resume 语义文档，讲清 `merge_ckpt_model_cfg` 的 override 规则与 `_ARCH_IDENTITY_KEYS` 拒绝规则，并坐实 `project.output_dir` 的文档/代码不一致。
**前置**：C3、C4 完成（有现成 run 可引用）；本任务以读代码 + 最小实验为主。
**步骤**：
1. 通读 `openwam/train/utils/ckpt_model_loader.py`：`:149 merge_ckpt_model_cfg`（ckpt model 配置为基底、live cfg 覆盖）、`:60 _ARCH_IDENTITY_KEYS`（`video_backbone.name`、`architecture.framework` 不一致即拒绝）、`warn_live_model_cfg_ignored`（resume 时 live 的 model 配置被忽略的告警）。
2. 最小实验：拿 C3 的 foundation ckpt，故意用 `model=single_system` 启动 finetune，观察拒绝报错与 CLI 提示。
3. 文档不一致验证：`assets/openwam_usage_docs/openwam-alpha-finetuning.md:117` 让用户设 `project.output_dir=<output_dir_path>`，但 `openwam/train/openwam_trainer.py:448 setup_output_dir` 实际读的是 `training.output_path`（`configs/train.yaml:51`），`project.output_dir`（`configs/train.yaml:66`）疑似遗留键。实验：只设 `project.output_dir` 启动一次冒烟，确认输出仍落在 `training.output_path`。
**验收标准**：语义文档覆盖：finetune（step 从 0、新目录、live cfg 覆盖 ckpt cfg、身份键除外）vs resume（复用原目录、恢复 optimizer/scheduler/RNG、live model cfg 被忽略）；`project.output_dir` 结论（代码行为 + 文档修正建议）写入语义文档。
**产出**：finetune/resume 语义文档（含 `project.output_dir` 勘误）。
**难度（估计）**：低-中。**工作量（估计）**：1 天。
**代码锚点**：`openwam/train/utils/ckpt_model_loader.py:60,149,216`、`configs/train.yaml:57-59,66`、`openwam/train/openwam_trainer.py:448`。

### 任务 C6：资源旋钮消融（zero_stage / gradient_checkpointing / bf16 vs fp16 / offload_optimizer_device）

**目标**：在本组卡型上实测吞吐（it/s）与峰值显存（`torch.cuda.max_memory_allocated` 或 `nvidia-smi` 采样），产出旋钮-资源对照矩阵。
**前置**：C1 通过；≥4×80GB（2 卡会在首个 Adam step OOM，见 §6 风险）。
**步骤**：固定 `dataloader=libero model/video_backbone=wan22_ti2v_5b training.debug=true`（20 步足够测稳态吞吐），逐格扫描，示例：

```bash
# 基线（train.yaml 默认：ZeRO-2 + grad-ckpt + bf16 + offload none）
NPROC_PER_NODE=4 bash scripts/train.sh dataloader=libero training.debug=true
# 变体（每次只改一个旋钮）
... training.zero_stage=1
... training.use_gradient_checkpointing=false
... training.mixed_precision=fp16
... training.offload_optimizer_device=cpu
```

每格记录：是否 OOM、稳态 it/s、单卡峰值显存。OOM 格标注 OOM 发生的阶段（前向/反向/首个 Adam step）。
**验收标准**：对照表覆盖 ≥8 格（4 旋钮 × 2 取值），每格含吞吐/显存/是否 OOM 三项；表头注明本组卡型与驱动版本；给出「本组硬件下的推荐配置」结论。
**产出**：旋钮-资源对照表（含实测值）。
**难度（估计）**：中。**工作量（估计）**：1.5-2 天（每格一次 debug run）。
**代码锚点**：`configs/train.yaml:31-40`（四个旋钮的默认值）、`scripts/train.py:61-90 _build_accelerator`（旋钮注入 DeepSpeedPlugin 的位置）。

## 5. 建议推进顺序与里程碑

1. **C1**（冒烟）→ 2. **C2**（正式小规模）→ 3. **C3**（微调，训练时长最长，尽早启动）→ 4. **C4**（恢复语义，依赖 C2 的目录认知）→ 5. **C5**（审计，可利用 C3/C4 等待训练的间隙做）→ 6. **C6**（消融，独立，可与 C4/C5 并行排队用卡）。
- 里程碑 M1：C1+C2 通过，冒烟与保存/清理语义确认（对应全局 M2 阶段）。
- 里程碑 M2：C3 完成，微调 checkpoint 交付方向 B/E（全局 M2→M3 的关键交付）。
- 里程碑 M3：C4-C6 收尾，恢复语义文档 + 资源旋钮表入库（对应全局 M3）。

## 6. 风险与坑

1. **2×H100 在首个 Adam step 即 OOM**：官方验证记录 `assets/openwam_usage_docs/docker-validation.md:68` 与 `docker.md:196-199`：两张 H100 80GB 在第一次 Adam 更新时 OOM，四张才通过（ZeRO-2、batch=1、grad-ckpt 开启）。规避：4×80GB 起步；仍不够时叠加 `training.offload_optimizer_device=cpu`、`training.initialize_model_on_cpu=true`。注意 debug 模式不省优化器状态内存（`docker.md:359`）。
2. **save_steps=2000 的磁盘隐性风险**：每个 `checkpoint_step_N.safetensors` 约 23.1 GiB，若开 `save_full_states_for_resume=true` 每个 `accel_state_step_N/` 还要 ~102.3 GiB（实测见 `docker-validation.md:25-29`）。规避：磁盘预留 >150GB；常规 run 保持 `save_full_states_for_resume=false`（默认）；只有明确要 resume 的 run 才开；及时靠 `keep_last_k_ckpts`（`configs/train.yaml:55`）滚动清理。
3. **project.output_dir 文档/代码不一致**：`configs/train.yaml:66` 定义了 `project.output_dir`，官方微调文档 `openwam-alpha-finetuning.md:117` 也让用户设它，但 trainer 实际只用 `training.output_path`（`openwam/train/openwam_trainer.py:448`）。规避：所有命令里输出目录一律用 `training.output_path=...`；结论与修正建议写入 C5 文档。
4. **wandb 默认 offline**：官方镜像在 `docker/Dockerfile:97` 注入 `WANDB_MODE=offline`。规避：离线时日志在 `wandb/offline-run-*` 目录，事后 `wandb sync` 上传；需要在线看板则在启动前 `export WANDB_MODE=online` 并配置 API key。
5. **resume state 的原子标记**：`accel_state_step_N/` 只有在 `trainer_state.json` 出现后才算完整（`openwam/train/utils/checkpointing.py:293 find_latest_accel_state`；`docker.md:211-212`）。规避：做 C4 中断实验时，等 marker 出现后再 kill；中途 kill 留下的残缺 state 会被自动跳过，不会报错但会让人误以为 resume 位置不对。
6. **finetune 与 resume 互斥**：`configs/train.yaml:57-59` 明确二者互斥；resume 时若还挂着 `finetune_ckpt_path` 会语义冲突。规避：resume 命令里显式 `training.finetune_ckpt_path=null`（`docker.md:209` 同样强调）。
7. **NCCL 在 H100 上的已知坑**：镜像默认 `NCCL_RAS_ENABLE=0`（RAS + NSS 导致进程退出时 SIGSEGV）；历史上四卡 NCCL 初始化 hang 需关 NVLS（`docker.md:362-371`）。规避：用官方镜像默认值；裸机复现若遇到 hang/关机崩溃，先对齐这两个环境变量。

## 7. 推荐阅读顺序

1. 本目录 README.md —— 知道方向 C 在全局的位置与上下游依赖。
2. `configs/train.yaml` 全文 —— 所有训练旋钮的总表，之后每个任务都在改它。
3. `scripts/train.sh` + `scripts/train.py` —— 知道命令是怎么变成多进程 DeepSpeed 训练的。
4. `openwam/train/openwam_trainer.py`（重点 `:58-200` 与 `:440-490`）—— 主循环与输出目录语义。
5. `openwam/model/architectures/base.py:1044 compute_loss` 及其调用的 `:1198`/`:1254` —— 知道 loss 怎么来的（debug 时看得懂 csv 列）。
6. `openwam/train/utils/checkpointing.py` —— 双轨 checkpoint 全貌，C2/C4 的直接依据。
7. `openwam/train/utils/ckpt_model_loader.py` —— finetune/resume 重建语义，C3/C5 的直接依据。
8. `assets/openwam_usage_docs/openwam-alpha-finetuning.md` + `docker-validation.md` —— 官方流程与硬件实测数字，带着 §6 的坑清单对照着读。
