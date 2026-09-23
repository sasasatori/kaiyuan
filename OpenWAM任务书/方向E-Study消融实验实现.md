# 方向 E：Study 消融实验实现

> 一句话定位：拆解 OpenWAM-Study 的受控消融**在代码里是怎么实现的**——Hydra 组合即实验矩阵、seeding 保证确定性、tests/ 固化不变量，并亲手复现一组官方对照实验。读者：第一次接触该模块的同学。
> 前置：先读 [README.md](README.md) 的总览。**本方向第一阶段（W1–W2）纯阅读+文档化，不需要任何 GPU/环境**。

仓库根记作 `OpenWAM/`，下文路径均为仓库相对路径。行号以 main @ 90e94ae 为准。注意分工边界：架构机制本身（四种耦合范式、mask 语义）归 [方向B-Model-Infra模型管线.md](方向B-Model-Infra模型管线.md)；本方向回答的是"控制变量"的工程实现——怎么保证两次 run 只差一个旋钮、怎么让结论可信。

## 0. 你的环境与资源路径

- **第一阶段（W1–W2，任务 E1/E4/E5 的阅读部分）**：纯阅读，零依赖。只需要一份仓库 clone。
- **任务 E2（确定性验证）**：仅需 CPU。`pip install -e '.[dev]'`（CPU torch 即可），用 mock backbone 跑 seeding 单元测试与小脚本，不下载任何权重。
- **任务 E4（测试安全网）**：仅需 CPU。`make test` 全绿即验收，CI 等价命令见 README §5。
- **任务 E3（官方消融复现，唯一吃资源任务）**：三样外来物——
  1. **方向 C 的部署能力**：`bash scripts/deploy.sh <ckpt_dir>` 起 WebSocket server（见 [方向C-Deployment-Infra部署管线.md](方向C-Deployment-Infra部署管线.md)）；
  2. **方向 D 的 LIBERO 评测 harness**：manifest + labtasker 客户端（见 [方向D-Evaluation-Infra评测管线.md](方向D-Evaluation-Infra评测管线.md)）；
  3. **2 个 study checkpoint**：各约 24.8 GB（tri 29.3 GB、wan21_i2v_14b 49.7 GB，避开这两个），用 `python scripts/download_assets/download_openwam_checkpoints.py` 下载到 `assets/openwam_ckpt/openwam_study/<group>/`。
  另需 1 张 ≥24GB GPU 跑 server。**不依赖全组统一训练环境，也不需要任何训练数据**——官方 checkpoint 已训好，本方向只复现"评测对比"。
- **你明确不依赖别人的**：方向 A 的数据管线（不碰训练数据）、方向 B 的扩展实验、方向 F 的 alpha 微调。重训类对照（小规模重训复现）属可选加分项，需要 4×80GB 时单独申请（README R3）。

## 1. 这个方向做什么、为什么值得做

OpenWAM-Study 是论文的消融体系：官方在 HuggingFace 发布了 **34 个 study checkpoint**（外加 14 个 alpha checkpoint），按 5 个维度分组——架构、预训练配方、视觉编码器、视频骨干、注意力掩码。本方向要回答三个问题：

1. **实验矩阵在哪**：这些 checkpoint 的命名与分组，如何一一对应到 `configs/` 里的 Hydra 旋钮？（E1）
2. **对照为什么可信**：同一 trainer 只切一个 config 键时，是什么机制保证其余一切逐位相同？（E2/E4）
3. **怎么亲手复现一组**：用官方 checkpoint + 评测 harness 做省算力路线的对比实验，并建立组内实验记录规范。（E3/E5）

为什么值得做：消融是"结论"的生产方式，而大多数开源项目只发布结论不发布方法论。OpenWAM 难得地把受控旋钮（Hydra 组）、确定性基建（seeding）、不变量测试（tests/）全部开出来了——把这套"怎么做可信对照实验"的工程方法吃透，比记住任何一个消融数字都更有迁移价值。同时，本方向的产出（矩阵对照表、实验记录模板）是方向 F（alpha 是 study 结论的产物）与全组实验复用的公共资产。

## 2. 最小必要背景

**Hydra 组合即实验矩阵** — 训练配置 = defaults 链 + CLI override。入口 `configs/train.yaml:1-4`：

```yaml
defaults:
  - model: dual_system            # single_system | tri_system
  - dataloader: robotwin          # libero 等 7 选 1
  - _self_
```

每个 Study 维度就是一个 Hydra 组或叶子键：`model=single_system|dual_system|tri_system`（framework）、`model.architecture.variant=joint_self_attn|joint_cross_attn|idm`（`configs/model/dual_system.yaml:21`）、`model.architecture.attention_mask_mode`（`dual_system.yaml:29`）、`model/video_backbone=wan22_ti2v_5b|wan21_vace_1_3b|...`、`model.video_backbone.encoder.*`（视觉编码器）、`dataloader=robotwin|libero|...`。一次 CLI override 链 = 矩阵里的一个格子。

**registry 的 (framework, variant) 索引** — yaml 二元组 → 6 个注册名之一：`openwam/model/architectures/registry.py:96 resolve_architecture_config`，`:110-111` 查 `_FRAMEWORK_VARIANT_INDEX`，`:130` 得 `registry_name`；`variant=None` 且 framework 只有一个候选时自动补默认（`:112-116`）。机制细节归方向 B，这里只需知道：消融的"架构旋钮"最终经这一个函数收口。

**训练确定性（seeding 体系）** — `openwam/train/utils/seeding.py`：`:52 RANK_OFFSET = 1_000_000`（每个 rank 一个 1M 宽的种子窗口，per-step 计数不会溢出到隔壁 rank）；`:55 seed_process`（Python/NumPy/torch 三件套，不碰 cudnn）；`:73 seed_everything`（额外锁 cudnn deterministic）；`:133 per_step_seed`（`seed + RANK_OFFSET*rank + step`，主循环每步 `torch.manual_seed`，调用点 `openwam_trainer.py:316-320`）；`:176 wire_sampler_seed`（把 shuffle 顺序钉到 run_seed，调用点 `openwam_trainer.py:249`）。种子源头是 `project.seed: 42`（`configs/train.yaml:67`），由 `scripts/train.py:93 _inject_project_seed` 下发给 dataloader、`:131 seed_everything` 在数据集构造前播种。**设了 seed = 可复现 run；seed=null = 生产随机 run**。

**debug 模式（20 步恒定 LR）** — `openwam_trainer.py:190-193`：`training.debug=true` → `max_steps=20`、step 10 存一次、恒定 LR。这是所有消融冒烟的统一入口。

**checkpoint 自描述（config.yaml）** — `openwam/train/utils/checkpointing.py:22 save_config` 把完整 Hydra DictConfig 随训练写出（调用点 `openwam_trainer.py:484`，只写一次），部署/重建时以它为准。这是"实验记录"的权威载体——不靠人记，靠目录自述。

## 3. 代码地图

入口文件：先读 `scripts/download_assets/download_openwam_checkpoints.py`（看官方矩阵长什么样），再进 `configs/` 对账。

```
configs/
├── train.yaml                     # 【入口】训练旋钮总表；:1-4 defaults 链；:8 debug；:55 keep_last_k_ckpts；:67 project.seed
├── model/
│   ├── single_system.yaml         # framework=single_system；variant vanilla|moe
│   ├── dual_system.yaml           # :21 variant 三选一；:29 attention_mask_mode；:38-40 变体专属旋钮
│   ├── tri_system.yaml            # framework=tri_system（仅 joint_self_attn）
│   ├── video_backbone/            # wan22_ti2v_5b / wan21_vace_1_3b / cosmos25_2b ...（骨干维度）
│   └── action_backbone/           # shared_action_backbone / separate_action_dit
└── dataloader/                    # robotwin/libero/... 7 个 benchmark 组（数据维度）

scripts/download_assets/
└── download_openwam_checkpoints.py  # 【官方矩阵】ALPHA :99-118（14 个）；STUDY_GROUPS :120-185（5 组 34 个）

openwam/model/architectures/
└── registry.py                    # :96 resolve_architecture_config；(framework,variant) 索引 :110-130

openwam/train/utils/
├── seeding.py                     # 【确定性核心】RANK_OFFSET :52；seed_process :55；per_step_seed :133；wire_sampler_seed :176
├── checkpointing.py               # save_config :22（config.yaml 自描述）；manage_checkpoints :340；finalize_keep_weights_only :368
├── training_utils.py              # init_wandb :88；write_debug_loss_row :150（debug_loss_history.csv 落盘 :165）
└── ckpt_model_loader.py           # :216 build_architecture_from_ckpt_dir（从 ckpt 目录自包含重建）

openwam/train/openwam_trainer.py   # debug 语义 :190-193；seeding 调用点 :83/:249/:316-320；save_config 调用 :484

scripts/train.py                   # _inject_project_seed :93；seed_everything 调用 :120-131

tests/                             # 【实验安全网】
├── test_split_attn_equivalence.py # split-attn 与整 block forward 零 atol 等价（docstring :1-8）
├── test_train_deploy_consistency.py  # :50 idm 训练↔部署一致；:105 抗扰动不变量；GPU 门控 :429
├── test_mask_modes.py             # 四模式断言 :56/:69/:75/:82/:90
├── test_architecture_variants.py  # _FRAMEWORKS×_BACKBONES :767-768；9 格 compose 矩阵 :781-783
├── test_seeding.py                # seeding 单元测试（E2 直接素材）
└── test_dataloader_seed.py        # dataloader 种子下发测试
```

## 4. 任务书

### 任务 E-1：Study 实验矩阵考古（官方消融矩阵 ↔ config 键对照表）

**目标**：枚举全部受控旋钮如何经 Hydra defaults 组合成实验矩阵；对账 `download_openwam_checkpoints.py:120-185 STUDY_GROUPS`（34 个 checkpoint 分 5 组），产出"官方消融矩阵 ↔ config 键"对照表。
**前置**：无 GPU、无下载；纯阅读。
**步骤**：
1. 通读 `configs/train.yaml:1-4` 与三份 `configs/model/*.yaml`，列出每个 Study 维度对应的 Hydra 组/键：`architecture.framework`、`architecture.variant`、`attention_mask_mode`、`video_attention_mask_mode`、`video_backbone`、`encoder`、`detach_bridge`、`dataloader`。
2. 逐组对账 STUDY_GROUPS 的命名规则如何编码旋钮：
   - `OpenWAM_Study_Architecture`（:121-133，7 个）：`{data}_{framework}_{variant}`，`_detach` 后缀 = `detach_bridge=true`；
   - `OpenWAM_Study_Pretrain`（:134-151，12 个）：`pretrain_*`/`sft_*` 前缀 = 训练阶段，`_mutual`/`_action_sees_video` 后缀 = mask 旋钮，`robot_only`/`two_stage`/`from_scratch` = 配方；
   - `OpenWAM_Study_Visual_Encoder`（:152-163，6 个）：尾部 `_dinov3|_vjepa21|_flux2|_wan22_vae`（`_svae` = 叠加 SVAE reducer）→ `model.video_backbone.encoder.*`；
   - `OpenWAM_Study_Video_Backbone`（:164-174，5 个）：尾部 `_cosmos25|_cosmos3|_wan21_i2v_14b|_wan21_vace_1_3b`，无后缀基座 = `wan22_ti2v_5b`；
   - `OpenWAM_Study_Attention_Mask`（:175-184，4 个）：尾部 `_isolated|_mutual|_video_sees_action`，无后缀基座 = `action_sees_video`（对照 `dual_system.yaml:28-29` 注释）。
3. 注意一个陷阱：`robotwin_dual_system_joint_self_attention` 同名 checkpoint 出现在 3 个组里（architecture/video_backbone/attention_mask），是同一基座在三组中当"对照原点"——对照表里标出各组的 baseline。
4. 对每个格子写出"复现该格的 CLI override 链"，如 attention_mask 组的 mutual 格：`model=dual_system model.architecture.variant=joint_self_attn model.architecture.attention_mask_mode=mutual dataloader=robotwin`。
**验收标准**：对照表覆盖全部 34 个 checkpoint，每行 = {组、checkpoint 名、编码的旋钮及取值、对应 config 键、CLI override 链、同组 baseline}；命名→键的映射逐组给出代码锚点。
**产出**：官方消融矩阵 ↔ config 键对照表（组内公共资产，E3 选对的依据）。
**难度（估计）**：低。**工作量（估计）**：2-3 天。
**代码锚点**：`scripts/download_assets/download_openwam_checkpoints.py:120-185`；`configs/train.yaml:1-4`；`configs/model/dual_system.yaml:21,28-40`；`registry.py:96-130`。

### 任务 E-2：控制变量的确定性基础（seeding 体系 + 逐位复现验证）

**目标**：读懂 seeding 全链路，并在 CPU 上验证"同一 trainer 只切一个 config 键"的对照流程——两次完全相同的 run 产出逐位一致的 loss 序列。
**前置**：`pip install -e '.[dev]'`（CPU torch 即可）；无需权重。
**步骤**：
1. 跑现成单元测试，读它们各自钉住了哪一层：
```bash
cd OpenWAM
python -m pytest -q tests/test_seeding.py tests/test_dataloader_seed.py -v
```
2. 梳理调用链并写成一张图：`project.seed`（`configs/train.yaml:67`）→ `_inject_project_seed`（`scripts/train.py:93`）→ `seed_everything`（`scripts/train.py:131`，数据集构造前）→ `seed_process`（`openwam_trainer.py:83`，模型 init 前）→ `wire_sampler_seed`（`openwam_trainer.py:249`，shuffle 顺序）→ `per_step_seed`（`openwam_trainer.py:316-320`，每步 forward 前重播种）。回答三个问题：为什么 rank 之间必须隔离（`seeding.py:52 RANK_OFFSET`）？为什么每步要重播种而不是只靠全局种子？`seed=null` 时哪几层自动失效？
3. 写确定性验证脚本（CPU，mock 小模型即可，参考 `tests/test_architecture_variants.py:54-62 _MockVideoBackbone` 的思路）：固定 `seed=42` 跑两轮"构造模型 → 伪 batch → compute_loss"序列，断言 loss 逐位相等（`torch.equal` 而非 allclose）；再把 seed 改成 43 跑第三轮，断言 loss 不同；最后固定 seed、只切一个无关 config 键（如 `project.wandb.run_name`），断言 loss 仍逐位相等——这就是"只切一个变量"的操作定义。
4. 顺带验证 debug 模式语义：`openwam_trainer.py:190-193` 的 20 步/恒定 LR 是消融冒烟的标准协议，写进结论。
**验收标准**：确定性脚本三次运行结果如上断言全部成立；seeding 调用链图 + 三问回答成文；明确写出"在 CPU/小规模成立，GPU 多卡逐位一致受 cudnn/SDPA 非确定性影响，结论是'同配置同曲线可复现'而非跨硬件逐位"这一边界。
**产出**：确定性验证脚本 + seeding 机制说明（E3/重训对照的方法论底座）。
**难度（估计）**：低-中。**工作量（估计）**：2-3 天。
**代码锚点**：`openwam/train/utils/seeding.py:52,55,73,133,176`；`openwam_trainer.py:83,190-193,249,316-320`；`scripts/train.py:93-131`。

### 任务 E-3：复现一组官方消融（评测侧，省算力路线）

**目标**：下载同一组内的两个 study checkpoint（对照对），用方向 D 的 LIBERO harness 做对比评测，先小样本验证流程，再扩量出对比数字。
**前置**：方向 C 的部署 server 可用（[方向C-Deployment-Infra部署管线.md](方向C-Deployment-Infra部署管线.md)）；方向 D 的 LIBERO 评测 harness 就绪（[方向D-Evaluation-Infra评测管线.md](方向D-Evaluation-Infra评测管线.md)）；1×≥24GB GPU；E1 完成（据此选对）。
**步骤**：
1. 按 E1 对照表选一对"单变量"对照，推荐 attention_mask 组（机制最干净）：
```bash
python scripts/download_assets/download_openwam_checkpoints.py \
  --family study --study-type attention_mask --yes
# 取 robotwin_dual_system_joint_self_attention{,_mutual} 两个（各 ~24.8GB）
```
   注意：study checkpoint 多数在 robotwin 数据上训练，若 LIBERO 评测分布偏移过大，改用方向 C/D 已支持的对应 benchmark，或改选架构组对照对；选对理由写进报告。
2. 逐一起 server 并跑小样本评测（每任务 5–10 trials，先验证流程与确定性）：
```bash
bash scripts/deploy.sh assets/openwam_ckpt/openwam_study/attention_mask/<ckpt_a>
# 另一终端按方向D的 LIBERO 流程起 labtasker 客户端，限制 trials 数
```
3. 流程验证通过后扩到方向 D 的标准 trial 数，两个 checkpoint 用**同一 manifest**（同一批 seed/初始状态，方向 D 的 manifest 机制保证）各跑一遍。
4. 对比成功率，并读两个 checkpoint 目录里的 `config.yaml`（`checkpointing.py:22` 写出），逐键 diff 确认它们真的只差一个旋钮——若差得更多，这组对照不成立，回 E1 重选。
**验收标准**：两个 checkpoint 的 config.yaml diff 只含目标旋钮（及其派生）；同一 manifest 下的小样本与扩量两轮评测记录齐全；对比结论带置信说明（trials 数、方差）。
**产出**：官方消融复现报告（对照对选择依据 + config diff 证据 + 评测数字）。
**难度（估计）**：中。**工作量（估计）**：1 周（含下载与排队评测）。
**代码锚点**：`scripts/download_assets/download_openwam_checkpoints.py:175-184`（attention_mask 组）；`openwam/train/utils/checkpointing.py:22`；方向 D 文档的 manifest/评测锚点。

### 任务 E-4：测试不变量作为实验安全网（跑通 + 写"安全网使用手册"）

**目标**：跑通四组把"改一个变量不破坏其他"固化的测试，理解每组钉住的不变量，并写一份"做消融前/后该跑哪些测试"的使用手册。
**前置**：`pip install -e '.[dev]'`；无权重（全部 CPU 可跑）。
**步骤**：
```bash
cd OpenWAM
python -m pytest -q tests/test_split_attn_equivalence.py -v
python -m pytest -q tests/test_train_deploy_consistency.py -v
python -m pytest -q tests/test_mask_modes.py -v
python -m pytest -q "tests/test_architecture_variants.py::test_framework_backbone_hydra_compose" -v
```
1. 逐组读断言意图：
   - `test_split_attn_equivalence.py:1-8`：split 路径与整 block forward **零 atol** 等价——这是 joint_self_attn 消融的前提（拆层实现不产生数值偏移）；
   - `test_train_deploy_consistency.py:50`（idm 训练 action == 部署 Stage2，atol=1e-5）与 `:105`（noisy_video 扰动下逐位不变，atol=1e-6）——训练/推理两态一致，评测数字才可归因到训练；
   - `test_mask_modes.py:56,69,75,82,90`：四模式的 v↔a 可见性真值表——mask 消融（E3 推荐对）的语义基线；
   - `test_architecture_variants.py:767-768,781-783`：3 framework × 3 backbone = 9 格 Hydra compose 必须全部干净——E1 矩阵里每个格子的"合法性门禁"。
2. 做一组"故意破坏"对照：临时改一处实现（如 mask 填充方向），确认对应测试变红，再还原。这验证安全网真的在守。
3. 写使用手册：按消融维度（切 variant / 切 mask / 切 backbone / 切 encoder / 改训练循环）各列出"动手前必跑"的测试子集与通过标准；注明 GPU 变体需 `OPENWAM_RUN_TRAIN_DEPLOY_GPU=1`（`test_train_deploy_consistency.py:429`）否则静默 skip。
**验收标准**：四组测试全绿（输出存档）+ 一组破坏-变红-还原记录 + 安全网使用手册（按消融维度索引测试）。
**产出**：安全网使用手册（组内公共资产）。
**难度（估计）**：低。**工作量（估计）**：2 天。
**代码锚点**：四个测试文件的上述行号；`tests/conftest.py`（gpu marker 与 stub）。

### 任务 E-5：实验追踪与记录规范（wandb / debug csv / config.yaml + 组内模板）

**目标**：摸清三条记录通道——wandb 日志、`debug_loss_history.csv`、checkpoint 目录的自描述 `config.yaml`——并产出组内统一的实验记录模板。
**前置**：E1/E2 完成（知道旋钮与确定性边界）；跑过至少一次 debug 冒烟或读通 `openwam_trainer.py` 的 log_step/finish 路径。无需 GPU（读代码 + 分析 E2 产物即可）。
**步骤**：
1. wandb 通道：`openwam/train/utils/training_utils.py:88 init_wandb`（吃 `cfg.project.wandb`，`configs/train.yaml:68-71`）。官方镜像默认离线：`docker/Dockerfile:97 WANDB_MODE=offline`，日志落在 `wandb/offline-run-*`，事后 `wandb sync` 上传；要在线看板则启动前 `export WANDB_MODE=online` 并配 API key。验证：`project.wandb.run_name=e5_probe` 跑一次冒烟，确认 run 名/配置透传。
2. csv 通道：`training_utils.py:150 write_debug_loss_row`（`:165` 落盘 `debug_loss_history.csv`，首行写 header，列随 labels 走）——debug run 的轻量留痕，不依赖 wandb。
3. config 通道：`checkpointing.py:22 save_config`（`openwam_trainer.py:484` 调用，只写一次）——run 目录 = 自描述实验档案。配合 `ckpt_model_loader.py:216 build_architecture_from_ckpt_dir`，任何人拿到目录即可重建模型，不需要口述配置。
4. 产出模板：规定每个对照实验必须记录——CLI override 链全文、`project.seed`、git commit、checkpoint 目录路径、config.yaml diff（对 baseline）、wandb run 名/csv 路径、（E3 类）manifest hash 与 trials 数。模板里把"记录以 config.yaml 为准、口述/截图无效"写成第一条纪律。
**验收标准**：模板覆盖上述字段且每项指到产生它的代码通道；用 E2 或一次冒烟 run 实填一份样例。
**产出**：组内实验记录模板 + 填好的样例一份。
**难度（估计）**：低。**工作量（估计）**：1-2 天。
**代码锚点**：`training_utils.py:88,150-165`；`checkpointing.py:22`；`openwam_trainer.py:484`；`configs/train.yaml:68-71`；`docker/Dockerfile:97`。

## 5. 建议推进顺序与里程碑

1. **W1–W2（第一阶段，纯阅读）**：E1（矩阵考古）与 E4（安全网）并行启动；E2 读完 seeding 部分。里程碑 M1：消融矩阵对照表 + 安全网手册初稿（对应全局 M1 管线文档）。
2. **W3–W5**：E2 收尾（CPU 确定性验证）；E5 出模板并用 E2 产物实填样例。里程碑 M2：确定性结论 + 实验记录模板（对应全局 M2 的"E 确定性验证"）。
3. **W5–W8（等 C/D 就绪后）**：E3 官方消融复现——下载对照对、小样本、扩量。里程碑 M3：首批对照实验数字（对应全局 M3）。
4. **可选加分（W8+）**：小规模重训对照（如 mask mode 20 步 debug 级曲线对比），需训练算力，单独申请（README R3）。

依赖提示：E1 是 E3 的选对依据，E2 是"只切一个键"操作定义的来源，E4 是所有结论性断言的背书——前四个任务本质是一套方法论，E3 是它的第一次实战。

## 6. 风险与坑

1. **Study 完整训练配方未全部发布**：预训练混合有缺口——egocentric human reader 未注册、`configs/dataloader/pretrain_data/` 全是 `/path/to` 占位（证据链见 [方向A-Data-Infra数据管线.md](方向A-Data-Infra数据管线.md) A7，README R1）。**结论：本方向只能复现"评测对比"（E3）或"小规模重训对比"，不能复现论文 Study 章节的全量训练数字**。写报告时把这条边界写在开头，避免下游误用。
2. **yaml `mutual` vs 代码默认 `ACTION_SEES_VIDEO`**：三份 model yaml 写 `attention_mask_mode: mutual`（如 `configs/model/dual_system.yaml:29`），但代码默认是 `ACTION_SEES_VIDEO`（`single_system/vanilla.py:50` 等四处）。绕过 yaml 直接 build 会静默换掩码——同一对"对照实验"在不同构建路径下可能根本不在比同一个变量。规避：实验记录一律以 checkpoint 内 `config.yaml` 为准（E5 第一条纪律）；idm 变体忽略该字段（`dual_system/idm.py:443-446` 注释），不受此坑影响。
3. **`keep_last_k_ckpts=1` 会删中间 checkpoint**：`configs/train.yaml:55` 默认只留最新一个，`manage_checkpoints`（`checkpointing.py:340`）滚动清理，训练正常结束还会 `finalize_keep_weights_only`（`:368`）删光 resume state。规避：做对照实验需要中间步曲线/多步评测时，提前把要留的 `checkpoint_step_*` 拷出 run 目录，或显式调大 `training.keep_last_k_ckpts`。
4. **同名 checkpoint 跨组复用**：`robotwin_dual_system_joint_self_attention` 在 architecture/video_backbone/attention_mask 三组同名出现（`download_openwam_checkpoints.py:128,168,179`），是同一基座当三组 baseline。规避：E1 对照表标清；E3 下载时按组目录区分，别下重（各 ~25GB）。
5. **GPU 逐位一致 ≠ CPU 结论外推**：E2 的逐位一致在 CPU/小规模成立；GPU 上 cudnn/SDPA 存在非确定性，`seed_everything`（`seeding.py:73`）锁 cudnn 后也只能保证同硬件同配置可复现。规避：报告措辞用"同配置同曲线可复现"，不承诺跨硬件逐位。
6. **GPU 测试静默 skip**：`test_train_deploy_consistency.py:429 _GPU_INTEGRATION_FLAG` 等门控未设时即使安全网"全绿"也只覆盖了 CPU 部分。规避：E4 手册注明环境变量清单；"测试绿"一律标注实际跑了哪些用例。

## 7. 推荐阅读顺序

1. 本目录 [README.md](README.md) §2/§3——本方向在六管线中的位置与仅有的跨方向依赖（E3 ← C+D）。
2. `scripts/download_assets/download_openwam_checkpoints.py:99-185`——官方矩阵的"发布形态"，E1 的对账对象。
3. `configs/train.yaml` 全文 + `configs/model/dual_system.yaml`——旋钮总表，E1 的映射目标。
4. `openwam/model/architectures/registry.py:96-130`——(framework,variant) 如何收口成注册名（机制深入读 [方向B-Model-Infra模型管线.md](方向B-Model-Infra模型管线.md)）。
5. `openwam/train/utils/seeding.py` 全文 + `openwam_trainer.py:74-85,249,316-320`——E2 全部素材。
6. `tests/test_seeding.py` → `tests/test_mask_modes.py` → `tests/test_split_attn_equivalence.py` → `tests/test_train_deploy_consistency.py` → `tests/test_architecture_variants.py:767-830`——E4 按此顺序读，从种子到矩阵门禁。
7. `openwam/train/utils/checkpointing.py:22-40` + `training_utils.py:88-170`——E5 的三条记录通道。
8. 交叉阅读：[方向C-Deployment-Infra部署管线.md](方向C-Deployment-Infra部署管线.md) 与 [方向D-Evaluation-Infra评测管线.md](方向D-Evaluation-Infra评测管线.md)（E3 的两个硬依赖）；[方向F-OpenWAM-Alpha模型实现.md](方向F-OpenWAM-Alpha模型实现.md)（alpha 是 study 结论的产物，本方向的对照表是它的前置读物）。
