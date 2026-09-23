# 方向 F：架构变体与注意力机制

> 一句话定位：复现并扩展 OpenWAM 的架构消融——6 个注册架构共享同一套生命周期，差异全部集中在 video↔action 的耦合方式与注意力掩码上，本方向要回答"哪种耦合在什么设定下最优"。
> 读者：第一次接触该模块的同学。
> 前置：先读本目录 README.md 的总览，了解本方向在全局中的位置。

仓库根记作 `$R = OpenWAM`，下文路径均为仓库相对路径。行号以 main @ 90e94ae 为准（文中以符号锚点为主，行号漂移时按符号名 grep 重新定位）。

## 1. 这个方向做什么、为什么值得做

OpenWAM 是「视频 DiT（世界模型）+ 动作 DiT（策略）+ 可选理解专家」的多流扩散架构。视频流和动作流怎么"说话"，直接决定策略能利用多少世界知识：是把动作 token 拼进视频序列共享注意力（single_system），还是两个独立 DiT 之间做交叉注意力（joint_cross_attn），还是逐层把 Q/K/V 拼接成一次混合自注意力（joint_self_attn / MoT），还是先让视频自己降噪、动作去 attend 冻结的视频 KV（idm）？这四种耦合范式就是论文 OpenWAM-Study 的核心设计空间。

架构层的实现非常"薄"：`openwam/model/architectures/base.py:154` 的 `BaseWAMArchitecture` 统一定义构造、冻结、scheduler 初始化、`compute_loss`/`generate` 生命周期；6 个具体架构类彼此没有继承关系，差异只在各自的 `forward` 怎么组织两侧 backbone。这意味着消融实验的控制变量天然干净——换耦合方式时，数据、加噪、loss、推理循环全都不变。

你能学到：多流扩散 Transformer 的几种主流耦合设计及其工程代价（显存、吞吐、几何约束）；注意力掩码作为"信息流阀门"的语义设计；以及如何用仓库已有的测试安全网（split-attn 零 atol 等价、detach 梯度不变量）做可信消融。

## 2. 最小必要背景

**BaseWAMArchitecture 生命周期** — 所有架构共用的骨架：构造时建视频 backbone，训练时 `compute_loss` 采样 timestep 加噪后调 `self.forward`，推理时 `generate` 逐步调 `forward`。
锚点：`openwam/model/architectures/base.py:154`（类）、`base.py:1044 compute_loss`、`base.py:1361 generate`、`base.py:1691` 抽象 `forward`。依赖方向严格单向：architecture → backbones，backbone 不得反向 import。

**registry 解析** — yaml 里写的 `(framework, variant)` 二元组经索引映射为 6 个注册名之一，再合并参数实例化。
锚点：`openwam/model/architectures/registry.py:96 resolve_architecture_config`；注册名清单（registry.py:12-17 docstring）：`single_system_vanilla`、`single_system_moe`、`dual_system_cross_attn`、`dual_system_self_attn`、`dual_system_idm`、`tri_system_joint_self_attn`。

**四种耦合范式（三族）**：
- single 共享序列：动作 token 经 `inject_shared_tokens` 拼到视频序列尾部，掩码挂到 `state.extras['shared_attention_mask']`，跑完同一个 DiT 再 `extract_shared_tokens` 取出（`single_system/vanilla.py:118,126,142`）。
- dual cross_attn bridge：视频 DiT 全跑完，按 `bridge_layers` 抓取中间 hidden_states 作为 K/V 送给独立的 ActionDiT 做交叉注意力（`dual_system/joint_cross_attn.py:43`）。
- dual self_attn MoT：`DualSystemMoTDriver`（`dual_system/mot_driver.py:38`）逐层把双侧 Q/K/V 拼接成单次混合自注意力再 split 回去，要求几何严格对齐。
- idm teacher-forcing：训练时把"噪声视频 + 干净条件视频 + 噪声动作"三分支合并进一次 joint loop；推理时视频先行、动作 attend 缓存的视频 KV（`dual_system/idm.py:392`）。

**mask 四模式** — `attention_mask_mode ∈ {mutual, action_sees_video, video_sees_action, isolated}` 控制 video↔action 两个方向的可见性（video↔video 由独立的 `video_attention_mask_mode` 控制）。
锚点：`openwam/model/architectures/utils/mask_modes.py:26-30`（四个常量）、`mask_modes.py:57 fill_cross_modal_va_blocks`、`mask_modes.py:90 build_cross_modal_attention_mask`。

**bridge_layers 解析** — `bridge_layers` 显式给层号列表；为 `null` 时用 `bridge_interval` 推导（1=每层，2=隔层）。
锚点：`openwam/model/architectures/utils/common.py:96 resolve_bridge_layers`。注意该字段在不同变体语义不同：joint_cross_attn 里是"哪些视频层给 bridge"，joint_self_attn/idm 里决定 ActionDiT 的层数（configs/model/dual_system.yaml:33-34 注释）。

## 3. 代码地图

入口文件：做消融先读 `openwam/model/architectures/registry.py`，再按变体进具体文件。

```
openwam/model/architectures/
├── base.py                      # 生命周期基类：compute_loss:1044 / generate:1361 / forward:1691（抽象）
├── registry.py                  # (framework,variant)→注册名解析 resolve_architecture_config:96
├── single_system/
│   ├── vanilla.py               # 【入口】SingleSystemVanillaArchitecture:37，共享序列最简实现
│   ├── moe.py                   # SingleSystemMoEArchitecture:39，bridge 层上加 expert FFN
│   └── state.py                 # 共享序列状态辅助
├── dual_system/
│   ├── joint_cross_attn.py      # DualSystemCrossAttnArchitecture:43；detach_bridge 生效点 :242
│   ├── joint_self_attn.py       # DualSystemSelfAttnArchitecture:44；代码默认 mask :97
│   ├── idm.py                   # DualSystemIDMArchitecture:392（1246 行，最长，见任务 F5）
│   └── mot_driver.py            # DualSystemMoTDriver:38；几何校验 :69-82
├── tri_system/
│   ├── joint_self_attn.py       # TriSystemJointSelfAttnArchitecture:67（本方向只做背景了解）
│   ├── mot_driver.py            # TriSystemMoTDriver:45，und 只读尾掩码 :113-176
│   └── und_expert.py            # UnderstandingExpert:82
└── utils/
    ├── mask_modes.py            # 四模式常量 :26-30、跨模态掩块填充 :57、联合掩码构建 :90
    └── common.py                # resolve_bridge_layers:96 等工具

openwam/model/action_backbone/
└── shared_action_backbone.py    # SharedVanillaActionBackbone:37；ExpertFFNBlock:86；
                                 # SharedMoEActionBackbone:137（expert_layer_to_index :177）

configs/model/
├── single_system.yaml           # attention_mask_mode: mutual（:28），与代码默认不一致，见 §6
├── dual_system.yaml             # variant 三选一；detach_bridge/idm_video_cond_noise_prob 在此（:38-40）
├── tri_system.yaml              # mot_checkpoint_mixed_attn: true（:39，关掉通常 OOM）
└── action_backbone/             # shared_action_backbone.yaml / separate_action_dit.yaml

tests/                           # 安全网（详见 方向H-测试与质量基建.md）
├── test_mask_modes.py           # 四模式行为测试（test_mutual:82 等）
└── test_wam_architecture.py     # detach 梯度不变量 test_..._blocks_grad_to_video:868
```

## 4. 任务书

### 任务 F-1：四种 attention_mask_mode 语义核对

**目标**：对照 `utils/mask_modes.py` 与 `tests/test_mask_modes.py`，逐模式验证 mutual / action_sees_video / video_sees_action / isolated 的 v→a、a→v 可见性及首帧行排除规则，产出模式对照矩阵。
**前置**：无 GPU、无权重；CPU 环境装好即可（`pip install -e '.[dev]'`）。测试用 `_MockVideoBackbone`（tests/test_architecture_variants.py:54-62）绕过真实权重。
**步骤**：
```bash
cd $R
python -m pytest -q tests/test_mask_modes.py -v        # 7 个用例，先看它们断言了什么
```
1. 读 `mask_modes.py:57 fill_cross_modal_va_blocks`：两个 if 分别决定 action→video（L76-78）与 video→action（L83-86）块是"全连"还是"仅首帧"，把四种模式两两填进矩阵。
2. 读 `tests/test_mask_modes.py:56 test_shared_invariants` 与 :69/:75/:82/:90 四个逐模式用例，确认你的矩阵与断言一致。
3. 写一个独立脚本用 mock backbone 构造 small joint mask，逐格打印 mask 值，肉眼核对（可选加分）。
**验收标准**：4×4（或 2×2 方向）模式对照矩阵 + 与测试断言的一致性确认记录；若发现测试未覆盖的格子（如与 `video_attention_mask_mode=causal` 叠加），单独标注。
**产出**：mask 语义矩阵（消融报告的附录材料）。
**难度（估计）**：低-中。**工作量（估计）**：1-2 天。
**代码锚点**：`utils/mask_modes.py:26-30,57,90`；`tests/test_mask_modes.py:50-98`。

### 任务 F-2：mask mode 训练消融（mutual vs action_sees_video）

**目标**：同一数据/骨干/步数下对比 `attention_mask_mode=mutual` 与 `action_sees_video` 的训练曲线，并评估"yaml 默认 mutual、代码默认 ACTION_SEES_VIDEO"这一不一致（见 §6 风险 1）对实验可复现性的影响。
**前置**：方向 C 的可训练环境（见 方向C-训练管线复现.md；4×80GB+，或先用 VACE-1.3B 小骨干）；F-1 完成。
**步骤**：
```bash
# 冒烟先跑通（20 步）
bash scripts/train.sh dataloader=libero model=dual_system \
  model/video_backbone=wan22_ti2v_5b model.architecture.variant=joint_self_attn \
  model.architecture.attention_mask_mode=mutual training.debug=true
# 正式两组实验仅改一个字段：
#   model.architecture.attention_mask_mode=mutual | action_sees_video
```
1. 固定种子/数据/骨干，只改 `model.architecture.attention_mask_mode`，各训一组。
2. 对比 loss_action 收敛曲线与（有条件时）LIBERO 成功率。
3. 影响评估：grep 全部代码默认值（`single_system/vanilla.py:50`、`single_system/moe.py:50`、`dual_system/joint_self_attn.py:97`、`tri_system/joint_self_attn.py:177` 均为 `ACTION_SEES_VIDEO`），论证"不带 yaml 直接 build"与"带 yaml 训练"实际跑的掩码不同；检查 idm 变体是否免疫（idm 忽略该字段，`dual_system/idm.py:443-446` 注释）。
**验收标准**：两条消融曲线 + 一段明确结论（含默认不一致的影响评估）；曲线需可复现（命令与随机种子记录完整）。
**产出**：消融曲线图 + 结论段落（论文 Study 章节素材）。
**难度（估计）**：中。**工作量（估计）**：1 周（含排队训练）。
**代码锚点**：`utils/mask_modes.py:57`；`configs/model/dual_system.yaml:29`；各架构默认值四处。

### 任务 F-3：dual 三变体对照（joint_self_attn vs joint_cross_attn vs idm）

**目标**：在同一骨干/数据/预算下对照三个 dual 变体的收敛速度、训练吞吐（it/s、峰值显存）与下游成功率，产出三变体对照表。
**前置**：方向 C 可训练环境；F-2 的训练流程已跑熟。三变体共用 `model=dual_system`，只切 `model.architecture.variant`。
**步骤**：
```bash
for V in joint_self_attn joint_cross_attn idm; do
  bash scripts/train.sh dataloader=libero model=dual_system \
    model/video_backbone=wan22_ti2v_5b model.architecture.variant=$V \
    training.debug=true    # 先各跑冒烟，确认能 build + 能 step
done
```
1. 记录每组：loss 曲线、it/s、`torch.cuda.max_memory_allocated`、（有条件）评测成功率。
2. 归因差异时对照实现：joint_self_attn 走 MoT 逐层拼接（`dual_system/mot_driver.py:38`）；joint_cross_attn 走 bridge 抓取 + ActionDiT 交叉注意力（`dual_system/joint_cross_attn.py:43`，注意 :238-242 的 Cosmos 5D→3D reshape）；idm 走 teacher-forcing 三分支（`dual_system/idm.py:644`）。
3. 注意 joint_self_attn/idm 有 `mot_checkpoint_mixed_attn` 而 joint_cross_attn 没有——吞吐对比时说明该旋钮状态（`configs/model/dual_system.yaml:38`）。
**验收标准**：三变体对照表（收敛/吞吐/显存/成功率四列以上）+ 每行差异的代码级归因。
**产出**：对照表（Study 章节主表）。
**难度（估计）**：中-高。**工作量（估计）**：2 周。
**代码锚点**：`registry.py:96`；三个变体文件类定义行；`configs/model/dual_system.yaml:33-40`。

### 任务 F-4：detach_bridge 梯度流验证

**目标**：验证 `detach_bridge=True/False` 对视频 DiT 梯度的阻断/放行效果，并做小规模训练对照评估其对最终性能的影响。
**前置**：F-3 中 joint_cross_attn 冒烟已通过；读懂 `tests/test_wam_architecture.py:868 test_dual_system_detached_joint_cross_attn_blocks_grad_to_video`。
**步骤**：
1. 先跑现成不变量测试：
```bash
python -m pytest -q tests/test_wam_architecture.py -k detached -v
```
2. 写一个梯度探测脚本：构造 joint_cross_attn（mock 或小骨干），分别在 `model.architecture.detach_bridge=true/false` 下前向 + 对 action loss 反传，断言 `video_backbone` 参数的 grad 为 None/非 None（复刻 :868 用例的断言逻辑，但在你的配置下）。
3. 训练对照：两组仅改 `model.architecture.detach_bridge`，对比 loss_video 是否被动作 loss 影响、最终动作性能。
**验收标准**：梯度探测脚本输出（True→视频侧无梯度，False→有）+ 训练对照曲线 + 结论（detach 是否值得默认开）。生效点必须指到 `dual_system/joint_cross_attn.py:242`（`bridge.detach() if detach_bridge else bridge`）。
**产出**：梯度探测脚本 + 对照结论。
**难度（估计）**：中。**工作量（估计）**：3-4 天。
**代码锚点**：`dual_system/joint_cross_attn.py:78,111-112,230-242`；`tests/test_wam_architecture.py:868`；配置默认值 `dual_system/joint_cross_attn.py:32`（False）。

### 任务 F-5：idm 两阶段推理复现

**目标**：复现 idm 的"Stage 1 视频独立降噪 + Stage 2 冻结视频 KV 的动作降噪"两阶段推理，核对训练侧 teacher-forcing mask 布局，并与联合推理（joint_self_attn）做延迟/质量对比。
**前置**：F-3 完成（有 idm 训练 checkpoint）或先用 mock backbone 走通逻辑；理解 idm 是六个变体中最复杂的（`dual_system/idm.py` 约 1246 行）。

先读懂两侧机制（这是本任务的核心学习量）：

- **训练三分支**（`dual_system/idm.py:644 _forward_idm_training`，由 `forward` 在 `cond_video_latents is not None` 时分派）：一次 forward 同时跑 A 噪声视频分支、B teacher-forcing 条件视频分支（以 `idm_video_cond_noise_prob` 概率加噪，0=干净，默认 0.5，`idm.py:453` 校验 ∈[0,1]）、C 噪声动作分支。`IDMMoTDriver.run_idm_training_loop`（`idm.py:101`）用 `vb.merge_idm_video_branches` 把 A/B 沿序列轴合并，由 `_build_teacher_forcing_mask`（`idm.py:59`）构造布局：v2v 各自子掩码、action→cond_video+action 全连、其余方向阻断，单次 joint loop 后 `split_idm_video_branches` 拆回。`compute_loss`（`idm.py:779`）对 A/B/C 独立采样 timestep。idm **忽略** `attention_mask_mode`（`idm.py:443-446` 注释）。
- **推理两阶段**（`dual_system/idm.py:934 generate`）：Stage 1 用 `_run_video_only_backbone` 独立把视频降噪干净；Stage 2 以 t=0 干净视频调 `driver.prefill_video_cache`（`idm.py:170`，`@torch.no_grad` 缓存每层 K/V + prefix 门），随后循环里 `run_action_with_video_cache`（`idm.py:246`，或编译版 `..._tensor_loop` `idm.py:301`）只跑动作分支、attend 缓存的视频 K/V。Stage 1 无视频步且无 `input_video_latents` 时 fail-fast（`idm.py:1105` 附近）。

**步骤**：
1. 用已有 checkpoint（或方向 C 的微调产物）起推理，逐 stage 打日志确认两段都走到：
```bash
bash scripts/deploy.sh <idm_ckpt_dir>     # 或写独立脚本直接调 architecture.generate
```
2. 在 Stage 2 循环里埋点：确认动作步不再跑视频 DiT（视频 K/V 来自缓存），记录每步耗时。
3. 对照组：同 checkpoint 格式下 joint_self_attn 的联合推理逐步耗时；产出延迟对比表（prefill 一次性成本 vs 每步节省）。
4. mask 布局核对：用小尺寸配置调 `_build_teacher_forcing_mask`（`idm.py:59`）打印分块掩码，对照 §2 的布局描述逐块确认。
**验收标准**：可运行的两阶段推理脚本 + 每阶段耗时分解表 + teacher-forcing mask 布局核对记录 + 与联合推理的延迟/质量对比结论。
**产出**：两阶段推理脚本与对比报告（高阶项，排期靠后）。
**难度（估计）**：高。**工作量（估计）**：2-3 周。
**代码锚点**：`dual_system/idm.py:392,644,779,934`；`idm.py:59,101,170,246,301`；`configs/model/dual_system.yaml:40`（`idm_video_cond_noise_prob: 0.5`）。

### 任务 F-6：single_system moe 的 expert FFN 贡献分析

**目标**：量化 `SharedMoEActionBackbone` 中 expert FFN 对动作预测的贡献，做 `bridge_layers` 位置/数量敏感性分析。
**前置**：方向 C 可训练环境；single_system moe 冒烟通过。
**步骤**：
1. 读懂机制：`openwam/model/action_backbone/shared_action_backbone.py:86 ExpertFFNBlock`、`:137 SharedMoEActionBackbone`（`expert_layer_to_index` :177，每个 bridge 层一个 expert）；架构侧在 bridge 层调 `ab.apply_expert` 做残差修正，越界有校验（`single_system/moe.py:84`）。
2. 敏感性网格：改 `configs/model/single_system.yaml` 的 `architecture.bridge_layers`（显式列表）或 `bridge_interval`，例如 `{全部层(interval=1)} × {每隔2层} × {仅后半段层} × {仅首层}` 四组，其余固定。
3. 极端对照：`bridge_layers` 只留 0 个/1 个 expert 时（interval 调大）退化为接近 vanilla，验证贡献单调性。
4. 每组记录 loss_action 收敛与（有条件）成功率；绘"expert 数量 vs 性能"曲线。
**验收标准**：敏感性报告：至少 4 组 bridge 配置的训练曲线 + 位置/数量两个维度的结论 + expert FFN 是否"值得其参数开销"的判断。
**产出**：敏感性报告（Study 章节素材）。
**难度（估计）**：中。**工作量（估计）**：1-1.5 周。
**代码锚点**：`shared_action_backbone.py:86,137,160,177`；`single_system/moe.py:39,84`；`utils/common.py:96 resolve_bridge_layers`。

## 5. 建议推进顺序与里程碑

1. **W1**：F-1（无 GPU 即可启动，顺带把 §2 概念读实）。里程碑：mask 语义矩阵。
2. **W2–W3**（等方向 C 环境就绪后）：F-2 启动训练；并行写 F-4 的梯度探测脚本（逻辑验证不依赖大规模训练）。里程碑：mask 消融曲线 + detach 梯度结论。
3. **W3–W5**：F-3 三变体对照（最吃算力，尽早排队）；F-6 可在 F-3 排队间隙用 single_system 跑。里程碑：三变体对照表。
4. **W6+**：F-5 idm 两阶段推理（依赖 F-3 的 idm checkpoint，逻辑最长，放最后）。里程碑：两阶段推理报告。

测试安全网：本方向所有结论性断言都应能对应到 `tests/` 既有不变量，测试维护与扩展见 方向H-测试与质量基建.md。

## 6. 风险与坑

1. **yaml `mutual` vs 代码默认 `ACTION_SEES_VIDEO` 不一致**：三份 model yaml 写 `attention_mask_mode: mutual`（`configs/model/single_system.yaml:28`、`dual_system.yaml:29`、`tri_system.yaml:31`），但代码默认是 `ACTION_SEES_VIDEO`（`single_system/vanilla.py:50`、`single_system/moe.py:50`、`dual_system/joint_self_attn.py:97`、`tri_system/joint_self_attn.py:177`）。绕过 yaml 直接 `build_architecture` 会静默换掩码。规避：实验记录一律以 checkpoint 内 config.yaml 为准；这正是 F-2 的影响评估对象。
2. **joint_self_attn 几何严格对齐**：`DualSystemMoTDriver` 构造期强校验 `num_layers`/`num_heads`/`head_dim` 双侧一致（`dual_system/mot_driver.py:69-82`），不满足即 raise，改 `bridge_layers`/`bridge_interval` 就是在改 ActionDiT 层数。规避：切骨干或改 bridge 配置后先跑 `training.debug=true` 冒烟，把构造期报错当预期行为而非 bug。
3. **tri_system 关 `mot_checkpoint_mixed_attn` 易 OOM**：`configs/model/tri_system.yaml:39` 注释明确"disabling typically OOMs Wan22 + Qwen3-VL"。本方向只做背景了解，但若顺手做 tri 实验，勿为"排除检查点干扰"关掉它；dual 侧同名字段（`dual_system.yaml:38`）只对 joint_self_attn/idm 生效，joint_cross_attn 没有该机制，F-3 对比吞吐时须说明。
4. **idm `lambda_action==0` 分支死计算**：`dual_system/idm.py:733` 注释 `TODO(perf)`——video-only（`lambda_action==0`）路径下 cond 分支被算了但从不读取（相关逻辑 `idm.py:732-738`、`:901`）。规避：复现 video-only 训练时知道这部分开销是已知的；不要把它当成自己引入的 bug，也不要在未评估前"顺手修复"。
5. **idm 对 backbone 的结构性要求**：teacher-forcing 需 backbone 支持 `merge/split_idm_video_branches`，且要求 per-token 4D `t_mod`（Wan 需 TI2V 原生或 `force_per_token_t_mod`，`idm.py:737-738` 为强制赋值）。规避：F-5 选骨干时优先 wan22_ti2v_5b；换 VACE/I2V 前先确认该路径。
6. **"CI 绿 ≠ 消融可信"**（母文档 R4）：CI 无 GPU/模型覆盖，长训练收敛未验证。规避：所有消融结论以本方向实测曲线为准，F-1 的矩阵只保证掩码逻辑正确，不保证训练效果。

## 7. 推荐阅读顺序

1. `openwam/model/architectures/registry.py`（:96 `resolve_architecture_config` + docstring 6 注册名）——知道 yaml 字段怎么变成对象。
2. `openwam/model/architectures/base.py`（:154 类、:1044、:1361、:1691）——知道训练/推理在哪进入架构。
3. `openwam/model/architectures/utils/mask_modes.py` + `tests/test_mask_modes.py`——F-1 的全部素材。
4. `openwam/model/architectures/single_system/vanilla.py`（:118/:126/:142 的注入/掩码/取出三步）——最简耦合范式，建立直觉。
5. `openwam/model/architectures/dual_system/mot_driver.py`（:38、:69-82）——MoT 逐层拼接与几何约束，F-3/F-5 的前置。
6. `openwam/model/architectures/dual_system/joint_cross_attn.py`（:43、:230-242）——bridge 与 detach，F-4 前置。
7. `openwam/model/architectures/dual_system/idm.py`（按 :392 → :644 → :59/:101 → :779 → :934 → :170/:246 顺序读）——F-5 专项，最后读。
8. `openwam/model/action_backbone/shared_action_backbone.py`（:86/:137/:177）——F-6 专项。
9. 交叉阅读：方向C-训练管线复现.md（F-2 至 F-6 的环境前置）、方向H-测试与质量基建.md（测试安全网的维护方式）。
