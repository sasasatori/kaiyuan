# 方向 G：视觉表征与骨干研究

> 一句话定位：研究 OpenWAM 最底层的可替换组件——视频骨干（5 个注册键）、外部视觉 encoder、动作骨干与 tri_system 的冻结 VLM，复现"预训练视觉表征 ↔ 生成式 VAE latent"的取舍实验。
> 读者：第一次接触该模块的同学。
> 前置：先读本目录 README.md 的总览，了解本方向在全局中的位置。训练相关命令的前置环境见 `方向C-训练管线复现.md`。

仓库根记为 `$R = OpenWAM`；下文所有路径均为仓库相对路径。行号以 main @ `90e94ae` 为准；若行号漂移，以符号锚点（`path:line 符号` 中的符号名）为准。

## 1. 这个方向做什么、为什么值得做

OpenWAM 是一个"世界-动作模型"（WAM）：视频 DiT 负责世界建模，动作 DiT 负责动作生成。本方向研究的是这个技术栈的**最底层可插拔组件**：视频骨干把原始帧编码成 latent 并提供扩散块循环；它默认用 Wan 系列自带的 VAE，但仓库预留了一个实验旋钮——当 `from_scratch=true` 时可以用外部预训练视觉 encoder（DINOv3 / V-JEPA2.1 / FLUX.2 VAE）替换原生 VAE。这正是论文"世界知识继承"议题的实验入口：生成式 VAE 的 latent 是为了像素重建优化的，而 DINOv3/V-JEPA 的特征是为了语义理解优化的，换掉它们意味着什么，是一个真正可发表的研究问题。

V-JEPA2.1 的特征维度（128 维 token 嵌入）与 Wan VAE 的 latent 维度（48）不一致，仓库为此提供了一条独立的 SVAE 工具链：先用 V-JEPA 在全数据集上收集高维特征，再训练一个小 VAE 把特征压到 48 维与 Wan VAE 对齐，最后通过 `svae_path` 接回主训练。这条链是 G3 的核心。

另一条表征路线是 tri_system 架构里的冻结 VLM（Qwen3-VL）：它不参与生成，只把 (prompt, 首帧) 编码成 hidden states 喂给一个可训练的"理解专家"（UnderstandingExpert），再投影进三模态联合注意力。这个专家到底贡献多少，是 G5 的消融问题。

做这个方向能学到：可插拔骨干的注册表/契约设计、DiT 从头重初始化的代价与机制、VAE latent 几何（z_dim、时空压缩率、因果首帧）、以及表征学习（自监督特征 vs 生成 latent）在下游控制任务上的实证方法。

## 2. 最小必要背景

**VideoBackbone ABC 三段式**。所有视频骨干都实现同一个抽象基类，架构只通过三个方法驱动扩散循环：`prepare`（前处理：patchify/频率/时间调制）→ `run_block`（逐块跑 DiT，梯度检查点对架构透明）→ `finalize`（head + unpatchify）。
锚点：`openwam/model/video_backbone/base.py:57 class VideoBackbone`；`prepare` :122、`run_block` :126、`finalize` :130。

**registry 工厂**。骨干不直接 new，而是通过名字注册/构建。5 个内置注册键：`wan22_ti2v_5b`、`wan21_vace_1_3b`、`wan21_i2v_14b_480p`、`cosmos_predict25_2b`、`cosmos3_edge`。
锚点：`openwam/model/video_backbone/__init__.py:43-57`（注册语句）；`openwam/model/video_backbone/registry.py:30 build_video_backbone`。

**外部 encoder 门控**。encoder 块**仅当 `from_scratch=true` 时才生效**；`from_scratch=false` 且 `encoder.name != wan22_vae` 会直接 fail-fast 报错。原因写在报错文案里：预训练 DiT 的第一层卷积通道数绑死原生 Wan VAE 的 z_dim=48，换 encoder 就必须重初始化 DiT。DINOv3（z_dim 1408）、V-JEPA2.1、FLUX.2 VAE 的 z_dim 都不是 48，因此这三个 encoder 都必须 `from_scratch=true`。
锚点：`openwam/model/architectures/base.py:199 _init_video_backbone`；门控四象限注释 :229-237；实际闸门 `if enc_cfg is not None and from_scratch:` :248；fail-fast raise :297-304；DiT 重初始化调用 :385-386 `reinit_for_from_scratch`。

**encoder ABC 契约**。每个 encoder 加载权重后导出一个 `VideoEncoderProperties`（冻结 dataclass），声明 latent 几何：`z_dim`（latent 通道数）、`spatial/temporal_compression`、`causal_temporal`（首帧是否独立成 token）、`pixel_decode`（能否解码回像素——DINOv3/V-JEPA 为 False，只能用 latent 级指标）。
锚点：`openwam/model/video_backbone/encoder/base.py:26 class VideoEncoderProperties`；`z_dim` :60、`causal_temporal` :63、`pixel_decode` :64；`batch_encode` 形状契约 :107-108。

**SVAE reducer**。把 V-JEPA 高维特征压到 `latent_dim=48`（与 Wan VAE 对齐）的小 VAE，作为 encoder 内部的可选冻结模块。
锚点：`openwam/model/video_backbone/encoder/svae/model.py:87 class SVAE`（`latent_dim: int = 48` :102，加载入口 `load_svae` :344）；`openwam/model/video_backbone/encoder/svae/reducer.py`（`build/reduce/effective_z_dim`）。

**action_backbone 两套 ABC**。动作侧不是独立模型，而是两组参数化 I/O/Transformer：`SharedActionBackbone`（single_system 用，把动作 token 直接拼进视频 DiT 序列，契约 `encode/encode_state/decode`）与 `ActionDiTBackbone`（dual/tri 用，独立动作 Transformer，契约 `forward/prepare_state/pre_attn_at_layer/post_attn_at_layer/extract_prediction`）。动作骨干**没有 registry**，由各架构 `__init__` 直接构造。
锚点：`openwam/model/action_backbone/base.py:38 class SharedActionBackbone`；:147 `class ActionDiTBackbone`。

**VlmBackbone**。tri_system 专用的冻结视觉-语言特征抽取器（Qwen3-VL-2B）。注意它与 video/encoder 的两段式不一致：**没有 `from_pretrained`**，构造即加载；且其权重以 `vlm_backbone.` 前缀整体排除在架构 state_dict 之外。
锚点：`openwam/model/vlm_backbone/base.py:20 class VlmBackbone`；实现 `openwam/model/vlm_backbone/qwen3_vl_backbone.py`。

**冻结与 `_NEVER_FREEZE`**。冻结走单点 API `freeze_modules`（子树级 `requires_grad_(False)` + `forward` 包 `no_grad`），列表在框架 yaml 顶层。tri_system 有一个硬约束：`understanding_expert` 永远不许冻结。
锚点：`openwam/model/architectures/base.py:742 freeze_modules`；`openwam/model/architectures/tri_system/joint_self_attn.py:85 _NEVER_FREEZE = frozenset({"understanding_expert"})`（拒绝逻辑 :242）。

## 3. 代码地图

```
openwam/model/
├── video_backbone/                  ← 本方向主战场
│   ├── base.py                      【入口文件①】VideoBackbone ABC（三段式 + reinit_for_from_scratch :157）
│   ├── registry.py                  build_video_backbone (:30) 工厂
│   ├── __init__.py                  5 个注册键 (:43-57)
│   ├── wan_backbone.py              Wan 系骨干实现（reinit_for_from_scratch :197）
│   ├── wan/                         Wan 管线 internals
│   │   ├── pipeline_builder.py      权重加载；_warn_on_name_path_mismatch (:87) 只 WARN
│   │   ├── encode.py                VAE/VACE 编码；VACE+外部 encoder 互斥注释 (:82)
│   │   ├── reinit.py                reinit_dit_from_scratch (:74) DiT 全随机重初始化
│   │   ├── encode.py / models/      文本编码与 DiT 定义
│   │   └── shared/configs/model_configs.py  三套 Wan 几何硬编码（dim/层数/头数）
│   ├── cosmos_predict25_backbone.py + cosmos_predict25/   Cosmos-Predict2.5-2B（pipeline_builder.py 几何/权重通配）
│   ├── cosmos3_backbone.py + cosmos3/                     Cosmos3-Edge（vendored transformer，双流+und 前缀 KV）
│   └── encoder/                     外部 encoder 层
│       ├── base.py                  【入口文件②】VideoEncoder ABC + VideoEncoderProperties
│       ├── registry.py              build_video_encoder
│       ├── wan22_vae.py             原生 VAE 包装（默认）
│       ├── dinov3.py / vjepa21.py / flux2_vae.py   三个外部 encoder（pixel_decode=False）
│       └── svae/                    model.py（SVAE）+ reducer.py（降维封装）
├── action_backbone/                 动作扩散侧（两套 ABC，无 registry）
│   ├── base.py                      SharedActionBackbone (:38) / ActionDiTBackbone (:147)
│   ├── components.py                ActionEncoder/ActionOutputMLP（注意：只是 MLP，不是 LAPA）
│   ├── separate_action_dit.py       dual/tri 的独立动作 Transformer
│   └── shared_action_backbone.py    single_system 的共享骨干
└── vlm_backbone/                    tri_system 专用冻结 VLM
    ├── base.py                      VlmBackbone ABC (:20，无 from_pretrained)
    ├── registry.py                  build_vlm_backbone
    └── qwen3_vl_backbone.py         Qwen3-VL-2B 实现
```

配置与脚本：
- `configs/model/video_backbone/*.yaml` — 5 个骨干 yaml；`wan22_ti2v_5b.yaml:11 from_scratch`、:5 encoder 可选值注释。
- `configs/model/video_backbone/encoder/*.yaml` — encoder 组；`vjepa21.yaml:16 svae_path`、:17 `svae_target_dim`。
- `configs/model/video_backbone/encoder/svae/train.yaml` — SVAE 训练配置（`latent_dim: 48` :7、`beta: 1.0e-4` :31、`num_epochs: 50` :18）。
- `scripts/svae_train/collect_svae_features.sh`（MASTER_PORT 29502，:49）/ `train_svae.sh`（29503，:43）。
- `scripts/download_assets/download_video_backbone.py` / `download_visual_encoder.py` / `download_vlm_backbone.py` — 权重下载（会回写 yaml 的 model_path）。
- `scripts/install_cosmos_predict25.sh` — Cosmos 依赖安装（编译 transformer-engine）。
- `openwam/model/architectures/base.py:199 _init_video_backbone` — 所有骨干构建的唯一入口。

## 4. 任务书

### 任务 G1：五骨干几何/代价基线表

**目标**：对 5 个注册骨干各测一组基线：几何参数（dim/层数/头数/latent z_dim）、单卡加载峰值显存、单次 forward 时延。
**前置**：环境可用（见方向C）；至少下载 Wan2.2-TI2V-5B 与 Wan2.1-VACE-1.3B 两个权重（其余可后补）。
**步骤**：
1. 下载权重（脚本会回写 yaml 的 `model_path`）：
   ```bash
   python scripts/download_assets/download_video_backbone.py --help   # 先看条目清单
   ```
2. 几何参数不必实测，直接抄代码里的硬编码：`openwam/model/video_backbone/wan/shared/configs/model_configs.py`（Wan2.2-5B：dim=3072/24头/30层/in_out_dim=48，:441-455；VACE-1.3B：dim=1536/12头/30层，:184-197；I2V-14B：dim=5120/40头/40层，:88-99）；Cosmos 两骨干看各自 `pipeline_builder.py` 的 `_*_NET_KWARGS` / `_*_GEOMETRY`。
3. 写一个 throwaway 测量脚本：`build_video_backbone(name, cfg)` 加载到单卡，`torch.cuda.max_memory_allocated()` 记峰值显存，`torch.cuda.Event` 计时 `prepare→逐块 run_block→finalize` 一轮。bf16、固定输入形状（如 B=1, T=33, 384×320，对齐 dataloader 默认）。
4. 汇总成 markdown 表：骨干 | 参数量 | z_dim | 峰值显存 | forward ms | 权重体积。
**验收标准**：5 行基线表 + 每行可复现的命令/脚本。
**产出**：基线表（markdown）+ 测量脚本。
**难度（估计）**：低。**工作量（估计）**：1–2 天（主要是等权重下载）。
**代码锚点**：`openwam/model/video_backbone/registry.py:30 build_video_backbone`；几何硬编码见上。

### 任务 G2：外部 encoder 替换端到端复现（DINOv3 或 FLUX.2）

**目标**：跑通"外部 encoder 替换原生 VAE"的完整路径：`from_scratch=true` + encoder 块 + DiT 重初始化，训练 smoke 跑通并做 latent 可视化。
**前置**：G1 完成；DINOv3 需提前在 HF 申请 gated 授权（facebook/dinov3-vitb16），FLUX.2-dev 同理；单卡可起步但正式训练需多卡（见 §6 风险）。
**步骤**：
1. 授权后下载 encoder 权重：`python scripts/download_assets/download_visual_encoder.py`（选 dinov3 或 flux2_vae 条目）。
2. 训练 smoke（20 步 debug 模式）：
   ```bash
   bash scripts/train.sh \
     dataloader=libero \
     model=dual_system \
     model/video_backbone=wan22_ti2v_5b \
     model.video_backbone.from_scratch=true \
     model.video_backbone.encoder.name=dinov3 \
     model.video_backbone.encoder.model_path=<dinov3 权重目录> \
     training.debug=true
   ```
   观察日志中 DiT 重初始化（`reinit_dit_from_scratch`，`openwam/model/video_backbone/wan/reinit.py:74`）与 encoder 构建（`architectures/base.py:253 build_video_encoder`）。
3. latent 可视化：注意 DINOv3 `pixel_decode=False`（`encoder/base.py:64`），**不能解码回像素**；用 PCA/t-SNE 对 `batch_encode` 输出的 (B, 1408, T_lat, H_lat, W_lat) latent 做降维可视化，并与原生 Wan VAE 的 48 维 latent 对照。若选 FLUX.2 VAE（pixel_decode=True），可直接解码像素对照。
**验收标准**：smoke 训练日志无错 + 至少一组 latent 可视化图 + 与原生 VAE 的对照说明。
**产出**：跑通记录 + 可视化报告。
**难度（估计）**：中。**工作量（估计）**：3–5 天（含 gated 授权等待）。
**代码锚点**：门控 `architectures/base.py:248`；fail-fast :297-304；重初始化 :385-386。

### 任务 G3：SVAE 两阶段复现

**目标**：在 V-JEPA2.1 上走完 SVAE 工具链：收集特征 → 训 reducer → 接回主训练，验证 latent 维度变成 z_dim=48，并监控 active_units 防 latent 坍缩。
**前置**：V-JEPA2.1 权重（16.9GB，官方直链下载，见 §6）；benchmark 数据集；G2 的经验有助于理解 encoder 层。
**步骤**：
1. 下载 V-JEPA2.1：`python scripts/download_assets/download_visual_encoder.py`（vjepa2_1_vitg_384 条目，直链 `dl.fbaipublicfiles.com`，见 `download_visual_encoder.py:111`）。
2. **阶段一：收集特征**（encoder 必须 `svae_path=null`，否则 `collect_svae_features.py:155-158` 直接报错）：
   ```bash
   bash scripts/svae_train/collect_svae_features.sh \
     model.video_backbone.encoder.name=vjepa21 \
     model.video_backbone.encoder.model_path=/path/to/vjepa2_1_vitg_384 \
     model.video_backbone.encoder.svae_path=null \
     training.batch_size=4
   ```
   注意嵌套入口下 encoder 组选择不解析，必须用上面这种 FIELD 覆盖写法（`collect_svae_features.py:126` 注释）。产物：分片 `features_rank{r}_part{p}.pt` + `stats.pt`。
3. **阶段二：训练 reducer**（默认配置 `configs/model/video_backbone/encoder/svae/train.yaml`：`latent_dim: 48`、`beta: 1.0e-4`、`num_epochs: 50`）：
   ```bash
   bash scripts/svae_train/train_svae.sh train.features_dir=<阶段一输出目录>
   ```
   产物 `svae.pt`；训练日志里的 `_latent_diagnostics`（`scripts/svae_train/train_svae.py`）会报告 active_units 等诊断量，画成曲线盯坍缩。
4. **接回验证**：主训练中设置
   ```bash
   model/video_backbone=wan22_ti2v_5b \
   model.video_backbone.from_scratch=true \
   model.video_backbone.encoder.name=vjepa21 \
   model.video_backbone.encoder.svae_path=<svae.pt 路径> \
   training.debug=true
   ```
   断言 encoder 的 `properties.z_dim == 48`（reducer 生效后 `effective_z_dim`，`encoder/svae/reducer.py`）。**两个坑**：encoder 须 `svae_path=null` 才能 collect；主训练必须 `from_scratch=true` 整个 encoder 块才生效。
**验收标准**：`svae.pt` 产物 + 训练曲线（含 active_units 监控）+ 主训练中 z_dim=48 的验证记录 + latent 诊断说明。
**产出**：SVAE 训练产物 + 诊断报告。
**难度（估计）**：中-高。**工作量（估计）**：5–7 天（全数据集特征收集耗时占大头）。
**代码锚点**：`scripts/svae_train/collect_svae_features.sh:49`（port 29502）、`train_svae.sh:43`（port 29503）；`encoder/svae/model.py:87 SVAE`。

### 任务 G4：骨干×架构可行性矩阵

**目标**：枚举 5 骨干 × 6 架构（single_system: vanilla/moe；dual_system: joint_self_attn/joint_cross_attn/idm；tri_system: joint_self_attn）共 30 个组合，实测哪些能 build，不能的归因到具体约束。
**前置**：无需全部权重——build 冒烟可用 `materialize_weights=false` 级别的骨架构建（CPU 即可）；需要各骨干的 yaml cfg。
**步骤**：
1. 写 throwaway 矩阵脚本：对每个 (video_backbone, architecture) 组合调用 `build_architecture` 路径（`openwam/model/architectures/registry.py resolve_architecture_config` → `BaseWAMArchitecture.__init__`），捕获异常并归类。
2. 已知约束（需在矩阵中验证并归因）：
   - **VACE + 外部 encoder 互斥**：`wan/encode.py:82` 注释标明 fail-fast；
   - **cosmos_predict25 + joint_cross_attn 需手工钉头数**：`+model.action_backbone.num_heads=16 +model.action_backbone.attn_head_dim=64`（其 DiT 是 16 头×128 head_dim，与默认动作 DiT 几何不对齐）；
   - **tri_system 必须有 vlm_backbone 配置**（Qwen3-VL）；
   - **joint_self_attn 要求 ActionDiT 与视频 DiT 几何严格对齐**（层数/头数/head_dim，driver 构造即报错——此项与方向F的 F3 互补：F3 做收敛对照，G4 只做 build 可行性）；
   - **cosmos3_edge 的 `freeze_und=false` 未实现**（NotImplementedError）。
3. 矩阵每格标注：✅ 可 build / ❌ + 报错信息 + 归因代码锚点。
**验收标准**：5×6 矩阵表 + 每个 ❌ 格的报错归因（锚到具体文件:行）。
**产出**：可行性矩阵 + 报错归因表。
**难度（估计）**：中。**工作量（估计）**：2–3 天。
**代码锚点**：`openwam/model/architectures/base.py:199 _init_video_backbone`；各架构构造点（如 `dual_system/joint_cross_attn.py`、`tri_system/joint_self_attn.py`）。

### 任务 G5：tri_system 理解专家参与度分析

**目标**：量化 tri_system 中"VLM → vlm_projector → understanding_expert（und）"这条理解支路的贡献：做 und 只读尾消融（冻结/移除 und），并验证 `_NEVER_FREEZE` 硬约束。
**前置**：方向C的可训练环境；Qwen3-VL-2B 权重（4.3GB）；大显存（tri_system 关掉 `mot_checkpoint_mixed_attn` 通常 OOM）。本任务与方向F的 F5（idm 两阶段推理）互补，可共用 tri 环境经验。
**步骤**：
1. 读契约：`tri_system/joint_self_attn.py:70-97`——und 是架构级**可训练**专家；`UnderstandingExpert` 构造在 :139。
2. 验证 `_NEVER_FREEZE`：在 yaml 的 freeze 列表里加 `understanding_expert`，启动训练，应被 :242 的拒绝逻辑拦下（`joint_self_attn.py:85 _NEVER_FREEZE = frozenset({"understanding_expert"})`）。记录报错作为约束验证证据。
3. 基线 run：标准 tri_system 训练 smoke，记录 loss 曲线与（若可行）下游指标。
4. 消融 run A（移除 und）：将 und 支路绕过（`joint_self_attn.py:189` 有 `self.understanding_expert is None` 的分支，说明无 und 是合法配置；构造时把 `model.architecture.understanding_expert` 置空），同数据同步数对照。
5. 消融 run B（冻结 VLM 本体之外的全部 vs 只冻结 VLM）：`vlm_backbone.vlm_model` 本就在冻结列表里；对比冻结范围的显存/性能影响，写 profiling 报告。
**验收标准**：und 贡献消融报告（基线 vs 移除 und 的对照曲线 + 结论）+ `_NEVER_FREEZE` 约束验证记录。
**产出**：消融报告。
**难度（估计）**：高。**工作量（估计）**：5–8 天（tri_system 显存与训练成本最高）。
**代码锚点**：`tri_system/joint_self_attn.py:85/_NEVER_FREEZE`、:139 und 构造、:189 无 und 分支；`vlm_backbone/base.py:20 VlmBackbone`。

### 任务 G6：Cosmos 路线打通（选做）

**目标**：把 Cosmos-Predict2.5-2B 骨干的环境与 smoke 跑通，记录安装坑。
**前置**：nvcc + cuDNN 环境（编译 transformer-engine 用）；建议提前一周发起（见 §6 风险 R9）。
**步骤**：
1. 拉子模块与装依赖：
   ```bash
   git submodule update --init --recursive
   bash scripts/install_cosmos_predict25.sh
   ```
2. 下载权重：`python scripts/download_assets/download_video_backbone.py` 的 Cosmos-Predict2.5-2B 条目（nvidia/Cosmos-Predict2.5-2B + 文本编码器 nvidia/Cosmos-Reason1-7B，共约 21GB）。
3. smoke：用 `model/video_backbone=cosmos_predict25_2b` 跑 `training.debug=true`；joint_cross_attn 变体记得钉 `+model.action_backbone.num_heads=16 +model.action_backbone.attn_head_dim=64`。
4. 把每一步的报错与解法记入"安装坑记录"。
**验收标准**：smoke 通过 + 安装坑记录（每条含报错原文与解法）。
**产出**：环境 SOP 片段 + smoke 日志。
**难度（估计）**：高。**工作量（估计）**：3–5 天（编译与环境问题不可预期）。
**代码锚点**：`scripts/install_cosmos_predict25.sh`；`cosmos_predict25/pipeline_builder.py`（几何与权重通配）。

## 5. 建议推进顺序与里程碑

1. **第 1 周**：G1（基线表，同时等 G2 的 HF gated 授权与 G3 的 V-JEPA 直链下载）。里程碑：基线表。
2. **第 2–3 周**：G2（外部 encoder 端到端）+ G4（build 矩阵，CPU 可做，与 G2 并行）。里程碑：替换路径跑通 + 5×6 矩阵。
3. **第 3–5 周**：G3（SVAE 两阶段，特征收集可挂后台跑）。里程碑：`svae.pt` + z_dim=48 验证。
4. **第 5 周起（高阶/选做）**：G5（需 tri 大显存环境）、G6（选做，依赖编译环境）。里程碑：und 消融报告 / Cosmos smoke。

依赖关系：G2/G3 依赖方向C的可训练环境；G4 无 GPU 依赖可最早穿插；G5 依赖 G1 的显存常识与方向C环境。

## 6. 风险与坑

1. **外部 encoder 必须 from_scratch，算力代价大**。DINOv3/V-JEPA/FLUX.2 的 z_dim ≠ 48，门控强制 `from_scratch=true`，意味着 DiT 全随机重初始化、从头训练（`architectures/base.py:297-304` 报错文案、`wan/reinit.py:74`）。这不是微调路径，正式实验需单独申请算力（仓库 README 推荐 8×80GB）。规避：G2/G3 一律先用 `training.debug=true` 20 步 smoke 验证链路，再评估是否投入正式训练。
2. **DINOv3/FLUX.2 是 HF gated 资产**。需人工在 HuggingFace 申请授权后才能下载（`scripts/download_assets/download_visual_encoder.py` 对应条目）。规避：开工第一周就提交授权申请，等待期先做 G1/G4。
3. **V-JEPA 走官方直链，可达性存疑**。权重托管在 `dl.fbaipublicfiles.com`（`download_visual_encoder.py:111` 的 `direct_url`），无 HF 镜像。规避：尽早下载并落盘共享存储；若直链不可达，记录为阻塞项并切换 G3 的 encoder 为 DINOv3（SVAE 链同样支持）。
4. **Wan21 一个类注册两个名字，name/path 失配仅 WARN**。`wan21_vace_1_3b` 与 `wan21_i2v_14b_480p` 注册到同一个 `Wan21` 类（`video_backbone/__init__.py:44-45`），实际加载由 `model_path` 决定；`pipeline_builder.py:87 _warn_on_name_path_mismatch` 只 WARN 不中止——只改 name 不改 path 会**静默加载错骨干**。规避：G4 矩阵实验时逐格核对 name↔path 一致性。
5. **Cosmos 需 submodule + 现场编译 transformer-engine**。`scripts/install_cosmos_predict25.sh` 依赖 nvcc/cuDNN，编译失败是 G6 最常见的阻塞。规避：提前一周发起；把编译错记入安装坑记录。
6. **无 LAPA/latent-action 模块**。README 提到 "SVAE / LAPA tooling"，但全仓 grep 不到任何 `lapa`/`LatentAction` 标识符（已核验）；`action_backbone/components.py` 的 `ActionEncoder` 只是 raw action→hidden 的 MLP，不是可学习的 latent action 抽象。规避：本方向不做任何基于 LAPA 的假设；如论文需要 latent action 分析，明确标注为仓库缺口。
7. **外部 encoder 不可像素解码**。DINOv3/V-JEPA `pixel_decode=False`（`encoder/base.py:64`），`generate(decode_video=True)` 会 fail-fast（`architectures/base.py:117-119`）。规避：G2/G3 的可视化一律走 latent 级（PCA/t-SNE/active_units），不要试图解码成视频。

## 7. 推荐阅读顺序

1. 本目录 `README.md` —— 全局定位与方向间依赖。
2. `openwam/model/video_backbone/base.py` —— 先建立三段式契约的心智模型（57 行起的类 docstring 写得很好）。
3. `openwam/model/architectures/base.py:199-390` 的 `_init_video_backbone` —— 搞清 encoder 门控与 from_scratch 语义，这是 G2/G3 的地基。
4. `configs/model/video_backbone/wan22_ti2v_5b.yaml` + `configs/model/video_backbone/encoder/vjepa21.yaml` —— 看注释就知道所有旋钮（yaml 注释即文档）。
5. `openwam/model/video_backbone/encoder/base.py` —— VideoEncoderProperties 契约，理解 z_dim/pixel_decode 对实验设计的约束。
6. `openwam/model/video_backbone/encoder/svae/model.py` + `scripts/svae_train/` 两个脚本 —— G3 开工前读。
7. `方向C-训练管线复现.md` —— 所有 `train.sh` 命令的环境与配置背景。
8. `openwam/model/architectures/tri_system/joint_self_attn.py:60-200` —— G5 开工前读 und 契约与 `_NEVER_FREEZE`。
9. `方向F-架构变体与注意力机制.md` 的 F3/F5 节 —— 与 G4/G5 互补，避免重复劳动。
