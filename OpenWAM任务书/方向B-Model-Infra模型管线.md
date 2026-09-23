# 方向 B：Model-Infra 模型管线

> 一句话定位：搞清并掌控"一个模型是如何从 yaml 搭出来的"——Hydra 配置组 → registry 解析 → 架构/骨干组装 → 冻结与优化器分组的完整链路，并具备自行扩展一个最小 backbone/架构变体的能力。读者：第一次接触该模块的同学。
> 前置：先读 [README.md](README.md) 的总览。**本方向第一阶段（W1–W2）纯阅读+文档化，不需要任何 GPU/环境**。

仓库根记作 `OpenWAM/`，下文路径均为仓库相对路径。行号以 main @ 90e94ae 为准（行号漂移时按符号名 grep 重新定位）。

## 0. 你的环境与资源路径

- **第一阶段（W1–W2，B1/B2 的阅读部分）**：零依赖。只需要一份源码 checkout 和编辑器，不装环境、不碰 GPU、不下载权重。
- **第二阶段（B2 核对、B3、B4、B6）**：纯 CPU 即可。装开发环境 `pip install -e '.[dev]'`（在 `OpenWAM/` 下），不需要任何模型权重——测试用 mock backbone 绕过真实权重：`_MockVideoBackbone`（定义于 tests/test_openwam_trainer.py:121，用法见 tests/test_architecture_variants.py:61）。cosmos 系组合在 CPU 上也能做 registry/build 级核对（重依赖延迟到 `from_pretrained`，见 openwam/model/video_backbone/__init__.py:47-57 注释）。
- **第三阶段（B5）**：才需要下载 encoder 权重，二选一即可：DINOv3（facebook/dinov3-vitb16，约 0.4GB，HF gated 需提前申请授权）或 FLUX.2-dev VAE（black-forest-labs/FLUX.2-dev，约 0.4GB，同样 gated）。下载走 `OpenWAM/scripts/download_assets/download_visual_encoder.py`，脚本会回写对应 yaml 的 `model_path`。B5 只要求"构建跑通"（CPU/单卡均可），不做训练消融。
- **你不依赖别人的什么**：不依赖统一训练环境（方向 C/D 的 GPU 集群）、不依赖数据集（方向 A）、不依赖任何训练好的 checkpoint。B5 的 gated 授权等待期用 B1–B4/B6 填充。

## 1. 这个方向做什么、为什么值得做

OpenWAM 的模型层是「视频 DiT + 动作 DiT + 可选冻结 VLM」的多流扩散架构，但它的实现刻意做得"薄"：`BaseWAMArchitecture`（openwam/model/architectures/base.py:154）统一生命周期，6 个架构类彼此无继承，5 个视频骨干、4 个 encoder、1 个 VLM 全部走 registry 插拔。这条"yaml → 对象图"的组装链路是全组所有实验的公共地基：方向 E 的每个消融、方向 F 的 OpenWAM-α 配置，第一步都是把某个 yaml 组合 build 成一个可训模型。链路里任何一个环节理解错（mask 默认值不一致、name/path 失配、冻结连带、encoder 门控），实验结论就会静默失真。

本方向的价值：把这条链路从"少数人能改"变成"全员可文档化、可审计、可扩展"。你能学到：registry/Hydra 组的工程化组装模式、多流 DiT 的耦合设计空间（作为"管线结构"理解，效果对照归方向 E）、可插拔 backbone 的 ABC 契约设计、以及冻结/优化器分组这类"训练基建语义"的审计方法。

与邻方向的边界：本方向回答"结构是什么、怎么搭、能不能搭"；**"搭出来哪个更好"（训练消融、曲线对照）归 [方向E-Study消融实验实现.md](方向E-Study消融实验实现.md)**；OpenWAM-α 模型的具体配置与微调归 [方向F-OpenWAM-Alpha模型实现.md](方向F-OpenWAM-Alpha模型实现.md)。

## 2. 最小必要背景

**三个 registry 对照** — 模型层有三套注册表，模式相同、粒度不同：
- architecture：`register_architecture`（openwam/model/architectures/registry.py:175）→ `resolve_architecture_config`（registry.py:96）把 yaml 的 `(framework, variant)` 二元组映射为 6 个注册名之一（清单见 registry.py:12-17 docstring）→ `build_architecture`（registry.py:229）。
- video_backbone：`register_video_backbone`（openwam/model/video_backbone/registry.py:18）→ `build_video_backbone`（registry.py:30）；5 个内置键注册在 openwam/model/video_backbone/__init__.py:43-57。
- encoder：`build_video_encoder`（openwam/model/video_backbone/encoder/registry.py:35），4 个键（wan22_vae/dinov3/vjepa21/flux2_vae）。
- 反例：action_backbone **没有 registry**，由各架构 `__init__` 直接 new；vlm_backbone 有 registry（build_vlm_backbone，openwam/model/vlm_backbone/registry.py:31）但只有 qwen3_vl_2b 一个键。

**VideoBackbone ABC 三段式** — 架构只通过三个方法驱动扩散循环：`prepare`（patchify/频率/时间调制）→ `run_block`（逐块跑 DiT，梯度检查点对架构透明）→ `finalize`（head + unpatchify）。
锚点：openwam/model/video_backbone/base.py:57 `class VideoBackbone`；`prepare` :122、`run_block` :126、`finalize` :130。

**action_backbone 两套 ABC** — `SharedActionBackbone`（single_system 用，动作 token 直接拼进视频序列，契约 `encode/encode_state/decode`）与 `ActionDiTBackbone`（dual/tri 用，独立动作 Transformer，契约 `forward/prepare_state/pre_attn_at_layer/post_attn_at_layer/extract_prediction`）。
锚点：openwam/model/action_backbone/base.py:38 / :147。

**mask 四模式** — `attention_mask_mode ∈ {mutual, action_sees_video, video_sees_action, isolated}` 控制 video↔action 两个方向的可见性。
锚点：openwam/model/architectures/utils/mask_modes.py:26-30（常量）、:57 `fill_cross_modal_va_blocks`、:90 `build_cross_modal_attention_mask`。

**bridge_layers 解析** — 显式层号列表，为 `null` 时用 `bridge_interval` 推导；语义随变体不同（cross_attn 里是"哪些视频层给 bridge"，self_attn/idm 里决定 ActionDiT 层数）。
锚点：openwam/model/architectures/utils/common.py:96 `resolve_bridge_layers`。

**外部 encoder 门控** — encoder 块**仅当 `from_scratch=true` 时才生效**；`from_scratch=false` 且 `encoder.name != wan22_vae` 直接 fail-fast（预训练 DiT 第一层 conv 通道绑死原生 Wan VAE 的 z_dim=48）。
锚点：openwam/model/architectures/base.py:199 `_init_video_backbone`；四象限注释 :229-237；闸门 :248；fail-fast raise :296-304。

**freeze 单点 API** — `freeze_modules`（openwam/model/architectures/base.py:742）：子树级 `requires_grad_(False)` + forward 包 `no_grad`；列表在各框架 yaml 顶层 `freeze:`。tri_system 有硬约束 `_NEVER_FREEZE = frozenset({"understanding_expert"})`（tri_system/joint_self_attn.py:85，拒绝逻辑 :241-245）。

## 3. 代码地图

入口文件：**先读 `openwam/model/architectures/registry.py`，再读 `openwam/model/architectures/base.py` 的 `__init__`/`_init_video_backbone`，然后按 yaml 的 defaults 跳进各 backbone。**

```
openwam/model/architectures/           ← 组装层（编排，不实现组件行为）
├── base.py                            # 【入口①】BaseWAMArchitecture:154；_init_video_backbone:199；
│                                      #   freeze_modules:742；compute_loss:1044；generate:1361
├── registry.py                        # 【入口②】resolve_architecture_config:96；build_architecture:229
├── single_system/vanilla.py           # SingleSystemVanillaArchitecture:37（代码默认 mask :50）
├── single_system/moe.py               # SingleSystemMoEArchitecture:39
├── dual_system/joint_cross_attn.py    # DualSystemCrossAttnArchitecture:43；detach_bridge 生效点 :242
├── dual_system/joint_self_attn.py     # DualSystemSelfAttnArchitecture:44（代码默认 mask :97）
├── dual_system/idm.py                 # DualSystemIDMArchitecture:392（最长，忽略 attention_mask_mode）
├── dual_system/mot_driver.py          # DualSystemMoTDriver:38；几何强校验 :69-82
├── tri_system/joint_self_attn.py      # TriSystemJointSelfAttnArchitecture:67；_NEVER_FREEZE :85
├── tri_system/mot_driver.py           # TriSystemMoTDriver:45；und 只读尾掩码 :113-176
└── utils/mask_modes.py, common.py     # mask 四模式 :26-30；resolve_bridge_layers common.py:96

openwam/model/video_backbone/
├── base.py                            # 【入口③】VideoBackbone ABC:57（prepare:122/run_block:126/finalize:130）
├── registry.py + __init__.py          # build_video_backbone registry.py:30；5 键注册 __init__.py:43-57
├── wan_backbone.py + wan/             # Wan 系实现；pipeline_builder.py:87 _warn_on_name_path_mismatch（仅 WARN）；
│                                      #   reinit.py:74 reinit_dit_from_scratch；encode.py:82 VACE+外部 encoder 互斥注释
├── cosmos_predict25_backbone.py + cosmos_predict25/   # Cosmos-Predict2.5-2B
├── cosmos3_backbone.py + cosmos3/     # Cosmos3-Edge（freeze_und=false 未实现）
└── encoder/                           # base.py:26 VideoEncoderProperties（z_dim:60/pixel_decode:64）；
                                       #   registry.py:35 build_video_encoder；wan22_vae/dinov3/vjepa21/flux2_vae

openwam/model/action_backbone/
├── base.py                            # SharedActionBackbone:38 / ActionDiTBackbone:147（无 registry）
├── separate_action_dit.py             # dual/tri 的独立动作 Transformer
└── shared_action_backbone.py          # single 用；ExpertFFNBlock:86、SharedMoEActionBackbone:137

openwam/model/vlm_backbone/
├── base.py                            # VlmBackbone:20（无 from_pretrained，构造即加载）
└── qwen3_vl_backbone.py               # Qwen3-VL-2B 实现

configs/model/                         ← Hydra 组（组选择即"选型"）
├── single_system.yaml / dual_system.yaml / tri_system.yaml   # 框架 yaml：freeze 列表、mask、bridge 在此
├── video_backbone/*.yaml              # 5 骨干 + encoder/ 子组（dinov3/flux2_vae/vjepa21/wan22_vae）
├── action_backbone/*.yaml             # shared_action_backbone / separate_action_dit
└── vlm_backbone/qwen3_vl_2b.yaml

openwam/train/utils/optimizer_groups.py  # video/action/other 三 LR 分桶 build_trainable_parameters:6
assets/openwam_usage_docs/architecture-extension.md        # 扩展 SOP（B4 的操作手册）
tests/test_architecture_variants.py      # mock 组装路径（:54-62）；tests/test_mask_modes.py（mask 行为）
```

## 4. 任务书

### 任务 B-1：模型组装链路全景文档化（主线）

**目标**：产出"一个模型是如何从 yaml 搭出来的"端到端图解文档：从 `configs/model/` Hydra 组选择（框架 yaml + video_backbone/action_backbone/vlm_backbone 组）→ `resolve_architecture_config`（openwam/model/architectures/registry.py:96）→ `build_architecture`（registry.py:229）→ `BaseWAMArchitecture.__init__` / `_init_video_backbone`（architectures/base.py:199）→ `freeze_modules`（base.py:742）→ optimizer 分桶。
**前置**：零依赖，纯阅读。
**步骤**：
1. 读 `configs/train.yaml` 的 `defaults` 与三份框架 yaml，画出 Hydra 组选择树（`model=dual_system` + `model/video_backbone=wan22_ti2v_5b` 这类 CLI 覆盖如何落到组文件）。
2. 逐函数跟踪 `resolve_architecture_config`：`(framework, variant)` 如何映射注册名、`architecture.*`/`action_backbone.*` 参数如何合并、`video_backbone`/`vlm_backbone` 子配置如何透传（registry.py:132-155）。
3. 跟踪 `BaseWAMArchitecture.__init__` → `_init_video_backbone`：encoder 门控四象限（base.py:229-237）、action_backbone 由架构直接 new、几何参数（num_layers/num_heads/text_dim）如何从 vb setdefault 回填。
4. 用一张序列图（mermaid 或表格）串联全链，每一跳标 `path:line 符号`。
**验收标准**：一篇组装链路图解（markdown），从任意一个 `configs/model/` 组合出发能照图复述每一步发生在哪个文件哪一行；至少覆盖 single/dual/tri 各一条链路差异点（tri 多 vlm_backbone 构建）。
**产出**：组装链路图解文档（全组公共参考，方向 E/F 的前置读物）。
**难度（估计）**：低-中。**工作量（估计）**：3-4 天。
**代码锚点**：registry.py:96,132-155,229；architectures/base.py:154,199,226,248,742；optimizer_groups.py:6。

### 任务 B-2：6 个注册架构与三族耦合范式梳理 + mask 四模式核对

**目标**：(a) 把 6 个注册架构按耦合范式归类成结构对照表：single 共享序列（vanilla/moe，`inject_shared_tokens` 拼接）、dual cross_attn bridge（`bridge_layers` 抓 hidden_states 送 ActionDiT 交叉注意力）、dual self_attn MoT（`DualSystemMoTDriver` 逐层拼 Q/K/V，dual_system/mot_driver.py:38）、idm 两阶段（训练 teacher-forcing 三分支 / 推理视频 KV 缓存）、tri 只读尾（und expert 作为 n_readonly_tail）；(b) 用 CPU 测试核对 `attention_mask_mode` 四模式的 v↔a 可见性矩阵。
**前置**：B-1 完成；CPU 开发环境。
**步骤**：
```bash
cd OpenWAM
python -m pytest -q tests/test_mask_modes.py -v     # 先看测试断言了什么
```
1. 每个架构填一行：注册名 / 类锚点 / 耦合机制一句话 / 关键配置字段 / ActionDiT 还是 Shared。
2. 读 `mask_modes.py:57 fill_cross_modal_va_blocks`，把 mutual/action_sees_video/video_sees_action/isolated 的 a→v、v→a 两个方向（含首帧行排除）填成对照矩阵，与 tests/test_mask_modes.py 的断言逐格核对。
3. 标注特例：idm 忽略 `attention_mask_mode`（dual_system/idm.py:443-446 注释）；yaml 写 mutual 而代码默认 ACTION_SEES_VIDEO（见 §6 风险 1）。
**验收标准**：6 架构结构对照表 + 4 模式可见性矩阵 + 与测试断言一致的确认记录。
**产出**：对照表 + 矩阵（B-1 文档的章节；方向 E 的消融在此矩阵上做对照）。
**难度（估计）**：中。**工作量（估计）**：3-5 天。
**代码锚点**：registry.py:12-17；mot_driver.py:38,69-82；mask_modes.py:26-30,57,90；idm.py:443-446；tests/test_mask_modes.py。

### 任务 B-3：骨干 × 架构可行性矩阵

**目标**：枚举 5 骨干 × 6 架构共 30 个组合，实测哪些能 build（骨架级，CPU 即可），不能的归因到具体约束与代码锚点。
**前置**：B-1/B-2 完成；不需要权重。
**步骤**：
1. 写 throwaway 矩阵脚本：对每个 (video_backbone, architecture) 组合走 `build_architecture` 路径，捕获异常归类。mock/骨架构建即可（参考 tests/test_architecture_variants.py:54-62 的 mock 注入模式；cosmos 系重依赖延迟到 `from_pretrained`，registry 级组合 CPU 安全）。
2. 必须验证并归因的已知约束：
   - VACE + 外部 encoder 互斥（wan/encode.py:82 注释）；
   - joint_self_attn 要求双侧几何严格对齐（层数/头数/head_dim，mot_driver.py:69-82 构造即 raise）；
   - cosmos_predict25 + joint_cross_attn 需手工钉 `+model.action_backbone.num_heads=16 +model.action_backbone.attn_head_dim=64`；
   - tri_system 必须有 vlm_backbone 配置（Qwen3-VL）；
   - cosmos3_edge 的 `freeze_und=false` 未实现（NotImplementedError，tests/test_cosmos3_registry.py 断言）。
3. 矩阵每格标 ✅ 可 build / ❌ + 报错原文 + 归因锚点。
**验收标准**：5×6 矩阵表 + 每个 ❌ 格的报错归因（锚到具体 `path:line`）。
**产出**：可行性矩阵（方向 E 选配置组合、方向 F 选 alpha 骨架时直接查表）。
**难度（估计）**：中。**工作量（估计）**：2-3 天。
**代码锚点**：architectures/base.py:199；mot_driver.py:69-82；video_backbone/__init__.py:43-57；tests/test_cosmos3_registry.py。

### 任务 B-4：按扩展 SOP 搭一个最小 video backbone 变体（动手）

**目标**：严格按 `assets/openwam_usage_docs/architecture-extension.md` 的流程，扩展一个最小 video backbone（或最小架构变体，二选一），走通"注册 → Hydra compose → CPU 冒烟（mock 权重）"全链，验证 B-1 文档的每一步都可操作。
**前置**：B-1 完成（你写的文档就是本任务的施工图）；CPU 环境。
**步骤**：
1. 照 SOP 子类化 `VideoBackbone`（实现三段式 + 几何属性），用 `@register_video_backbone('my_video')` 注册（SOP 示例在 architecture-extension.md:55-63）；骨架可大量复用 `_MockVideoBackbone`（tests/test_openwam_trainer.py:121）的最小实现思路。
2. 新增 `configs/model/video_backbone/my_video.yaml`，用 Hydra compose 验证 `model/video_backbone=my_video` 能解析到你的类。
3. CPU 冒烟：在 single_system vanilla 下 build + 一次 mock forward；注册重名应报 "already registered"（参照 tests/test_cosmos_predict25_registry.py:50 的断言模式）。
4. 把踩到的每个"SOP 没写但实际要做的事"回注到 B-1 文档。
**验收标准**：新注册键可 compose、可 build、可跑 mock forward；冒烟脚本输出完整；SOP 缺口清单。
**产出**：最小扩展样例（脚本 + yaml）+ SOP 修订注记。
**难度（估计）**：中。**工作量（估计）**：2-3 天。
**代码锚点**：architecture-extension.md:50-63；video_backbone/base.py:57,122-130；video_backbone/registry.py:18,30；tests/test_openwam_trainer.py:121。

### 任务 B-5：外部 encoder 接入机制复现（构建级）

**目标**：跑通"外部 encoder 替换原生 VAE"的**构建路径**：`from_scratch=true` 门控 → `build_video_encoder` → `reinit_dit_from_scratch` 的 DiT 重初始化触发，验证 z_dim 契约生效。只做构建与 encode 冒烟，**不做训练消融**（那是方向 E 的事）。
**前置**：B-3 完成；DINOv3（0.4GB，gated）或 FLUX.2 VAE（0.4GB，gated）权重已下载（`python scripts/download_assets/download_visual_encoder.py`）；CPU/单卡即可。
**步骤**：
1. 负例先行：`from_scratch=false` + `encoder.name=dinov3` 应触发 fail-fast（architectures/base.py:296-304），记录报错原文。
2. 正例：`model.video_backbone.from_scratch=true model.video_backbone.encoder.name=dinov3 model.video_backbone.encoder.model_path=<权重目录>` 构建架构，观察日志中的 encoder 构建（base.py:253）与 DiT 重初始化（wan/reinit.py:74 `reinit_dit_from_scratch`，经 `reinit_for_from_scratch` 调用，base.py:385-386）。
3. 契约核对：断言 `encoder.properties.z_dim != 48`（DINOv3 为 1408）、`pixel_decode=False`（encoder/base.py:60,64）；确认 `generate(decode_video=True)` 对该 encoder fail-fast（architectures/base.py:116-121）。
4. 用一帧随机图跑 `batch_encode`，记录 latent 形状契约。
**验收标准**：负例报错记录 + 正例构建日志（含 reinit 触发证据）+ z_dim/pixel_decode 契约断言通过记录。
**产出**：机制复现记录（机制理解归档；训练侧实验归方向 E）。
**难度（估计）**：中。**工作量（估计）**：2-4 天（含 gated 授权等待）。
**代码锚点**：architectures/base.py:226,248,253,296-304,385-386；wan/reinit.py:74；encoder/base.py:26,60,64。

### 任务 B-6：freeze 策略与优化器分组审计

**目标**：审计"哪些参数会被训练"这条链的语义正确性：`freeze_modules` 的子树 no_grad 语义（architectures/base.py:742）、`_NEVER_FREEZE` 约束（tri_system/joint_self_attn.py:85）、`build_trainable_parameters` 的 video/action/other 三 LR 分桶（openwam/train/utils/optimizer_groups.py:6）。
**前置**：B-1 完成；CPU 环境（mock backbone 可测）。
**步骤**：
1. 读 `freeze_modules`（base.py:742-）：确认两个效应（`requires_grad_(False)` + forward 包 `no_grad`）与"缺失名静默跳过"行为；用 mock 架构验证：冻结 `video_backbone` 后其参数 `requires_grad` 全 False，且 forward 在 no_grad 下执行。
2. 验证连带语义：冻结祖先模块会连带冻结其下可训练子节点（子树级）——写一个小用例证明这一点，并在文档中标注"yaml 目前只冻结完整子树所以不触发，部分微调会踩坑"。
3. 验证 `_NEVER_FREEZE`：对 tri 架构 freeze 列表加 `understanding_expert`，应被 :241-245 拒绝逻辑 raise。
4. 审计分桶：读 optimizer_groups.py:22-43——`get_trainable_modules()` 的顶层子模块按 video_backbone/action_backbone/其他 分三桶，`video_lr`/`action_lr` 只覆盖对应桶，other 桶永远用 base LR（configs/train.yaml:28-29）；frozen 参数因 `requires_grad=False` 根本进不了优化器。用 mock 架构构造三组参数，断言分桶归属与 LR 覆盖。
**验收标准**：4 项语义各有一条可运行的验证记录（脚本输出）+ 一页 freeze/optimizer 语义备忘（含连带冻结陷阱与 tri 特例）。
**产出**：审计备忘（方向 C/F 配训练超参时的依据）。
**难度（估计）**：中。**工作量（估计）**：2-3 天。
**代码锚点**：architectures/base.py:742；tri_system/joint_self_attn.py:85,241-245；optimizer_groups.py:6,22-43；configs/train.yaml:28-29。

## 5. 建议推进顺序与里程碑

1. **W1**：B-1（纯阅读，零依赖）。里程碑：组装链路图解初稿。
2. **W2**：B-2（CPU 测试核对）+ B-3 启动（矩阵脚本）。里程碑：mask 矩阵 + 6 架构对照表。同时提交 B-5 的 HF gated 授权申请。
3. **W3**：B-3 收尾 + B-4（动手扩展，检验 B-1 文档可操作性）。里程碑：5×6 可行性矩阵 + 最小扩展样例。
4. **W4**：B-6 审计；B-5 待授权到位后插入（2 天量级）。里程碑：freeze/optimizer 备忘 + encoder 机制复现记录。

依赖关系：B-1 是全组公共地基，最先做；B-2/B-3 相互支撑可并行；B-4 依赖 B-1 文档质量；B-6 独立；B-5 唯一有外部等待（gated 授权）。

## 6. 风险与坑

1. **yaml `mutual` vs 代码默认 `ACTION_SEES_VIDEO` 不一致**：三份框架 yaml 写 `attention_mask_mode: mutual`（configs/model/single_system.yaml:28 等），但代码默认是 `ACTION_SEES_VIDEO`（single_system/vanilla.py:50、dual_system/joint_self_attn.py:97 等四处）。绕过 yaml 直接 `build_architecture` 会静默换掩码。规避：B-1 文档中把"配置来源优先级"单列一节；实验记录一律以 checkpoint 内 config.yaml 为准。
2. **Wan21 一个类注册两个名字，name/path 失配仅 WARN**：`wan21_vace_1_3b` 与 `wan21_i2v_14b_480p` 注册到同一个 `Wan21` 类（video_backbone/__init__.py:44-45），实际加载由 `model_path` 决定；`_warn_on_name_path_mismatch`（wan/pipeline_builder.py:87）只 WARN 不中止——只改 name 不改 path 会静默加载错骨干。规避：B-3 矩阵实验逐格核对 name↔path 一致性。
3. **冻结祖先连带冻结可训练子节点**：`freeze_modules` 是子树级 no_grad 包装（architectures/base.py:742）；yaml 现冻结列表只含完整子树所以不触发，但"只训投影层"这类部分微调会踩坑。规避：B-6 用用例固化该语义，备忘中显式警示。
4. **VlmBackbone 无 `from_pretrained`**：与 video/encoder 的两段式不一致，构造即加载（vlm_backbone/base.py:20；registry.py:31 docstring 明示）；且其 state_dict 以 `vlm_backbone.` 前缀整体排除在架构 checkpoint 之外——在 VLM 上加可训练适配层会被静默丢弃。规避：tri 相关改动先读该文件头部注释。
5. **cosmos3_edge 的 `freeze_und=false` 未实现**：yaml 默认 `freeze_und: true`，设 false 直接 NotImplementedError（tests/test_cosmos3_registry.py 断言）。规避：B-3 矩阵中标注为"已知未实现"而非 bug。
6. **joint_self_attn 几何强校验**：改 `bridge_layers`/`bridge_interval` 就是在改 ActionDiT 层数，双侧层数/头数/head_dim 不一致构造即 raise（mot_driver.py:69-82）。规避：把构造期报错当预期行为；B-3 矩阵逐格记录。

## 7. 推荐阅读顺序

1. `openwam/model/architectures/registry.py`（:96 resolve_architecture_config + :12-17 docstring）——yaml 字段怎么变成对象。
2. `openwam/model/architectures/base.py`（:154 类、:199 _init_video_backbone、:229-237 门控注释、:742 freeze_modules）——组装与冻结主链。
3. `openwam/model/video_backbone/base.py`（:57 类 docstring 起）——三段式契约心智模型。
4. `openwam/model/action_backbone/base.py`（:38/:147）——两套动作 ABC 的分工。
5. `configs/model/` 三份框架 yaml + 任一 video_backbone yaml——yaml 注释即文档，对着 B-1 的图读。
6. `openwam/model/architectures/utils/mask_modes.py` + `tests/test_mask_modes.py`——B-2 的全部素材。
7. `openwam/model/architectures/single_system/vanilla.py` → `dual_system/mot_driver.py`（:38、:69-82）→ `dual_system/joint_cross_attn.py`——耦合范式由简到繁。
8. `assets/openwam_usage_docs/architecture-extension.md`——B-4 的操作手册。
9. `openwam/train/utils/optimizer_groups.py` + `configs/train.yaml:27-29`——B-6 的分桶语义。
10. 交叉阅读：[方向E-Study消融实验实现.md](方向E-Study消融实验实现.md)（消融在你建立的机制理解之上做对照）、[方向F-OpenWAM-Alpha模型实现.md](方向F-OpenWAM-Alpha模型实现.md)（OpenWAM-α 的具体配置归 F）。
