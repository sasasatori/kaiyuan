# 方向 B：部署与推理复现

> 一句话定位：把训练好的 OpenWAM checkpoint 通过 WebSocket PolicyServer 服务化，把「下载 → 起服务 → 客户端跑通」做成人人可复用的流水线，并量化推理延迟与优化开关收益。读者：第一次接触该模块的同学。
> 前置：先读本目录 README.md 的总览，了解本方向在全局中的位置；本方向依赖 [方向A-环境搭建与验证矩阵.md](方向A-环境搭建与验证矩阵.md)（环境可用 + checkpoint 已下载），产出供下游 [方向E-仿真评测复现.md](方向E-仿真评测复现.md) 使用（所有 benchmark 客户端都连本方向部署的 server）。

行号锚定：main @ `90e94ae`。引用格式 `path:line 符号`；若代码漂移，以符号名为准。

## 1. 这个方向做什么、为什么值得做

OpenWAM 的部署层是「训练 → 论文成功率」之间的最后一公里。它把训练侧保存的**自包含 checkpoint**（目录里既有 `config.yaml` 也有 `checkpoint_step_*.safetensors` 权重与归一化统计）加载为一个推理引擎，包一层「obs→action」语义，再通过一条持久 WebSocket 暴露成 policy server。仿真器（LIBERO、RoboTwin 等 8 个 benchmark）的客户端只发每步的图像 + prompt，动作回传后驱动仿真器——评测层被刻意做「薄」，所有图像拼接、归一化、动作反归一化都在服务端完成。

这个方向值得做有三层原因。第一，它是**全组的复用件**：方向 E 的每一个 benchmark 评测都必须先有一个能 ping 通的 server，方向 C 微调出的 checkpoint 也要经部署层验证才可用。第二，它是**性能研究的入口**：部署层自带 sync/async 两种执行器、async 降噪时间步调度（Latent-Forcing）、以及 DiT 速度缓存 / torch.compile / prompt 嵌入缓存 / 跳过 VAE 解码四组优化开关——「动作生成快不快」直接决定评测吞吐与实时部署可行性，而这些开关的收益与正确性（改了动作没有？）都还没有人系统量化过。第三，它能产出**文档级交付物**：WebSocket 线协议是 8 个 benchmark 客户端与服务端之间的冻结契约，把它写成逐条核对的协议文档，全组（尤其方向 E/H）都会反复引用。

你能学到：WebSocket 服务化设计、receding-horizon 动作执行（buffer-and-replan）、异步预取与 stale-skip、diffusion 降噪时间步调度、以及「如何为一个推理系统做延迟测量与消融实验」。

## 2. 最小必要背景

**① WebSocket 线协议（冻结契约）。** 客户端→服务端只有三种消息 `obs`/`reset`/`ping`，服务端回 `action`/`reset_ack`/`pong`/`error`，协议字符串常量是单一事实源：`openwam/deploy/server.py:60-75`（`OBS/RESET/PING/ACTION/RESET_ACK/PONG/ERROR`、`ERR_UNKNOWN_TYPE/ERR_OBS_VALIDATION/ERR_INTERNAL`、`MAX_MESSAGE_BYTES = 32 MiB`）。benchmark 客户端在 `benchmarks/utils/transport.py:20-31` 维护一份**镜像**常量，两边字符串必须逐字节一致。单条 obs 结构见 `openwam/deploy/server.py:14-37` 的 docstring：

```python
{"type": "obs",
 "images": {"head_camera": <b64_png>,            # 必填
            "left_wrist_camera": <b64_png>|null, # 可选
            "right_wrist_camera": <b64_png>|null},
 "prompt": "<原样转发给模型>", "state": [floats]} # state 可选
```

**② obs→action 主链路。** 一条 obs 的服务端旅程：`PolicyServer.predict()`（`openwam/deploy/server.py:189`）→ `ObsPreprocessor.preprocess()`（`openwam/deploy/obs_preprocess.py:120`，解码/校验/黑填充/L-shape 拼图）→ `WAMPolicy.predict_action()`（`openwam/deploy/policy.py:25`，唯一门面）→ 执行器（sync `openwam/deploy/executors/sync_executor.py:22` 或 async `openwam/deploy/executors/async_executor.py:175`）→ `JointInferenceEngine.generate()`（`openwam/deploy/engine.py:247`）→ `make_schedule()` 组时间步序列（`openwam/deploy/denoise_schedule.py:254`）→ `architecture.generate(...)`（`openwam/model/architectures/base.py:1361`）。

**③ sync vs async 是两个正交旋钮。** `inference.denoise_mode` 控制**降噪轨迹**（sync=视频/动作同步降噪；async=Latent-Forcing 领先-滞后轨迹），`inference.inference_mode` 控制**执行器**（sync=缓冲耗尽才生成；async=后台线程预取下一 chunk）。两者可独立组合，见 `configs/deploy.yaml:15-22`。

**④ 自包含 checkpoint。** 部署不需要训练代码现场与外部权重路径：`load_from_checkpoint_dir()`（`openwam/deploy/model_loader.py:61`）从 checkpoint 目录的 `config.yaml` 重建训练配置，挑最新的 `checkpoint_step_*.safetensors`（`model_loader.py:23-50`），并把归一化器从 `normalization_stats.npy` 恢复（`model_loader.py:293-299`）。Cosmos 系附带 `reason1/` 目录、tri_system 附带 `vlm_backbone/` 目录（`model_loader.py:135-155, 187-195`）。

**⑤ `configs/deploy.yaml` 全部旋钮（已逐行核对）。** `server.host/port`；`inference.denoise_steps`（默认 10）、`denoise_mode`、`lead_modality`、`variance_shift_alpha(≥1)`、`linear_offset(∈[0,1))`、`inference_mode`、`inference_horizon`（null=整段 chunk）、`inference_delay_steps`（async 专属）；`optimization.decode_video`（false=跳过 VAE 解码）、`optimization.dit_cache.{enabled,cosine_threshold,max_skips}`、`optimization.compile.*`（分架构 fast path）、`optimization.prompt_embed_cache.{enabled,maxsize}`。每个 `inference.*` 字段都有同名 CLI override（`--denoise-steps` 等），见 `openwam/deploy/server.py:439 _build_argparser`。

## 3. 代码地图

```
openwam/deploy/                     # 本方向主战场
├── server.py                       # ★入口：协议常量 + PolicyServer + CLI
│   #   server.py:60-75 协议常量；:122 PolicyServer；:189 predict；
│   #   :246 ws_handler；:313 merge_deploy_cfg；:339 build_server_from_config；:657 main
├── obs_preprocess.py               # obs 校验/解码/拼图；:18 ObsValidationError；:67 ObsPreprocessor
├── policy.py                       # WAMPolicy obs→action 门面；:25 WAMPolicy
├── engine.py                       # 引擎；:87 JointInferenceEngine；:247 generate；:182 _init_optimizations
├── denoise_schedule.py             # 时间步调度；:40 DenoiseConfig；:116 denoise_async_is_noop；
│   #   :131 schedule_sync；:171 schedule_variance_shift；:254 make_schedule
├── model_loader.py                 # 自包含加载；:61 load_from_checkpoint_dir；:237 repr_contract_from_cfg；
│   #   :377 _CommandAwareNormalizer；:427 _UnifyAwareNormalizer；:508 _build_normalizer
├── executors/
│   ├── sync_executor.py            # :22 SyncInferenceExecutor（buffer-and-replan）
│   └── async_executor.py           # :30 ExecutionConfig；:175 AsyncInferenceExecutor（后台预取+stale-skip）
└── optimizations/
    └── dit_cache.py                # :19 DiTVelocityCache（跨步 DiT 速度缓存）
scripts/deploy.sh                   # ★启动脚本（单卡前台 / NUM_GPUS 多卡各起一个服务）
scripts/deploy.py                   # 源码树启动器（deploy.sh 单卡模式直接 exec 它，deploy.sh:36）
scripts/inference_test/
├── inference_single_test.py        # ★参考客户端；:91 --state-dim（默认 20）；:93 --test 冒烟
└── inference_continuous_test.py    # 压测脚本；:38 -n 默认 128 请求；:54 --test 随机图
benchmarks/utils/
├── transport.py                    # 客户端协议镜像；:20-31 常量；:43 WSPolicyClient
└── client.py                       # 载荷工具；:37 resize_for_lshape_slot；:66 ServerError；
                                    #   :102 encode_numpy_b64；:123 build_payload
configs/deploy.yaml                 # ★部署 base 配置（全部旋钮，见 §2⑤）
```

## 4. 任务书

### 任务 B1：最小部署复现

**目标**：从 HuggingFace 下载 1 个 alpha checkpoint，`deploy.sh` 起服务，用两个参考客户端跑通 ping/predict/reset。

**前置**：方向 A 环境就绪（Docker 或 native 均可）；单卡 ≥24GB 显存（bf16 5B 骨干）；磁盘 ≥30GB。

**步骤**（命令可直接复制；在仓库根执行）：

```bash
# 1) 下载 checkpoint（约 24.8GB；--yes 跳过交互确认）
python scripts/download_assets/download_openwam_checkpoints.py \
  --family alpha --name OpenWAM-Alpha-Sim-RoboTwin-Full --yes
# 也可换 LIBERO 版：--name OpenWAM-Alpha-Sim-LIBERO（以下载器菜单实际列出的名字为准）

# 2) 确认 checkpoint 目录（下载器会打印落盘路径；下文记为 $CKPT）
ls $CKPT/config.yaml $CKPT/checkpoint_step_*.safetensors $CKPT/normalization_stats.npy

# 3) 查 state_dim —— 以 checkpoint 内 config.yaml 为准，不要信文档里的数字
grep -A2 'state_dim' $CKPT/config.yaml   # 取 model.architecture.state_dim 的值，记为 $SDIM

# 4) 起服务（前台，默认 ws://0.0.0.0:8848；首请求有 torch.compile 预热，耐心等待）
bash scripts/deploy.sh $CKPT
# 常用 override：--device cuda:1 --denoise-steps 5 --inference-mode async

# 5) 另开终端，单请求冒烟（ping → 一次 predict → reset 全过即通）
python scripts/inference_test/inference_single_test.py \
  --server ws://127.0.0.1:8848 --test --state-dim $SDIM

# 6) 连续压测（默认 128 个背靠背请求，每步打印延迟）
python scripts/inference_test/inference_continuous_test.py \
  --server ws://127.0.0.1:8848 --test --state-dim $SDIM
```

**验收标准**：单请求脚本 ping/predict/reset 三步全过；连续脚本跑满 128 请求无 `ServerError`；记录首帧延迟（含 compile 预热）、稳态 p50/p95 延迟与峰值显存（`nvidia-smi`）。

**产出**：可复跑的部署 SOP 一页（命令 + 耗时/显存记录），直接成为方向 E 的前置手册。

**难度（估计）**：低。**工作量（估计）**：0.5–1 天（下载时间另计）。

**代码锚点**：`scripts/deploy.sh`；`scripts/download_assets/download_openwam_checkpoints.py:391 --family`；`scripts/inference_test/inference_single_test.py:91`；`openwam/deploy/server.py:657 main`。

### 任务 B2：WebSocket 协议契约文档

**目标**：逐条复现并写成文档：obs 多视角字段规则、proprio 校验行为、reset 生命周期、错误码矩阵。

**前置**：B1 有一个可 ping 通的 server；读 `openwam/deploy/server.py:14-37` docstring 与 `obs_preprocess.py` 全文。

**步骤**：

1. 逐条实验 obs 规则（可用 `inference_single_test.py` 改参数，或写 30 行 websocket 脚本直连）：
   - `head_camera` 缺失/损坏 base64 → 期望 `obs_validation_error`；
   - wrist 字段缺失或 `null` → 单视角 ckpt 被忽略、多视角 ckpt 黑填充后按 `camera_layout` 拼 L-shape（`openwam/deploy/obs_preprocess.py:177-206`，`assemble_multiview_layout` 在 :201）；
   - `state` 非数值列表 → `ObsValidationError("state must be a flat numeric list/array")`（`obs_preprocess.py:217-221`）；
   - proprio 必需但缺 `state` → 报错（`obs_preprocess.py:223-226`）。
2. reset 生命周期：`reset` 清 policy/executor 缓冲（`server.py:220`），回 `reset_ack`；用 `inference_continuous_test.py --reset-every 20` 观察 chunk 计数归零。
3. 错误码矩阵：`unknown_message_type`（发 `{"type":"bogus"}`）、`obs_validation_error`、`internal_error`（`server.py:69-71`）三行×触发方式×消息体示例。
4. 把 `ObsValidationError` 的每个分支写成表驱动测试（无需 GPU，直接对 `ObsPreprocessor` 构造 cfg 实例）。

**验收标准**：协议矩阵文档（每条规则=触发条件+期望响应+代码锚点）+ 分支表驱动测试全绿。

**产出**：协议契约文档（方向 E/H 引用）；表驱动测试可交给方向 H（任务 H3）沉淀。

**难度（估计）**：低。**工作量（估计）**：1–2 天。

**代码锚点**：`openwam/deploy/server.py:60-75`；`openwam/deploy/obs_preprocess.py:18, 120-226`；`benchmarks/utils/transport.py:20-31`；`benchmarks/utils/client.py:123 build_payload`。

### 任务 B3：sync vs async 执行器延迟实测

**目标**：量化两种 `inference_mode` 的 p50/p95 延迟、参数敏感性与 stale-skip 行为。

**前置**：B1 完成；读 `openwam/deploy/executors/sync_executor.py:22-71` 与 `async_executor.py:30-381`。

**步骤**：

1. sync 基线：`--inference-mode sync` 起服务，`inference_continuous_test.py --test -n 128` 收延迟序列；再各扫一遍 `--denoise-steps {3,5,10}` 与 `--inference-horizon {null,8,16}`。
2. async 实测：`--inference-mode async --inference-horizon 16`，扫 `--inference-delay-steps {2,4,8}`（默认 null=horizon//2，见 `configs/deploy.yaml:22` 注释语义），观察后台预取命中与 `skip_steps` 过期前缀丢弃（`async_executor.py:175` 起的 stale-skip 逻辑）。
3. 极端情形：故意把 `--inference-delay-steps` 设远小于真实延迟，统计 stale-skip 频率与报错（`RuntimeError` 风险，见 §6）。
4. 汇总 p50/p95/吞吐曲线与「延迟差 − delay 估计差 → skip 次数」敏感性图。

**验收标准**：延迟曲线图 + 参数敏感性报告（每组 ≥128 请求；机器/ckpt/denoise_steps 固定）。

**产出**：执行器选型结论（什么场景该用 async、delay 怎么估）。

**难度（估计）**：中。**工作量（估计）**：2–3 天。

**代码锚点**：`openwam/deploy/executors/sync_executor.py:22`；`openwam/deploy/executors/async_executor.py:30, 175`；`scripts/inference_test/inference_continuous_test.py:38-54`。

### 任务 B4：denoise_schedule 等价性验证

**目标**：证明 `denoise_mode=async` 在 `variance_shift_alpha=1, linear_offset=0` 时与 sync **逐位一致**，并扫参看偏离。

**前置**：读 `openwam/deploy/denoise_schedule.py` 全文与 `openwam/model/architectures/base.py:1361 generate`；B1 环境。

**步骤**：

1. 纯函数层（无需 GPU）：构造两个 mock scheduler，调 `make_schedule("sync",...)` 与 `make_schedule("async", alpha=1, offset=0, ...)`（`denoise_schedule.py:254`），断言 `(t_video, t_action)` 序列逐位相等——`denoise_async_is_noop`（`denoise_schedule.py:116`）正是判这个「对角线即 sync」的谓词。
2. 端到端层（GPU）：同一 checkpoint 同一固定输入（固定随机图的 seed），分别用 `--denoise-mode sync` 与 `--denoise-mode async`（保持 alpha=1/offset=0 默认）各跑一次 predict，比较返回的 action 数组（应逐位一致或在 bf16 容差内）。
3. 扫参：`--variance-shift-alpha {1.0,1.5,2.0}` × `--linear-offset {0,0.2,0.4}` × `--lead-modality {video,action}`，记录 action 轨迹偏离（范数差）并做轨迹可视化。

**验收标准**：等价性测试（GPU 前后数值对照）+ alpha/offset/lead_modality 扫参的轨迹可视化图。

**产出**：「async 何时等于 sync、何时开始偏离、偏离长什么样」的一页结论。

**难度（估计）**：中。**工作量（估计）**：2 天。

**代码锚点**：`openwam/deploy/denoise_schedule.py:116, 131, 171, 254`；`openwam/deploy/engine.py:247`（schedule 在此组装后传入 `architecture.generate`）。

### 任务 B5：优化开关消融

**目标**：逐项开闭 `dit_cache` / `decode_video=false` / `prompt_embed_cache` / `compile`，对比延迟与动作差异；验证 CFG>1 时 dit_cache 静默失效的互斥。

**前置**：B1+B3 的测量流程；读 `openwam/deploy/engine.py:182 _init_optimizations` 与 `openwam/deploy/optimizations/dit_cache.py:19`。

**步骤**：

1. 固定 checkpoint/输入/请求数（≥128），以「全部默认开」为基线，逐项翻转：
   - `--compile-enabled false`（CLI 同名覆盖，`configs/deploy.yaml:33-34`）；
   - dit_cache：yaml 覆盖 `optimization.dit_cache.enabled=false`（deploy 接受任意 OmegaConf dotlist 追加在命令行尾部）；
   - prompt_embed_cache：`optimization.prompt_embed_cache.enabled=false`；
   - `optimization.decode_video=true`（观察 VAE 解码占多少时间）。
2. 每项记录 p50/p95 与「与基线 action 的最大绝对差」——开关必须只提速、不改动作（在 bf16 容差内）。
3. 互斥验证：将 `cfg_scale` 调到 >1（Cosmos 系；`base.py:1437-1438` 在 `cfg_scale_f > 1.0` 时直接 `dit_cache = None`，无任何告警），确认 dit_cache 被静默禁用；若本组只有 Wan ckpt，则做代码路径核对 + 在报告中标注「Wan 推理不做 CFG，本互斥仅影响 Cosmos」。

**验收标准**：消融表（开关 × p50/p95 × 动作差异）+ CFG 互斥验证记录（含证据行号）。

**产出**：部署推荐配置单（不同 GPU/ckpt 下该开什么）。

**难度（估计）**：中。**工作量（估计）**：2 天。

**代码锚点**：`configs/deploy.yaml:25-54`；`openwam/deploy/engine.py:63, 182`；`openwam/deploy/optimizations/dit_cache.py:19`；`openwam/model/architectures/base.py:1437-1438`。

### 任务 B6：自包含 checkpoint 加载审计

**目标**：构造缺件 checkpoint 触发各 `FileNotFoundError`，核对 Reason1/VAE/VLM/normalizer 各分支的加载行为。

**前置**：B1 的 checkpoint 一份（用拷贝做实验，别动原件）；读 `openwam/deploy/model_loader.py` 全文。

**步骤**：

1. 复制 checkpoint 目录到 `/tmp/ckpt_audit`，逐个删件起服务，记录报错：
   - 删 `config.yaml` → `FileNotFoundError: config.yaml not found`（`model_loader.py:85`）；
   - 删全部 `checkpoint_step_*.safetensors` → `model_loader.py:35`；文件名步数不可解析 → `model_loader.py:47`；
   - 删 `normalization_stats.npy`（且 cfg 开了 normalize）→ `model_loader.py:295` 的明确报错（含「旧版 action_stats.npy 改名」提示）。
2. 分支核对（代码阅读 + 有条件时实测）：
   - Reason1（CosmosPredict25）：ckpt 带 `reason1/` 目录时清空外部 `text_encoder_path` 走自包含（`model_loader.py:135-146`）；缺 `reason1/` 会在 `from_empty` 硬报错（注释 :128）；
   - VAE 组件：ckpt 自带 VAE state component 时走自包含，否则 `_resolve_vae_path` 在权重加载前抛 `FileNotFoundError`（`model_loader.py:151-155`）；
   - tri_system：`<ckpt_dir>/vlm_backbone/` 存在则接管 `checkpoint_path`（`model_loader.py:187-195`）；
   - normalizer 三层：`_build_normalizer`（:508）→ `_CommandAwareNormalizer`（:377，binary command dims ±1 投影）→ `_UnifyAwareNormalizer`（:427，unify_action ckpt 的 80-D 逆映射）。
3. 汇总「最小可服务 checkpoint」清单：哪几件必须有、哪几件按架构可选。

**验收标准**：加载失败矩阵（缺失件 × 架构家族 × 报错类型/信息 × 代码行号）+ 最小可服务 checkpoint 说明。

**产出**：checkpoint 完整性检查清单（方向 C 产出新 checkpoint 时可直接对照验收）。

**难度（估计）**：中。**工作量（估计）**：1–2 天。

**代码锚点**：`openwam/deploy/model_loader.py:35, 47, 85, 135-155, 187-195, 295, 377, 427, 508`。

## 5. 建议推进顺序与里程碑

1. **第 1 天：B1**（拿到可复跑 SOP；这是全组阻塞项，优先解除）。
2. **第 2–3 天：B2**（趁热把协议文档写掉，B3–B5 的实验脚本都依赖协议理解）。
3. **第 3–5 天：B3 + B4**（共用测量基建；B3 管「执行器」，B4 管「降噪轨迹」，两套 async 别混淆）。
4. **第 5–7 天：B5 + B6**（B5 复用 B3 的延迟统计脚本；B6 纯审计可与 B5 穿插）。
5. 里程碑：M1=B1 SOP 发布；M2=协议契约文档 + 延迟/等价性报告；M3=消融表 + 加载失败矩阵 + 部署推荐配置单。

## 6. 风险与坑

- **`--state-dim` 文档不一致**：README 写 `--state-dim 10`（`README.md:347,350`），`assets/openwam_usage_docs/docker.md:112` 写 `--state-dim 20`，脚本默认 20（`scripts/inference_test/inference_single_test.py:91`）。规避：**一律以 checkpoint 内 `config.yaml` 的 `model.architecture.state_dim` 为准**（服务端对 unify ckpt 故意不校验宽度，见 `obs_preprocess.py:211-216`，所以传错宽度不一定立刻报错，而是下游 Normalizer/模型才炸）。
- **HTTP_PROXY 影响 loopback**：`websockets>=15` 默认读 `HTTP(S)_PROXY/ALL_PROXY`，会把 `ws://127.0.0.1` 流量送进代理导致连不上。客户端已 `proxy=None` 规避（`benchmarks/utils/transport.py:82` 注释），但你自写的直连脚本要记得同样处理，或 `unset HTTP_PROXY HTTPS_PROXY ALL_PROXY`。
- **首请求 torch.compile 预热慢**：`optimization.compile.enabled=true`（`configs/deploy.yaml:34`）时首个 predict 可能慢到分钟级，属预期；测量延迟时丢弃首个请求（B3/B5 的 p50/p95 统计口径里要注明）。
- **单连接串行处理、慢推理阻塞后续消息**：`ws_handler` 对单连接串行分发（`openwam/deploy/server.py:246`），服务端设 `ping_interval=None`（心跳让位于长推理）；async 执行器后台预取但当前请求仍在 event loop 上等结果。规避：压测/评测一律单连接顺序发，别假设并发语义；需要多客户端并行评测时用 `NUM_GPUS=N bash scripts/deploy.sh` 一卡一服务（`scripts/deploy.sh:15-19`）。
- **async 的 `inference_delay_steps` 与真实延迟脱钩**：默认 `horizon//2` 是保守启发式（`configs/deploy.yaml:22`），估小了会频繁触发 stale-skip 甚至 `RuntimeError`。规避：先用 B3 测出真实 chunk 延迟换算成步数再设。
- **CFG>1 与 dit_cache 互斥无告警**：`base.py:1437-1438` 直接 `dit_cache = None`。规避：消融报告里写明；误配时优化静默失效，性能数字会「看起来没变」。

## 7. 推荐阅读顺序

1. `README.md`（仓库根）部署一节 —— 知道官方给出的最小命令长什么样（顺便亲眼验证 `--state-dim 10` 的不一致）。
2. `configs/deploy.yaml` 全文（54 行）—— 一次认全所有旋钮，后文到处引用。
3. `openwam/deploy/server.py:1-75`（docstring + 协议常量）—— 冻结线协议，B2 的地基。
4. `openwam/deploy/server.py:189-310`（`predict`/`reset`/`ws_handler`）—— 消息分发与串行语义。
5. `openwam/deploy/obs_preprocess.py` 全文 —— obs 校验规则与 L-shape 拼图（B2 直接素材）。
6. `openwam/deploy/policy.py` + `executors/sync_executor.py` —— 最短主链路与 sync 执行器。
7. `openwam/deploy/executors/async_executor.py` —— 后台预取与 stale-skip（B3）。
8. `openwam/deploy/denoise_schedule.py` —— 时间步调度与等价性谓词（B4）。
9. `openwam/deploy/engine.py:182-366` + `optimizations/dit_cache.py` —— 优化开关（B5）。
10. `openwam/deploy/model_loader.py` 全文 —— 自包含加载审计（B6）。
11. `benchmarks/utils/transport.py` + `benchmarks/utils/client.py` —— 客户端镜像，理解「薄客户端」契约（衔接方向 E）。
12. `assets/openwam_usage_docs/docker.md` —— Docker 部署路径与 healthcheck（方向 A 走 Docker 路线时对照）。
