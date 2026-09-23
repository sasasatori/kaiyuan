# 方向 C：把训练存档变成一个随叫随到的服务

> 把训练存档（checkpoint，模型权重加它的配置说明书）启动成一个 7×24 的网络服务：机器人（或仿真器）连上来，发一张"我现在看到什么"的照片，服务回一句"下一步这么动"。
> 这个方向把服务从下载到上线的每一步搞清楚、量明白。前两周只需读代码写文档，不需要显卡、不需要装环境。

> **这个方向的最终目标**：学完后，给你任何一个训练存档，你都能独立把它部署成服务——起服务、发通请求、量出延迟、调开加速开关。
> **分阶段路线**：① 最小部署，读懂协议和存档格式（C1–C3）→ ② 性能实测与加速开关调优（C4–C6）→ ③ 用 Docker 打包服务、演练离线搬运（C7）。

## 你需要准备什么

| 什么东西 | 什么时候才需要 |
|---|---|
| 1 张 ≥24GB 显存的显卡 + 磁盘 ≥30GB | C1、C4、C5、C6，同一台机器复用 |
| 一个官方 α 存档（约 25GB） | C1 起，用仓库自带的下载脚本拉，命令见任务 C1 |
| Docker 三件套（Engine、Compose、NVIDIA Container Toolkit）+ 磁盘 ≥60GB | C7；导出和校验离线包不需要显卡，最后在目标机起服务才需要 |

前两周纯阅读，什么都不用准备。不需要 benchmark 数据，不需要训练环境，也不用等别人的产出。只有想多客户端并行压测时才需要多张卡（一卡起一个服务）。

## 先搞懂这几件事

### 1. 客户端和服务端怎么对话：像点菜

WebSocket 是一种"长连接"：客户端连上服务后这条线一直挂着，随发随收，不用每次重新握手。这个服务就像一个点菜窗口，全部对话只有六种消息：

- 客户端能说的三句：**下单**（`obs`，发照片加一句任务指令）、**换桌要喊清台**（`reset`，让服务端忘掉做了一半的动作计划）、**在吗**（`ping`）。
- 服务端会回的四句：**上菜**（`action`，"下一步这么动"）、**清台完毕**（`reset_ack`）、**在**（`pong`）、**出错**（`error`，带错误码）。

一条"下单"长这样（JSON 格式）。两条铁规矩：消息字符串是双方签死的合同，服务端有一份"标准答案"，benchmark 客户端那边抄了一份一模一样的副本，两边必须逐字节一致；单条消息最大 32 MiB。

```python
{"type": "obs",
 "images": {"head_camera": <b64_png>,            # 必填，base64 编码的 PNG
            "left_wrist_camera": <b64_png>|null, # 可选
            "right_wrist_camera": <b64_png>|null},
 "prompt": "<原样转发给模型>", "state": [floats]} # state 可选
```

### 2. 一条请求在服务端经历了什么

服务端收到"下单"后按固定流水线处理：先预处理（解码照片、检查格式、缺手腕相机补黑图、多视角拼成一张 L 形大图）；再过一个统一门面，所有请求都从这一个入口进；然后执行器决定干活节奏（下一节细说）；推理引擎组装"降噪时间表"（扩散模型从噪声一步步还原出结果，这份表就是步骤安排）；最后模型真正开算，返回动作。图像拼接、归一化这些活全放在服务端做，是故意的：客户端越"薄"越好，仿真器只管发照片、收动作。

### 3. 有两套"sync/async"开关，别搞混

代码里有两个独立的开关，名字都叫 sync/async，管的是两件事。两个旋钮可以任意组合；注意切到 sync 模式时，配置会自动把 async 专属的参数重置掉。

- `inference.denoise_mode` 管**模型内部**的降噪时间表：sync 是视频和动作同步降噪；async 是一种叫 Latent-Forcing 的排法，让动作领先、视频滞后。
- `inference.inference_mode` 管**服务对外**的干活节奏。用厨师类比：sync 是"做完全部菜才上下一桌"，一批动作用完了才去算下一批；async 是"边做边上"，后台线程提前做下一批，客人催得紧就把过期的头几步扔掉。

### 4. 训练存档是自包含的

部署不需要训练代码在场，也不需要外部权重路径。加载器读存档里的 `config.yaml` 重建训练配置，自动挑编号最新的权重文件，再从 `normalization_stats.npy` 恢复"归一化器"（训练时把动作数值缩放到统一范围的小工具，部署时靠它把模型输出还原成真实动作）。有的架构还会在存档里多带东西：Cosmos 系带 `reason1/` 目录（文本编码器），三系统架构带 `vlm_backbone/` 目录（语言模型），VAE（把图像压成潜变量的组件）要么自带、要么直接报错。

### 5. 一个配置文件装下所有开关

部署配置文件一共 54 行，所有开关都在这：服务地址和端口；降噪步数（默认 10）；上面说的两套模式；async 专属的延迟参数；四个优化开关——跳过 VAE 解码、DiT 缓存（相邻两步输出很像就复用，阈值 0.99、最多连跳 3 步）、torch.compile（编译加速，按架构分档）、prompt 嵌入缓存（同一句指令不重复编码，最多存 32 条）。每个 `inference.*` 参数都能在启动命令里临时覆盖。

## 任务清单

### 任务 C1：下载存档，把服务跑起来

- 这件事是干嘛的：从 HuggingFace 下载一个官方 α 存档，用启动脚本把服务跑起来，再用两个参考客户端各发一遍测试请求。这是全组的阻塞项：方向 D 的每次评测、方向 F 的任务 F5，都先要有这个能 ping 通的服务。
- 你要准备什么：1 张 ≥24GB 显卡 + 磁盘 ≥30GB。不依赖任何其他方向。
- 具体怎么做（命令照抄，在仓库根目录执行）：

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

- 做完交出什么：一页可复跑的部署操作手册（SOP，命令 + 耗时/显存记录）。验收标准：单请求脚本 ping/predict/reset 三步全过；连续脚本跑满 128 个请求不报错；记下首帧延迟（含编译预热）、稳态延迟的中位数和最慢 5% 那一档（即 p50/p95）和峰值显存（用 `nvidia-smi` 看）。大概要 0.5–1 天（下载时间另计）。

### 任务 C2：写一份"服务语言手册"

- 这件事是干嘛的：把客户端和服务端之间的约定逐条试出来、写成文档：照片字段的规则、机器人关节状态的校验行为、reset 的完整流程、三种错误码分别在什么情况下出现。这份文档全组反复引用，前两周纯阅读就能开工，先写骨架，C1 之后补实验验证。
- 你要准备什么：前两周只要源码；实验验证需要 C1 那个能 ping 通的服务。
- 具体怎么做：
  1. 逐条试 obs 规则（改参考客户端的参数，或写 30 行直连脚本）：`head_camera` 缺失或图片编码损坏，期望收到 `obs_validation_error`；手腕相机缺失或填 `null`，单视角存档会忽略它，多视角存档会补黑图再拼 L 形大图；`state` 不是数字列表会收到明确报错；存档要求关节状态却缺 `state`，也会报错。
  2. 试 reset 流程：`reset` 会清空服务端的动作缓冲并回 `reset_ack`；压测脚本加 `--reset-every 20` 能观察到计数归零。
  3. 做错误码矩阵：`unknown_message_type`（发 `{"type":"bogus"}` 触发）、`obs_validation_error`、`internal_error`，三行 × 触发方式 × 消息体示例。
  4. 把校验错误的每个分支写成一张"出错情况 → 期望报错"的对照表，逐行自动验证（这叫表驱动测试；不需要显卡，直接构造预处理对象）。
- 做完交出什么：协议矩阵文档（每条规则 = 触发条件 + 期望响应 + 代码出处）+ 分支表驱动测试全绿。大概要 1–2 天。

### 任务 C3：解剖训练存档

- 这件事是干嘛的：把"一个能用的存档最少要装哪几件东西"写成权威文档。做法是故意构造缺件的存档，逐个触发报错，核对每个架构家族的加载分支。以后任何人产出新存档，都能对照这份清单验收。阅读半前两周可完成，实测半等 C1 的存档到位。
- 你要准备什么：前两周读加载器源码；实测时用 C1 存档的拷贝，别动原件。
- 具体怎么做：
  1. 把存档复制到 `/tmp/ckpt_audit`，逐个删件再起服务，记录报错：删 `config.yaml` 会报"config.yaml not found"；删全部 `checkpoint_step_*.safetensors`，或把文件名改到步数解析不出来，各有明确报错；删 `normalization_stats.npy`（且配置开了归一化），报错里还会提示"旧版 action_stats.npy 改名"。
  2. 核对各架构分支（读代码为主，有条件就实测）：Cosmos 系存档带 `reason1/` 目录就走自包含，缺了会直接报错；VAE 组件自带就走自包含，否则在加载权重之前报错；三系统架构看到 `vlm_backbone/` 就用它接管语言模型；归一化器有三层，基础版之上还有"指令敏感版"（把二元指令维度做 ±1 投影）和"unify 版"（unify_action 存档的 80 维逆映射）。
  3. 汇总"最小可服务存档"清单：哪几件必须有，哪几件按架构可选。
- 做完交出什么：加载失败矩阵（缺哪件 × 架构家族 × 报错内容 × 代码出处）+ 最小可服务存档说明。大概要 1–2 天。

### 任务 C4：实测两种干活节奏谁更快

- 这件事是干嘛的：把 sync（做完全部菜才上下一桌）和 async（边做边上）两种执行器的延迟量出来，看参数怎么影响速度、过期丢弃什么时候触发。结论会告诉大家什么场景该用 async、延迟参数怎么估。
- 你要准备什么：C1 完成；读懂两个执行器的源码。注意服务对一条连接是串行处理的，慢推理会堵住后面的消息，所以压测一律单连接顺序发。
- 具体怎么做：
  1. sync 基线：`--inference-mode sync` 起服务，压测脚本 `--test`（默认 128 个请求）收延迟序列；再各扫一遍 `--denoise-steps {3,5,10}` 和 `--inference-horizon {null,8,16}`。
  2. async 实测：`--inference-mode async --inference-horizon 16`，扫 `--inference-delay-steps {2,4,8}`（默认 null，等于 horizon 的一半），观察后台预取命中和过期前缀丢弃。
  3. 极端情形：故意把 `--inference-delay-steps` 设得远小于真实延迟，统计过期丢弃的频率和报错（可能直接崩，见"容易踩的坑"）。
  4. 汇总 p50/p95/吞吐曲线，画一张"延迟估计差多少，对丢弃次数影响多大"的敏感性图。
- 做完交出什么：延迟曲线图 + 参数敏感性报告（每组至少 128 个请求；机器、存档、降噪步数固定），外加执行器选型结论。大概要 2–3 天。

### 任务 C5：验证"加速模式算出来的结果和正常模式一模一样"

- 这件事是干嘛的：async 降噪排法在默认参数（`variance_shift_alpha=1`、`linear_offset=0`）下，理论上和 sync 排法完全等价。这个任务用实验把"等价"证出来，再扫参数看它什么时候开始偏离、偏离长什么样。
- 你要准备什么：读降噪时间表的源码；端到端那半需要 C1 的环境。
- 具体怎么做：
  1. 纯函数层（不需要显卡，前两周可做）：构造两个假调度器，分别要 sync 和 async（alpha=1、offset=0）的时间表，断言两条时间步序列逐位相等。代码里有一个现成的判断函数，干的就是"async 在默认参数下退化成 sync"这件事。
  2. 端到端层（要显卡）：同一个存档、同一个固定输入（固定随机图的 seed），分别用 `--denoise-mode sync` 和 `--denoise-mode async`（保持 alpha=1、offset=0 默认）各跑一次 predict，比较返回的动作数组，应逐位一致，或差别小到 bf16 数值精度允许的误差之内。
  3. 扫参：`--variance-shift-alpha {1.0,1.5,2.0}` × `--linear-offset {0,0.2,0.4}` × `--lead-modality {video,action}`，记录动作轨迹的偏离量（范数差）并画出来。
- 做完交出什么：等价性测试（GPU 前后数值对照）+ 扫参轨迹可视化图，外加一页结论："async 何时等于 sync、何时开始偏离、偏离长什么样"。大概要 2 天。

### 任务 C6：逐个试开加速开关，量出各值多少钱

- 这件事是干嘛的：四个优化开关（DiT 缓存、跳过 VAE 解码、prompt 嵌入缓存、torch.compile）每个能省多少时间、会不会改变动作输出，目前没人系统量过。这个任务逐项开关，对比延迟和动作差异，最后给出"不同显卡和存档下该开什么"的推荐配置单。开关必须只提速、不改动作（差别不能超过数值精度允许的误差）。
- 你要准备什么：C1 和 C4 的测量流程；读引擎里装优化开关的那段代码。
- 具体怎么做：
  1. 固定存档、输入、请求数（≥128），以"全部默认开"为基线，逐项翻转：`--compile-enabled false`；`optimization.dit_cache.enabled=false`；`optimization.prompt_embed_cache.enabled=false`；`optimization.decode_video=true`（观察 VAE 解码占多少时间）。后三项直接在启动命令尾部追加这类"点分键=值"的覆盖即可。
  2. 每项记录 p50/p95，以及"和基线动作的最大绝对差"。
  3. 互斥验证：把 `cfg_scale` 调到大于 1（仅 Cosmos 系），模型代码会直接关掉 DiT 缓存而且不给任何告警，确认它静默失效。如果组里只有 Wan 存档，就做代码路径核对，并在报告里标注"Wan 推理不做 CFG，这条互斥只影响 Cosmos"。
- 做完交出什么：开关对比表（开关 × p50/p95 × 动作差异，每次只改一个开关）+ CFG 互斥验证记录（含证据出处），外加部署推荐配置单。大概要 2 天。

### 任务 C7：用 Docker 把服务打包带走

- 这件事是干嘛的：走通"构建镜像、compose 起服务、健康检查、导出离线包、到目标机校验装载"的全链路，产出一份离线交付 SOP。交付对象可能是一台完全不能联网的机器，所以最后要演练"离线搬运"。导出和校验不需要显卡，只有最后在目标机起服务推理才需要。
- 你要准备什么：Docker 三件套；构建镜像要联网和 ≥60GB 磁盘。阅读半（compose.yaml 加 docker/ 目录五个文件）前两周可做。
- 具体怎么做（命令照抄。注意离线包**不含权重和数据**：存档要在联网机器上用下载器落盘，随离线包一起搬运，并在目标机 `.env` 里设 `OPENWAM_CHECKPOINT_DIR`）：

```bash
# 1) 构建镜像（digest 钉死 CUDA 基底 + uv 锁；首次最慢）
cp docker/.env.example .env   # 设 OPENWAM_UID/GID 为 id -u / id -g
make docker-build && make docker-check
# 2) 起 serve（compose 里的 serve 服务；存档经 OPENWAM_CHECKPOINT_DIR 以只读方式挂进容器）
docker compose up -d serve
docker compose ps             # 等 healthy（healthcheck 用 docker/healthcheck.py，start_period 10m 容忍 compile 预热）
python docker/healthcheck.py --timeout 120    # 容器外也可直连 ws://127.0.0.1:8848
# 3) 离线 bundle 导出（schema-4：image.tar.gz + CONFIG_FILES + manifest.json；无需 GPU）
make docker-export PYTHON=python3
# 4) 目标机（可完全离线）校验与装载
python3 docker/offline.py verify dist/openwam-offline   # SHA256 + image identity 比对
python3 docker/offline.py load   dist/openwam-offline   # docker image load 后二次 identity 比对
# 5) 目标机起 serve 并用 C1 的客户端冒烟（此步才需要 GPU）
docker compose up -d serve && python docker/healthcheck.py
```

- 做完交出什么：构建日志 + 校验和装载通过记录 + 目标机服务 healthy + 单次 predict 成功证据 + 各环节耗时表，合成一份离线交付 SOP（要写清"离线包不含什么"）。大概要 1.5–2 天（镜像构建和下载时间另计）。
- 推进顺序一句话：前两周做 C2 骨架、C3 和 C7 的阅读半、C5 的纯函数半；拿到显卡后先做 C1，再补 C2 实验列，然后 C4 和 C5 共用测量基建，接着 C6 和 C3 实测半穿插，最后 C7 和 GPU 实验错峰。

## 容易踩的坑

- **`--state-dim` 三处文档对不上**（README 写 10，docker 文档写 20，脚本默认 20），而且服务端对 unify 存档故意不校验宽度，传错不会立刻报错，而是下游才炸。一律以存档里 `config.yaml` 的 `model.architecture.state_dim` 为准。
- **本机设了代理环境变量会把本机连接送走**：HTTP_PROXY 这类变量会让 `ws://127.0.0.1` 的流量进代理，连不上本机服务。自己写直连脚本时禁用代理，或者 `unset HTTP_PROXY HTTPS_PROXY ALL_PROXY`。
- **开了 torch.compile 时第一个请求慢到分钟级**，这是编译预热，属预期。量延迟时丢掉第一个请求，并在统计口径里注明。
- **compose 起服务不会自动建存档目录**：`OPENWAM_CHECKPOINT_DIR` 指向不存在的路径时容器启动即失败，报错还不直观。动手前先 `ls $OPENWAM_CHECKPOINT_DIR/config.yaml` 确认。

## 代码地图

```
openwam/deploy/                     # 本方向主战场：把存档变成服务的全部代码
├── server.py                       # 服务入口：六种消息的约定、收发循环、启动参数都在这
├── obs_preprocess.py               # 检查客户端发来的照片和指令，缺相机补黑图、拼多视角大图
├── policy.py                       # 所有请求的必经门面，往里调引擎
├── engine.py                       # 推理引擎：真正算动作的地方，四个加速开关在这装
├── denoise_schedule.py             # 排"降噪时间表"：决定每一步先算视频还是先算动作
├── model_loader.py                 # 读存档：按配置说明书重建模型、挑最新权重、恢复归一化器
├── executors/
│   ├── sync_executor.py            # 同步执行器：一批动作用完才算下一批
│   └── async_executor.py           # 异步执行器：后台提前算下一批，过期的头几步扔掉
└── optimizations/
    └── dit_cache.py                # 相邻两步输出很像就复用结果的缓存
configs/deploy.yaml                 # 部署配置文件，所有开关都在这（54 行）
scripts/deploy.sh                   # 启动脚本；多卡时一卡起一个服务
scripts/deploy.py                   # 源码树里的启动器
scripts/inference_test/
├── inference_single_test.py        # 参考客户端：发一个测试请求
├── inference_continuous_test.py    # 压测客户端：默认连发 128 个请求
benchmarks/utils/
├── transport.py                    # benchmark 客户端抄的那份消息约定镜像
├── client.py                       # 客户端打包工具：缩放图片、编码、组消息
compose.yaml                        # Docker 服务编排
docker/entrypoint.sh                # 容器入口，负责转到启动脚本
docker/healthcheck.py               # 健康检查：只发"在吗"，不加载模型
docker/offline.py                   # 离线打包：导出、校验、装载
docker/docker.mk                    # 离线导出的 make 命令定义
```
