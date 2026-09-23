# 方向 D：搭 8 个仿真考场，复现论文成功率

> 这个方向像组织考试：模型是考生，8 个仿真 benchmark（评测集）是 8 个考场。
> 考生住在服务里（方向 C 搭的），考场住在另一个环境里，两边用网络对话。
> 我们要把考场挨个搭起来，按论文的评分标准把成功率复现出来。前两周只需要读代码写文档，不需要显卡、不需要装环境。

> **这个方向的最终目标**：学完后，给你任何一个仿真考场，你都能独立搭好评测——装环境、接上服务、跑出成功率，并读懂数字背后的含义。
> **分阶段路线**：① 读代码，画出"考试怎么进行"的总图（D1）→ ② 先考三个容易的考场（D2）→ ③ 啃硬骨头考场、补 Labtasker 编排（D3–D6）。

## 你需要准备什么

前两周（任务 D1）什么都不用准备。D2 起真跑评测才需要：

- **方向 C 搭好的模型服务**：考场连上它问"下一步怎么动"。这是唯一硬依赖；服务怎么起不用你管，但你得会用。
- **至少 2 张显卡**：渲染和推理要分到不同卡上，不然考场会崩（见"容易踩的坑"第 1 条）。
- **官方模型存档 + 磁盘**：存档每个约 24.8GB，运行 `python scripts/download_assets/download_openwam_checkpoints.py` 下载。D3 的 RoboTwin 数据还要 415GB，D5 的 EBench 数据要 305GB 外加 34GB 素材——先确认磁盘，下载和装环境并行启动。

## 先搞懂这几件事

### 1. 考场只负责拍照和传话，计算全在服务那边

评测是两个进程、两个独立环境。考场这边只装仿真器和几百行适配代码，负责拍相机照片、传话；模型和预处理全在服务那边。两边只靠网络对话，考场连 torch 都不用装（依赖只有 numpy、Pillow、websockets）。所以每个考场一个独立 conda 环境，互不污染，也绝不装进服务的环境。

### 2. 网络对话有固定格式，开考前还要"对暗号"

考场和服务用 WebSocket（一种长连接网络协议）通信，消息格式在两端各写死一份、完全一致，服务端注释明说它是"冻结契约，永远不许改"。考场每走一步：发一条观测消息（base64 编码的相机照片 + 题目文字逐字转发 + 可选的机器人状态），收一条动作消息（动作指令 + 步数 + 耗时）。每考一题前必须发一次重置；不发的话，上一题没用完的动作缓存会漏到下一题。网络断了会自动重连一次。

另外，服务加载存档时会从训练配置里提取一份"表示契约"，核心是一个记录训练时动作格式的字段。考场一上来先打招呼，服务回的话里带这份契约；考场硬校验，对不上直接报错。作用是：拿 RoboTwin 的存档去考 LIBERO，第一步就被拒绝，而不是悄悄产出一堆垃圾成功率。还有一个"翻译官"：每种机器人的动作语言不一样（维度和单位都不同），靠一个七百多行的纯 numpy 文件统一换算，考场可以单独用它。注意：服务返回的动作已经是物理单位（机械臂末端用米、关节用弧度），考场不要再乘均值方差。

### 3. Labtasker：分布式考试管理系统

Labtasker 管"多台机器、多个工人同时考"时怎么不出乱子，目前只有 RoboTwin 和 LIBERO 两个考场接入了它。它做三件事：把考题列表和随机种子封存起来、盖上哈希指纹，封了就不许改，换存档重考也复用同一份；让每个工人独占一个服务，靠令牌防串台；让同时干活的工人各用各的随机数，互不干扰。

八个考场难易差很多，先用这张表建立印象：

| 考场 | 用的什么仿真器 | 多少题 | 环境好不好装 | 考多久 |
|---|---|---|---|---|
| LIBERO | MuJoCo | 4 组 × 10 题，每题考 50 次 | 最好装：一键脚本，素材随仓库自带（约 1.9GB） | 1–2 天 |
| LIBERO-plus | MuJoCo | 7 类扰动，每题只考 1 次 | 一键脚本 + 额外下载数 GB 素材 | 约 1 天 |
| RoboCasa365 | MuJoCo | 50 题（18 道小题 + 32 道组合题） | 一键脚本，素材约 10GB | 1–2 天 |
| RoboTwin | SAPIEN | 50 题 × 100 次，两种模式 | 最难装：一堆版本必须钉死 | 2–4 天 |
| VLABench | MuJoCo | 6 个赛道 | 没有脚本，手动装，必须钉版本 | 2–3 天 |
| RoboCasa GR1 | MuJoCo | 24 个环境 | 没有脚本，手动装 | 1–2 天 |
| EBench | Isaac Sim | test_mini 共 510 题 | 最难，还挑显卡（Blackwell 不行） | 2–3 天 |
| RoboDojo | 外部平台 XPolicyLab | — | 本方向不覆盖 | — |

## 任务清单

### 任务 D1：画出"考试是怎么进行的"全流程图（纯阅读）

- **这件事是干嘛的**：把上面这几件事画成一份总图文档——数据怎么流（谁发给谁、什么格式）、两端代码怎么对照、"改了某个东西会在哪一步爆掉"。这是全组最早能出成果的任务，也是后面所有任务的地图。
- **你要准备什么**：无，零环境。
- **具体怎么做**：
  1. 通读 `benchmarks/README.md` 全文：网络对话格式、相机字段规则、重置的时机、错误码表。
  2. 画两张对照表：一张把考场端和服务端的消息常量逐字段对上；一张追"对暗号"的链路，从服务提取契约到考场校验。
  3. 给翻译官画一张"机器人 × 换算函数"矩阵，标出每种的输入输出维度和单位。
  4. 以 `benchmarks/libero/LABTASKER.md` 为线索，把 Labtasker 的流程（封存清单 → 提交 → 工人独占服务 → 汇总）画成时序图。
  5. 把 `benchmarks/libero/openwam2libero_interface.py`（227 行）逐行注解，它是一个完整考场适配器的最小标本。
- **做完交出什么**：`评测管线总图.md`（放本任务书目录）+ LIBERO 接口逐行注解。验收标准：找一个没读过代码的同学，按图能复述"一张照片从相机到 env.step 的完整旅程"。
- **大概要多久**：3–4 天。

### 任务 D2：先考三个容易的——LIBERO、LIBERO-plus、RoboCasa365

- **这件事是干嘛的**：这三个考场用同一种仿真器、都有一键脚本，是最好的练兵场。目标数字：LIBERO Avg 99.3、LIBERO-plus Avg 69.2、RoboCasa365 Avg 38.2。
- **你要准备什么**：方向 C 的服务可用；联网；至少 2 张显卡最舒服。注意 LIBERO-plus 和 LIBERO 共用同一个存档，没有 plus 专用存档。
- **具体怎么做**：
  1. 装 LIBERO 环境（钉死 commit `8f1084e3132a39270c3a13ebe37270a43ece2a01`，Python 3.10 + mujoco 3.3.2 + robosuite 1.4.0，脚本会自动校验版本）：
     ```bash
     CONDA_BIN=/path/to/miniconda3/bin/conda \
     LIBERO_ENV_PREFIX=/path/to/miniconda3/envs/libero \
     LIBERO_PATH=/path/to/LIBERO \
     bash benchmarks/libero/setup_env.sh
     ```
  2. 自检（不需要模型）：`LIBERO_PATH=/path/to/LIBERO LIBERO_PYTHON=/path/to/miniconda3/envs/libero/bin/python bash benchmarks/libero/run_smoke.sh env`。然后下载存档：`python scripts/download_assets/download_openwam_checkpoints.py`，菜单选 `OpenWAM_Alpha → OpenWAM-Alpha-Sim-LIBERO`。
  3. 考 LIBERO 全量。托管脚本会自己起服务，启动顺序永远是先服务后考场；先加 `--smoke` 跑一题端到端验证，再跑全量：
     ```bash
     SERVER_PYTHON=/path/to/miniconda3/envs/openwam/bin/python \
     LIBERO_PYTHON=/path/to/miniconda3/envs/libero/bin/python \
     LIBERO_PATH=/path/to/LIBERO \
     GPUS=0,1 REPLICAS_PER_GPU=1 \
     bash benchmarks/libero/run_eval.sh \
       assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-LIBERO checkpoint_step_10690.safetensors \
       --compile-enabled false --render-gpus 2,3
     ```
     评分协议由代码钉死：每题考 50 次、seed 42、步数上限 600（`libero_10` 是 700）。
  4. 考 LIBERO-plus。环境钉死 commit `4976dc30028e805ff8094b55501d532c48fec182`，还要从 HuggingFace 的 `Sylvest/LIBERO-plus` 下载数 GB 素材包并做 SHA-256 校验。它的协议不一样：每题只考 1 次、seed 10000、`rng_mode: official_global`，启动时强制校验。复用第 3 步的命令模板，把环境变量换成 `LIBERO_PLUS_*`、脚本换成 `benchmarks/libero-plus/run_eval.sh`，存档不变。它的 `summary.csv` 自带按扰动类别拆分的结果。
  5. 装并考 RoboCasa365（钉死 robocasa `a07e365c958c4216cd6bbd5f30b47f09a65c6f00` v1.0.1 + robosuite `5ce6643f3092639d08f7b0f90ed1c6a84f50552c`，Python 3.11 + mujoco 3.3.1）：
     ```bash
     CONDA_BIN=/path/to/miniconda3/bin/conda \
     ROBOCASA365_ENV_PREFIX=/path/to/miniconda3/envs/robocasa365 \
     ROBOCASA365_PATH=/path/to/robocasa ROBOSUITE_PATH=/path/to/robosuite \
     bash benchmarks/robocasa365/setup_env.sh
     ```
     自检：`ROBOCASA365_PYTHON=... bash benchmarks/robocasa365/run_smoke.sh env OpenDrawer`。下载 `OpenWAM-Alpha-Sim-RoboCasa365` 存档。默认配置是冒烟规模（每题 5 次），正式考要复制一份配置改成 `num_trials: 50`，用 `ROBOCASA365_POLICY_CONFIG=/path/to/custom.yml` 指过去，`max_steps_override` 保持 `null`。全量 50 题：
     ```bash
     ROBOCASA365_PYTHON=/path/to/miniconda3/envs/robocasa365/bin/python \
     ROBOCASA365_POLICY_CONFIG=/path/to/custom.yml \
     bash benchmarks/robocasa365/multi_eval.sh --out ./results_robocasa365 target
     ```
     其中 `target` 是任务列表记号，会展开官方的 50 题清单。
- **做完交出什么**：三份成功率对照表 + 偏差分析 + 环境记录（驱动版本、是否用了 `--render-gpus` 和 `--compile-enabled false`）。验收：LIBERO 4 组 × 10 题 × 50 次跑完且对上论文；LIBERO-plus 7 类扰动矩阵完整，Camera/Robot/Noise 三个弱项单独讨论；RoboCasa365 的 `summary_pretrain.csv` 对上论文 Atomic/Comp.-Seen/Comp.-Unseen 三列。
- **大概要多久**：LIBERO 装环境 0.5 天 + 评测 1–2 天；LIBERO-plus 装环境 0.5 天 + 评测 1 天；RoboCasa365 装环境 0.5 天 + 评测 1–2 天。

### 任务 D3：考 RoboTwin（环境最难装的一个）

- **这件事是干嘛的**：复现论文 RoboTwin2.0-Full 数字（OpenWAM-α Clean 93.74 / Randomized 93.46）和 Clean2Random 泛化数字，全程留下"这次考试用了什么版本"的记录。
- **你要准备什么**：方向 C；415GB RoboTwin2.0 数据（下载和装环境并行启动）；驱动和 CUDA 组合能跑 SAPIEN 3.0.0b1 + cuRobo。
- **具体怎么做**：
  1. 按官方版本表搭独立环境：RoboTwin 钉死 commit `0aeea2d669c0f8516f4d5785f0aa33ba812c14b4`、Python 3.10、SAPIEN 3.0.0b1、cuRobo v0.7.8、warp-lang 1.13.0、mplib 0.2.1、torch 2.4.1，再 `pip install websockets pyyaml`。不要把这套装进 OpenWAM 的环境。
  2. 下载 `OpenWAM-Alpha-Sim-RoboTwin-Full` 存档，`bash scripts/deploy.sh <ckpt_dir>` 起服务（先服务后考场，考场健康检查最多等 300 秒）。
  3. 对齐配置和存档：存档是 ee 格式就配 `action_type: ee` + `state_dim: 20`，是 qpos 格式就配 `qpos` + `14`。
  4. 冒烟：`ROBOTWIN_TEST_NUM=5 ROBOTWIN_PATH=/path/to/RoboTwin ROBOTWIN_PYTHON=... bash benchmarks/robotwin/single_eval.sh adjust_bottle demo_clean openwam 0`。日志里出现"预热 CUDA""打了 warp.torch 兼容补丁""RoboTwin commit=…"三行都是预期输出，不是报错。
  5. 先单题考 100 次（去掉 `ROBOTWIN_TEST_NUM` 重跑同一题），再考全 50 题：`bash benchmarks/robotwin/multi_eval.sh -m demo_clean -n run1 -d <ckpt_dir> all`，然后把 `-m` 换成 `demo_randomized` 再跑一遍。结果在 RoboTwin 目录的 `eval_result/`，每题日志和环境记录另存在 `<ckpt_dir>/robotwin_eval_logs/`。
- **做完交出什么**：RoboTwin 复现报告（对照论文两张表）+ 环境搭建 SOP。验收：两种模式全 50 题 × 100 次，成功率日志和版本记录齐全。
- **大概要多久**：装环境 1–2 天 + 评测 2–4 天。

### 任务 D4：考 VLABench 和 RoboCasa GR1

- **这件事是干嘛的**：VLABench 分赛道复现三项指标（成功率、进度分、意图分），对照论文 Avg SR 71.6；RoboCasa GR1 考 24 个 `gr1_unified/*` 环境，对照论文 SR 60.5。
- **你要准备什么**：方向 C。VLABench 要 17GB 素材 + 13.2GB 数据，且必须钉死 commit（上游最新代码有题跑不起来）。GR1 要 EGL 可用，考试只需桌面素材。
- **具体怎么做**：
  1. 手动装 VLABench 环境（没有一键脚本，钉死 `numpy==1.25.0`、`mujoco==3.2.2`、`dm_control==1.0.22`；无显示器机器再装 conda-forge 的 `libegl mesalib libglvnd` 和 CPU 版 torch）。装完自检：`VLABENCH_PATH=/path/to/VLABench VLABENCH_PYTHON=... bash benchmarks/vlabench/run_smoke.sh env select_fruit`。
  2. 下载 `OpenWAM-Alpha-Sim-VLABench` 存档，起服务，逐赛道考：`bash benchmarks/vlabench/single_eval.sh all track_1_in_distribution 50 8848`。赛道从 `track_1_in_distribution` 到 `track_6_unseen_texture`，每题预算 30–60 分钟。
  3. 两个注意：赛道 5 上游不发布考题配置，只能给脚本显式题目名、或用 `multi_eval.sh` 内置的 8 道保留题，报告里必须注明这个协议差异。多卡加速是每个端口起一个服务（`for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$i nohup bash scripts/deploy.sh <ckpt> --device cuda:0 --port $((8880+i)) & done`），再 `bash benchmarks/vlabench/multi_eval.sh all all 50 8880,8881,8882,8883`；注意"请求考几题"不等于"实际记录几题"，引用分数前先查日志。
  4. 手动装 GR1 环境。关键坑：装完后要先卸载再钉版本：
     ```bash
     conda create -c conda-forge -n robocasa-gr1 python=3.10 -y && conda activate robocasa-gr1
     git clone https://github.com/robocasa/robocasa-gr1-tabletop-tasks /path/to/robocasa-gr1-tabletop-tasks
     cd /path/to/robocasa-gr1-tabletop-tasks && pip install -e .
     pip uninstall -y robosuite mujoco && pip install robosuite==1.5.1 mujoco==3.2.6
     pip install pyyaml numpy Pillow websockets
     python robocasa/scripts/download_tabletop_assets.py -y
     ```
  5. GR1 自检：`ROBOCASA_GR1_PATH=... ROBOCASA_GR1_PYTHON=... ROBOCASA_GR1_ENABLE_RENDER=1 bash benchmarks/robocasa_gr1/run_smoke.sh env`，成功会打印 `env_smoke=ok`；报 `'NoneType' object has no attribute 'eglQueryString'` 说明 EGL 没配好。
  6. 下载 `OpenWAM-Alpha-Sim-RoboCasa-GR1` 存档，起服务，逐环境考（一个服务串行处理 24 个环境）：
     ```bash
     ROBOCASA_GR1_PATH=... ROBOCASA_GR1_PYTHON=... ROBOCASA_GR1_GPU=0 \
     bash benchmarks/robocasa_gr1/single_eval.sh gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env 8848 127.0.0.1
     ```
     正式跑时复制一份 `policy_config.yml` 调 `num_episodes`，步数上限 720。
- **做完交出什么**：VLABench 分赛道三项指标表 + 赛道 5 协议说明，与论文逐格对比；GR1 24 个环境的考试记录，与论文对比；两个手动环境的搭建 SOP。
- **大概要多久**：VLABench 装环境 1 天 + 评测 2–3 天；GR1 装环境 1 天 + 评测 1–2 天。

### 任务 D5：考 EBench（先用假考场练手，硬件允许再考真的）

- **这件事是干嘛的**：先不装 Isaac Sim，用仓库自带的假考场验证这条链路的动作契约；硬件允许时再复现 test_mini 的 510 题（26 题 ×20 次，`make_sandwich` 和 `microwave` 各 15 次），对照论文 Overall SR 49.4 / Score 64.7。
- **你要准备什么**：方向 C 的服务。**完整评测前先查显卡型号**：Isaac Sim 4.1.0（CUDA 12.1）不支持 Blackwell 架构的显卡，Blackwell 机器上直接判定不可行并记录结论。数据量：305GB 数据 + 34GB 素材。
- **具体怎么做**：
  1. 下载一个数据集小分片（`huggingface-cli download InternRobotics/EBench-Dataset --repo-type dataset --local-dir assets/benchmark_data/ebench --include 'simple_pnp/task1/*'`），然后起假考场（用训练环境，需要 `pandas`/`pyarrow`/`av`）：
     ```bash
     python benchmarks/ebench/mock_genmanip_server.py \
       --dataset-dir assets/benchmark_data/ebench --bucket simple_pnp/task1 \
       --episodes 2 --steps-per-episode 8 --port 8087
     ```
  2. 另一个终端起服务：`bash scripts/deploy.sh assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-EBench`，然后跑单工人：`EBENCH_PYTHON=... bash benchmarks/ebench/single_eval.sh --url http://127.0.0.1:8087 --ckpt-config assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-EBench/config.yaml`。
  3. 完整评测：`nvidia-smi --query-gpu=name --format=csv` 核对显卡代际，不是 Blackwell 才继续。仿真机按 EBench 官方指南装 Isaac Sim 4.1.0 + cuRobo + 素材。考场环境（无 torch）：`git clone --recursive https://github.com/InternRobotics/EBench && cd EBench && pip install -e third_party/genmanip-client && pip install numpy Pillow "websockets>=15" PyYAML opencv-python-headless "PyTurboJPEG<2" filelock`。
  4. 仿真机起考试服务：`python ray_eval_server.py --host 0.0.0.0 --port 8087 --no_save_process`，提交 `gmp submit ebench/generalist/test_mini --run_id <run_id>`。同时按论文协议起模型服务：`NUM_GPUS=4 bash scripts/deploy.sh assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-EBench --compile-enabled false optimization.dit_cache.enabled=false`（dit_cache 必须关，否则数字和论文不可比）。
  5. 并行工人：`NUM_WORKERS=4 EBENCH_PYTHON=... bash benchmarks/ebench/multi_eval.sh --url http://<sim-host>:8087 --run-id <run_id> --ckpt-config <ckpt>/config.yaml`。结果在仿真侧的 `saved/eval_results/<task>/<run_id>/`，考场侧镜像到 `client_results/`。
- **做完交出什么**：假考场产出的 `ebench_mock_actions.jsonl` 和校验记录（日志要出现 `run complete: 2 episodes, 16 steps bridged` 和 `16 actions, 0 violations`，违例会直接报错退出）；test_mini 510 题按三类任务拆分的成绩对照；Blackwell 机器上的硬件结论记录。
- **大概要多久**：假考场 0.5–1 天；完整评测装环境 1–2 天 + 评测 2–3 天。

### 任务 D6：给还没用上 Labtasker 的考场补上

- **这件事是干嘛的**：目前只有 RoboTwin 和 LIBERO 接入了 Labtasker（分布式考试管理系统，见上面第 3 条）。给 robocasa365 和 libero-plus 各补一套：一篇 LABTASKER.md 加提交、工人、汇总三个脚本。
- **你要准备什么**：D2 的传统跑法已跑通；通读现有的两篇 LABTASKER.md。
- **具体怎么做**：
  1. 读 `benchmarks/utils/` 里 Labtasker 相关的四个文件，弄清封存清单、工人独占服务、随机数隔离怎么实现。然后以 LIBERO 的四个 Labtasker 脚本为模板，把每个考场不同的部分抽出来（题目枚举、环境构造、每题考几次、清单的缓存键），为 robocasa365 实现一套。
  2. libero-plus 照搬前注意：它每题只考 1 次、seed 10000、`rng_mode: official_global`，随机数语义和 LIBERO 不同，先确认两者不冲突。
  3. 写两篇新的 LABTASKER.md，结构照抄现有的：装环境 → 首次运行建清单缓存 → 跑评测 → 汇总。最后做对照实验：同一个存档，分别用传统跑法和 Labtasker 跑同一批题，核对成功率一致。
- **做完交出什么**：两篇新 LABTASKER.md + 可运行的提交和工人链路 + 一致性对照记录。代码提交到组内 fork，不要改动 OpenWAM 调研仓库。
- **大概要多久**：2–3 天/考场。

## 容易踩的坑

- **仿真器和 CUDA 驱动打架**：MuJoCo 系考场的离屏渲染和模型推理共用一张物理显卡时，考场会在截图时直接崩掉（退出码 exit=-6）。这是官方确认过的驱动级冲突，不是你的 bug。躲法：渲染和推理分到不同显卡，即 `GPUS=0,1 ... --render-gpus 2,3`；服务端一律加 `--compile-enabled false`；单机单卡可设 `MUJOCO_GL=osmesa` 走 CPU 渲染兜底（慢）。
- **RoboTwin 版本强耦合**：我们的包装脚本给上游代码打了补丁，和上游布局硬绑定，commit 和五个库的版本一个都不能换，换了要么崩要么悄悄算错。躲法：先 checkout 验证过的 commit `0aeea2d669c0f8516f4d5785f0aa33ba812c14b4`，再调试任何其他问题。
- **EBench 挑显卡**：Isaac Sim 4.1.0 不支持 Blackwell 架构的显卡，强装是浪费时间。躲法：完整评测前先 `nvidia-smi` 查型号，Blackwell 机器直接判定不可行并记录结论。
- **一个仿真器必须配一个服务**：两个仿真器共用一个服务端口会互相污染动作缓存，成功率悄悄变坏还不报错。躲法：多工人就用 `NUM_GPUS=N bash scripts/deploy.sh` 起 N 个服务一一对应。

## 代码地图

```
benchmarks/
├── README.md                     # 考场接入总说明书：两边怎么对话、怎么拍照、怎么重置，先看这篇
├── utils/                        # 所有考场共用的小工具箱
│   ├── transport.py              # 负责和服务那边发消息、收消息，断线自动重连
│   ├── action_conversion.py      # 翻译官：把各种机器人的动作语言互相换算
│   ├── eval_manifest.py          # 封存考题清单：封了就不许改，换存档重考也用同一份
│   ├── labtasker_utils.py        # 提交考题，提交前先检查有没有漏题
│   ├── task_policy.py            # 让每个工人独占一个服务，防止串台
│   └── rng_domain.py             # 让同时干活的工人各用各的随机数
├── libero/                       # LIBERO 考场：有一键装环境、全自动考试、自检三套脚本，练兵从这里开始
├── libero-plus/                  # LIBERO-plus 考场：专门考各种干扰（换相机、换灯光等）下的表现
├── robocasa365/                  # RoboCasa365 考场：官方 50 道题的清单也在这个目录里
├── robotwin/                     # RoboTwin 考场：最难装的一个，里面有给上游代码打补丁的包装脚本
├── vlabench/                     # VLABench 考场：没有一键脚本，要手动装
├── robocasa_gr1/                 # RoboCasa GR1 考场：也没有一键脚本
├── ebench/                       # EBench 考场：里面有个假考场脚本，不装重型仿真器也能练手
└── robodojo/                     # 只有一篇说明：这个考场走外部平台，本方向不管
openwam/deploy/                   # 服务那边的代码：加载存档、回消息。方向 C 的地盘，你只需会用
scripts/deploy.sh                 # 一键把存档启动成服务
scripts/download_assets/          # 官方存档和数据的下载脚本，都有交互菜单，跟着选就行
```
