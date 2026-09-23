# 方向 A：数据管线——把 13 种数据集读成同一种训练样本

> OpenWAM 要用十几种长得都不一样的机器人数据一起训练。这些数据格式不同、机器人类型不同、动作含义也不同。这个方向就是搞清楚：它们怎么被读进来、统一成一种格式、喂给训练。就像把各国语言翻译成同一种文字再进印刷厂。
> 前两周只需要读代码写文档，不需要显卡、不需要装环境。

> **这个方向的最终目标**：学完后，给你任何一个新数据集，你都能自己搭建数据管线——写出读取代码、接进统一格式、算好归一化统计，顺利喂给训练。
> **分阶段路线**：① 先读代码，画出"数据怎么流进训练"的全景（A1、A2、A7）→ ② 动手跑通归一化链路、亲手写一个最小数据集类型（A3–A5）→ ③ 攻坚预训练混合数据（A6）。

## 你需要准备什么

前两周只需要 OpenWAM 源码和一个普通 Python（CPU 就够）。后面要用到的东西：

| 什么东西 | 什么时候才需要 | 怎么获取 |
|---|---|---|
| LIBERO 数据集（1.9GB，最小的可下载数据集） | 任务 A3、A4 时 | `python scripts/download_assets/download_benchmark_data.py --name LIBERO --yes`（下载完会自动生成统计文件并写回配置） |
| 大容量存储（几百 GB）+ HuggingFace 数据授权 | 只有任务 A6 需要 | 第二周就去申请授权，等审批时做别的任务 |

这个方向不依赖别人：不用等模型权重，不用等部署环境，所有任务验收都不需要显卡。

## 先搞懂这几件事

**训练样本只有一种长相。** 13 种数据集格式千差万别，但每种数据集的读取代码最后必须吐出同一个样子的字典：一串视频帧 + 动作张量 + 每个动作维度"有没有效"的开关 + 一句文字指令。注意那个开关：动作值全为 0 并不会让模型不学这一段，只有开关关掉才不学。

**每种数据集先登记户口，再按名字使用。** 所有数据集类型用名字登记在一个总表里，一共 13 个名字：robotwin、robodojo、agibotworld、mixture、robocasa_gr1、robocoin、ebench、libero、muka_franka、oxe_droid、robocasa365、interndata_a1、vlabench。训练时配置里写 `type: libero`，代码就按名字查出对应的类。每种数据集必须自己提供"从配置创建"的方法，没有就直接报错，没有兜底。

**一个基类加几个空槽，撑起大部分读取代码。** 13 种数据集里有 8 种直接继承同一个基类。它们的差别不靠再建中间层，而是基类留好几个空方法（选哪些相机、统计文件放哪、原始动作怎么换算），每种数据集填自己的实现。配置文件里允许写哪些字段，由基类里一份白名单决定。

**统一动作空间是一块 80 孔的万能插座面板。** 不同机器人的动作维度不一样：单臂 10 维、双臂 20 维、还有带底盘和灵巧手的。OpenWAM 准备了一块固定的 80 维面板，每种数据集在配置里写清楚自己的动作插到哪几个孔（比如 `["0-9","34-43"]`），空着的孔用开关标记"没插"。插进去和拔出来（部署时还原成原始维度）完全可逆。正因为有这块面板，单臂的 LIBERO 和双臂的 AgileX 才能混在同一个批次里训练。

**归一化就是把不同的尺子换算成同一把。** 不同数据集的动作数值范围差别很大，直接混在一起训练会乱套，所以要先换算——就像把摄氏、华氏温度统一成一种再比较。支持三种算法；其中两种会把结果截断在 [-1,1]。有一类特殊维度表示旋转，天生有界，统计值被直接钉死成固定值，不走计算。训练时的归一化和部署时的反归一化走的是同一条链路。

**多个相机画面拼成一张图，多个数据源配比例混着喂。** 机器人有头部相机和两个腕部相机，模型只吃一张图，做法是像监控室拼大屏：384×320 的画布，上面一整块放头部画面，底部左右两个半块放腕部画面，缺的视角填黑。预训练要同时用好几种数据源，靠一个"调度员"数据集按权重轮流抽样，默认策略是按各源的数据量比例来。

## 任务清单

### 任务 A1：画出"数据怎么流进训练"的全流程图

- 这件事是干嘛的：写一份主线文档，把"一条数据从磁盘到训练批次"的每一步讲清楚。这是本方向后面所有任务的公共底座。
- 你要准备什么：读完上面"先搞懂这几件事"；读官方文档 `assets/openwam_usage_docs/benchmark-integration.md` 和 `train-and-deploy.md`。
- 具体怎么做：
  1. 沿着调用链把每一步记在笔记里，画出调用链图。链条是：训练入口脚本 → 配置拼装（`configs/train.yaml` 加上 `configs/dataloader/<名字>.yaml`）→ 按名字创建数据集 → 逐字段读配置 → 读取器初始化（读元信息、拼 episode、过滤、读统计、选相机）→ 取样本（取一段数据 → 归一化 → 撒进 80 维面板并生成开关 → 解码视频拼图）→ 打成批次。
  2. 用现成的测试假数据（在 `tests/dataloader/conftest.py`）打印一个真实样本的每个字段的形状和类型，对照官方文档的字段表逐字段写注释。
  3. 画一页示意图：80 维向量按臂/手/底盘分段，标出每段的开关什么时候开、什么时候关，要和代码实现逐维对上。
  4. 补一节"怎么新增一个数据集类型"的 10 行速览（细节留给 A5 实操）。
- 做完交出什么：全景指南 1 份（调用链图 + 字段契约表 + 80 维示意图），组内评审通过。
- 大概要多久：2–3 天。

### 任务 A2：整理 80 维动作空间对照表（为什么是 80 维、每个槽位放什么）

- 这件事是干嘛的：把 13 种数据集的"动作插到哪些孔"汇总成一张大表，标出冲突（同一个孔在不同数据集里含义不同）和夹爪开合方向的差异。这张表是后面所有跨数据集工作的字典。
- 你要准备什么：A1 完成；读通 `openwam/dataloader/utils/unify_action.py`。
- 具体怎么做：
  1. 把所有配置里的映射抠出来：`grep -n "unify_action_map\|unify_state_map" configs/dataloader/*.yaml configs/dataloader/pretrain_data/*.yaml`；
  2. 重点核对两个特殊案例：`configs/dataloader/robocasa365.yaml`（状态 19 维、动作 15 维，不对称）；夹爪方向——AgiBotWorld 原始数据"0 是张开、1 是合拢"，读取代码内部做了反转（见 `openwam/dataloader/agibotworld.py` 文件头注释）；VLABench 的状态和动作方向相反是上游数据的 bug，**故意保持原样**（评测客户端读同一个有 bug 的接口，实际跑的时候两次反转刚好抵消，见 `openwam/dataloader/vlabench.py` 文件头注释）；
  3. 写个小脚本，把每段"插孔说明"展开成 80 维的占位图，汇总成表。
- 做完交出什么：映射总表（数据集 × 孔位段 × 含义 × 方向）+ 跨数据集冲突清单；表的行为要和 `tests/dataloader/test_unify_action.py` 一致。
- 大概要多久：2–3 天。

### 任务 A3：亲手跑一遍归一化统计的生成过程

- 这件事是干嘛的：用最小的 LIBERO 数据集（1.9GB），把统计文件从删掉到重建完整跑一遍。顺便验证两件事：旋转维度的统计值确实被钉死；训练侧和部署侧的归一化结果一致。
- 你要准备什么：一台普通电脑 + 下载 LIBERO：`python scripts/download_assets/download_benchmark_data.py --name LIBERO --yes`（下载器会自动建统计文件并写回配置）。不需要显卡。
- 具体怎么做：
  1. 下载后删掉 `meta/libero_normalization_stats.npy`，手动重跑 `openwam/dataloader/utils/stats_computation/libero_stats_computation.py` 重建；
  2. 检查旋转维度的统计值是不是等于钉死的固定值（最小 -1、最大 1、均值 0、标准差 1）；
  3. 取同一个原始向量，分别走训练侧归一化和部署侧"归一化再反归一化"的往返，确认往返误差小于 1e-5；
  4. 读懂多卡训练的约定：统计文件不存在时，0 号卡负责生成，其他卡原地等待，等待上限由环境变量 `OPENWAM_STATS_WAIT_TIMEOUT_S` 控制（默认 12 小时）。
- 做完交出什么：统计链路验证记录（重建结果与下载器产物数值一致）+ 一个训练/部署一致性的冒烟脚本。
- 大概要多久：2–3 天（含下载）。

### 任务 A4：核对下载脚本、数据读取代码、配置文件三者对得上

- 这件事是干嘛的：下载脚本、读取代码、yaml 配置是三套各自维护的东西，要逐条对账，确认没有"配置了但下载不了""下载了但读不了"的窟窿。
- 你要准备什么：A1 的代码地图；1–2 个已下载数据集用来抽验（可复用 A3 的 LIBERO）。普通电脑即可。
- 具体怎么做：
  1. 列出下载脚本 `scripts/download_assets/download_benchmark_data.py` 里的 8 个下载条目（RoboTwin2.0 / RoboDojo / RoboDojo-Real / LIBERO / VLABench / EBench / RoboCasa365 / RoboCasa_GR1），每个记下 HuggingFace 仓库名、子路径、文件过滤规则、对应哪个 yaml，以及体量（从 LIBERO 1.9GB 到 RoboDojo 488GB）；
  2. 对照代码地图里的 13 种数据集，标出没有下载入口的 5 个：robocoin、agibotworld、interndata_a1、oxe_droid、muka_franka；
  3. 抽 1–2 个已下载的数据集，核对下载后的解压/挪位置和统计文件生成的产物落在哪、回写的 `dataset_dir` 字段和读取代码期望的是否一致。
- 做完交出什么：对应关系表（数据集 × 下载条目 × 读取代码 × 配置 × 统计步骤）+ 缺口报告。
- 大概要多久：1 天。

### 任务 A5：动手：用假数据写一个最小的新数据集类型并注册成功

- 这件事是干嘛的：这是本方向唯一的动手搭建任务。用合成数据实现并登记一个最小的新数据集类型，亲手把 A1 文档里那条链走一遍——走过一遍，比读十遍都管用。
- 你要准备什么：A1 完成（知道统一样本每个字段的含义）；普通 Python 环境。
- 具体怎么做：
  1. 造假数据：在临时目录写 2 个 episode 的最小 LeRobot v3 目录结构；或者走更轻的路——直接继承最基础的基类，照官方文档 `benchmark-integration.md` 里的模板抄；
  2. 实现"从配置创建 / 数据有多长 / 取第几条"三个方法，取样本时吐出全字段的统一样本（可以走"只有视频"路径：动作全零 + 开关全关。记住：全零的动作值本身不会关掉学习，开关才关）；
  3. 用注册装饰器把它登记为 `synthetic_demo`，写 `configs/dataloader/synthetic_demo.yaml`（`type: synthetic_demo`、`num_frames: 33`、`multiview: false`、`normalize_mode: null`）；
  4. 验证能取出一个样本：
     ```bash
     cd OpenWAM && python -c "
     from omegaconf import OmegaConf
     from openwam.dataloader.registry import build_dataset
     cfg = OmegaConf.load('configs/dataloader/synthetic_demo.yaml')
     ds = build_dataset(cfg, split='train')
     s = ds[0]
     print(len(ds), s['action'].shape, s['action'].dtype, s['action_mask'].sum())"
     ```
  5.（进阶）改成继承那个通用基类并配上 80 维插孔说明，验证动作插进面板再拔出来能还原。
- 做完交出什么：可运行的最小示例（放组内共享目录或 playground 分支，**不进主仓库**）+ 一页踩坑记录。样本字段要和官方文档的契约表逐条对上。
- 大概要多久：2–3 天。

### 任务 A6：调查预训练混合数据能不能复现（机制层面）

- 这件事是干嘛的：预训练用的 4 个数据源（AgiBotWorld、RoboCOIN、InternData-A1、OXE-DROID）没有官方下载脚本。这个任务要把它们真正弄到手，在本组存储上跑通混合数据集的一步，并把混合采样的机制验证清楚。这里只回答"数据能不能进来、按什么权重采样"；实际训练配方归方向 F（[模型解剖](方向F-模型解剖.md)）。
- 你要准备什么：A4 完成；大容量存储（四源合计数百 GB）。第二周就去发起 HuggingFace 授权申请，等待期间做别的。
- 具体怎么做：
  1. 申请/下载四个数据集。DROID 还要按 `openwam/dataloader/oxe_droid.py` 的要求准备 `meta/excluded_episodes.json`；InternData-A1 用环境变量 `INTERNDATA_A1_ROOT` 指路；
  2. 把 `configs/dataloader/pretrain_data/` 下各 yaml 里的 `/path/to/pretrain_dataset/...` 占位符改成本组实际路径；
  3. 逐源单测：`python -m pytest -q tests/dataloader/test_robocoin_trim.py tests/dataloader/test_interndata_a1.py`（真实数据没到位时先用假数据验证逻辑）；
  4. 机制验证：分别用四种权重策略构建混合数据集，打印各子源的实际采样比例，和 `openwam/dataloader/mixture.py` 的实现对照；
  5. 最小冒烟（CPU 即可）：
     ```bash
     cd OpenWAM && python -c "
     from openwam.dataloader.registry import build_dataset
     from omegaconf import OmegaConf
     cfg = OmegaConf.load('configs/dataloader/pretrain_data/mixture.yaml')
     ds = build_dataset(cfg, split='train')
     print(len(ds), ds[0]['action'].shape, ds[0]['action_mask'].sum())"
     ```
  6. 核算各源的帧数/小时数（配置里的 `total_hours` 旋钮），估算和论文 518.5M 帧的差距，结论交给方向 F（模型解剖）用。
- 做完交出什么：可运行的混合配置 + 一步的真实样本输出 + 四种权重策略的验证记录 + 体量核算表。跑不起来的源在报告里写明卡在哪（授权/体积/缺文件）。
- 大概要多久：1–2 周（主要耗在数据申请与传输）。

### 任务 A7：给人视角视频数据缺失这件事下结论

- 这件事是干嘛的：官方没发布"人视角视频"（EgoDex/Ego4D）数据的读取代码，这是全仓库最大的复现缺口。这个任务要确认证据链，并评估自己补一个要多少功夫。这里只出"是否存在、要不要补"的结论；如果决定补，实现归方向 F（模型解剖）。
- 你要准备什么：A2 的槽位理解。
- 具体怎么做：
  1. 复核并记录下面 5 条证据（已核验）：
     - `openwam/dataloader/bases/lerobot_v3_reader.py` 文件头注释宣称 EgoDex 和 OXE、RoboCOIN 一样，都是通用基类的直接子类；
     - `openwam/dataloader/utils/lerobotv3.py` 文件头注释说 RoboCOIN 和 EgoDex 遵循同一套接入流程，`openwam/dataloader/bases/multi_lerobot_v3_reader.py` 的注释也提到了 EgoDex；
     - 但注册表里登记的 13 个名字没有 egodex/ego4d；
     - 全仓库搜索 EgoDex/Ego4D，找不到任何类定义——代码根本没发布；
     - `tests/dataloader/test_oxe_mixture_integration.py` 明确把 ego4d、egodex、oxe_bcz、oxe_bridge、oxe_fractal 列为"已废弃"，并断言它们没注册。
  2. 以 `openwam/dataloader/robocoin.py` 为模板起草实现方案：要填哪些空方法（选相机、动作换算、统计文件路径）、EgoDex 原始格式怎么映射到 80 维面板、统计文件要什么；
  3. 给出工作量评估（要填几个空方法、测试假数据怎么造、风险点），形成"做/不做"的建议。
- 做完交出什么：结论备忘录（含完整证据链）+ 实现草案；若决定实现，移交方向 F（模型解剖）立项。
- 大概要多久：2–4 天（只做定位和评估）。

推进顺序一句话：第 1 周做 A1、A4、A7（纯阅读），第 2 周做 A2、A5 并顺手发起 A6 的数据申请，第 3 周做 A3，A6 等数据到位就推进。

## 容易踩的坑

- **测试假数据会缓存，改了代码不自动更新**：旧缓存会导致莫名其妙的假失败。躲法：改了读取代码后先 `rm -rf tests/dataloader/.cache` 再跑测试。
- **改了裁剪/排除规则后必须重算统计文件**：旧统计文件带着数据规模的校验信息，加载时会直接报错。躲法：任何裁剪/排除/分片变更后，删掉旧统计文件重建。
- **VLABench 的方向 bug 是故意保留的**：状态和动作方向相反是上游数据的 bug，评测客户端读同一个接口，实际跑的时候刚好抵消。躲法：绝不在读取代码里"好心修正"，否则训练和评测对不上。
- **多卡训练时统计文件缺失会假死**：其他卡会原地等最长 12 小时。躲法：先单卡跑通生成统计文件再上多卡；必要时把环境变量 `OPENWAM_STATS_WAIT_TIMEOUT_S` 调小，让它快速报错。

## 代码地图

```
openwam/dataloader/
├── registry.py                 # 所有数据集都在这里登记户口，想加新的先来这看
├── bases/
│   ├── dataset.py              # 最基础的基类，规定每种数据集必须吐出什么样的一条数据
│   ├── lerobot_v3_reader.py    # 核心基类，8 种数据集都靠它，留了几个空方法给子类填
│   └── multi_lerobot_v3_reader.py  # 把多个同类数据桶合在一起读的基类
├── transforms/                 # 数据出锅前的加工间：归一化、多相机拼图、视频解码、旋转换算
├── utils/
│   ├── unify_action.py         # 80 孔万能插座面板的全部实现
│   ├── normalization.py        # 归一化的计算和统计文件的生成
│   ├── lerobotv3.py            # 各数据集共用的小工具，数据对不上时在这里报错
│   ├── stats_computation/      # 12 个脚本，一个数据集一个，专门生成归一化统计文件
│   └── 其余小工具              # 末端执行器、位姿、数据模式、排除清单、视频读写
└── 13 个数据集文件（见下表）
```

13 种数据集一览（配置文件在 `configs/dataloader/`，读取代码在 `openwam/dataloader/`）：

| 数据集 | 什么机器人/什么格式 | 动作怎么放上 80 孔面板 | 好不好下载 |
|---|---|---|---|
| libero | 单臂仿真，LeRobot v3 格式 | 原生 10 维 → 孔 [0-9] | 下载脚本一键 |
| vlabench | 单臂仿真，LeRobot v3 | 7 维补成 10 维 → [0-9] | 下载脚本一键 |
| robocasa365 | 单臂仿真，LeRobot v3 | 状态和动作不对称（19/15 维）：动作 → [0-9],[68-72] | 下载脚本一键 |
| robocasa_gr1 | 人形机器人，LeRobot v3 | 33 维 → [0-8],[10-15],[34-42],[44-49],[68-70] | 下载脚本一键 |
| robotwin | 双臂仿真，HDF5 文件 | 关节 14 维或末端 20 维 → [0-9],[34-43] | 下载脚本一键 |
| robodojo | 双臂，HDF5 | 末端 20 维 → [0-9],[34-43] | 下载脚本一键 |
| ebench | 带底盘双臂，LeRobot v2.1 | 23 维（含底盘 3 维）→ [0-9],[34-43],[68-70] | 下载脚本一键 |
| robocoin | 多种真机，LeRobot v3 | 末端 20 维或灵巧手 → [0-9],[34-43] | 要手工下载 |
| agibotworld | 双臂真机，LeRobot v3 | 末端 20 维（+移动 2 维）或灵巧手 30 维 → [0-9],[34-43],[68/70] | 要手工下载 |
| interndata_a1 | 单臂真机，LeRobot v3 | 末端 20 维 → [0-9],[34-43] | 手工，靠环境变量 `INTERNDATA_A1_ROOT` 指路 |
| oxe_droid | 单臂真机，LeRobot v3 | 单臂 10 维 → [0-9]（右半关掉） | 手工，还要一个排除清单文件 |
| muka_franka | 单臂真机，LeRobot v3（15fps） | 7 维补成 10 维 → [0-9] | 手工（HuggingFace 小仓库） |
| mixture | 不是数据，是调度员 | 继承各子源（强制各源维度、帧数一致） | — |

其他要看的文件：

| 文件 | 它是干什么的 |
|---|---|
| `scripts/train.py` | 训练入口，数据集就是在这里被创建的 |
| `scripts/download_assets/download_benchmark_data.py` | 数据集下载脚本，存着 8 个一键下载的清单 |
| `assets/openwam_usage_docs/benchmark-integration.md` | 官方"怎么接新数据集"文档，A5 的模板来源 |
| `assets/openwam_usage_docs/train-and-deploy.md` | 官方训练部署文档，A1 的前置阅读 |
