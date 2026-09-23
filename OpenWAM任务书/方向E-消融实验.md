# 方向 E：消融实验——怎么做一次可信的对比

> 消融实验就是"做实验的规矩"：想知道某个设计有没有用，就每次只改它一个，其他全部保持不变，看结果差多少。就像试菜谱——想知道盐放多少好，就一次只改盐，火候和食材都不动。
> 官方按这套规矩做了 34 个实验模型，按"改了哪个旋钮"分成 5 组。本方向搞懂三件事：规矩在代码里是怎么保证的、官方都做了哪些对比、我们怎么照做。
> 前两周只需要读代码写文档，不需要显卡、不需要装环境。

> **这个方向的最终目标**：学完后，给你一个研究问题，你能独立设计并执行受控对比实验——拆出变量、保证可复现、跑出对照、得出结论。
> **分阶段路线**：① 考古官方 34 个实验模型、验证实验可复现（E-1、E-2）→ ② 亲手做一组对比评测（E-3）→ ③ 沉淀安全网手册和实验记录模板（E-4、E-5）。

## 你需要准备什么

| 什么东西 | 什么时候才需要 | 怎么获取 |
|---|---|---|
| 一份仓库 clone | 第一天就要 | git clone |
| CPU 版 Python 环境 | 做 E-2、E-4 时 | `pip install -e '.[dev]'`，CPU 版 torch 就行，不下载任何权重 |
| 1 张 24GB 以上显卡 + 2 个官方实验模型（各约 24.8GB）+ 方向 C 的部署服务 + 方向 D 的评测环境 | 只做 E-3 时需要 | 显卡组内申请；模型用下载脚本（命令见 E-3），避开 tri 和 wan21_i2v_14b 这两个大号存档 |

不需要训练数据，也不需要全组统一搭环境。想做"小规模重训对比"是可选加分项，要 4 张 80GB 显卡，需要时单独申请。

## 先搞懂这几件事

**1. 配置文件就是实验矩阵。** OpenWAM 用一个叫 Hydra 的配置工具，说人话就是：训练配置 = 一份默认清单 + 命令行上临时改几个键。官方的 34 个实验，就是在这份清单上轮流改不同键改出来的。可改的旋钮有：选模型结构（`model=single_system|dual_system|tri_system`）、双系统里选交流方式（`model.architecture.variant=joint_self_attn|joint_cross_attn|idm`）、选注意力掩码（`model.architecture.attention_mask_mode`，掩码就是规定"计算时谁能看见谁"）、选视频骨干网络（`model/video_backbone=...`）、选视觉编码器（`model.video_backbone.encoder.*`）、选数据集（`dataloader=robotwin|libero|...`）。命令行上一次改键，就是实验矩阵里的一个格子。

**2. 配置怎么变成模型：一个总登记处。** 清单里写了"结构 + 交流方式"两个词，代码里有一个登记函数把它翻译成 6 个注册模型名之一；没写交流方式时会自动补默认。机制细节归方向 B，这里只需记住：所有"架构旋钮"都经过这一个登记处收口，对账时看它就够了。

**3. 随机种子：洗牌前记住牌序。** 训练里到处有随机：打乱数据、初始化、丢弃。想让两次训练一模一样，就得把每处随机都钉死——就像洗牌前先记住牌序，种子（seed）固定了，每次"洗"出来的顺序都一样。这套体系分四层：先把 Python、NumPy、torch 三件套的随机数钉住；再把显卡加速库锁成确定模式；然后把数据打乱的顺序钉在同一个种子上；最后每个训练步用"种子 + 卡的编号 × 100 万 + 步数"重新播种——给每张卡留 100 万宽的种子窗口，保证多卡训练时相邻两张卡的种子永远不撞。种子从配置里的 `project.seed: 42` 出发，在数据集构造之前就播好。**设了 seed = 这次训练可以原样复现；seed=null = 每次都不一样的正式跑。**

**4. debug 模式：20 步快速冒烟。** 配置里写 `training.debug=true`，训练就变成迷你版：只跑 20 步、第 10 步存一次档、学习率保持不变。这是所有消融实验统一的"先跑通再说"入口。

**5. 训练存档自带说明书。** 每个训练存档（checkpoint，模型权重文件）目录里都有一份 `config.yaml`，是训练时把完整配置原样写进去的，只写一次。部署和复现时以它为准，不靠人记。这是实验记录纪律的核心：目录自己会说话。

## 任务清单

### 任务 E-1：整理"官方实验对照表"——34 个实验模型各改了哪个旋钮

- **这件事是干嘛的**：官方把 34 个实验存档分 5 组，每组改一个维度，存档名字里藏着它改了哪些旋钮。你要把每个名字翻译回配置键，做成一张对照表。这张表是全组公共资产，E-3 选对比对象就靠它。
- **你要准备什么**：什么都不用，纯阅读。
- **具体怎么做**：
  1. 通读训练总配置和三份模型结构配置，列出每个维度对应的键：`architecture.framework`、`architecture.variant`、`attention_mask_mode`、`video_attention_mask_mode`、`video_backbone`、`encoder`、`detach_bridge`、`dataloader`。
  2. 打开下载脚本里的官方实验清单，逐组破译命名规则：
     - **Architecture 组**（7 个）：名字是 `{数据}_{结构}_{交流方式}`；尾巴带 `_detach` 就是 `detach_bridge=true`。
     - **Pretrain 组**（12 个）：开头 `pretrain_*` 或 `sft_*` 表示训练阶段；尾巴 `_mutual` 或 `_action_sees_video` 是掩码旋钮；`robot_only`、`two_stage`、`from_scratch` 是训练配方。
     - **Visual_Encoder 组**（6 个）：尾巴 `_dinov3|_vjepa21|_flux2|_wan22_vae` 对应编码器旋钮；`_svae` 表示叠加了一个压缩器。
     - **Video_Backbone 组**（5 个）：尾巴 `_cosmos25|_cosmos3|_wan21_i2v_14b|_wan21_vace_1_3b`；没尾巴的基座就是 `wan22_ti2v_5b`。
     - **Attention_Mask 组**（4 个）：尾巴 `_isolated|_mutual|_video_sees_action`；没尾巴的基座是 `action_sees_video`。
  3. 注意一个陷阱：`robotwin_dual_system_joint_self_attention` 这个名字出现在 3 个组里。它是同一个基座模型，在三组里都当"对照原点"（baseline）。对照表里要标出每组的 baseline 是谁。
  4. 给每个格子写出"复现这一格的命令行改键"。例：attention_mask 组的 mutual 格是 `model=dual_system model.architecture.variant=joint_self_attn model.architecture.attention_mask_mode=mutual dataloader=robotwin`。
- **做完交出什么**：对照表覆盖全部 34 个存档。每行包含：组、存档名、改了哪些旋钮及取值、对应配置键、命令行改键、同组 baseline。
- **大概要多久**：2-3 天。

### 任务 E-2：验证"同一个实验跑两遍，结果一模一样"

- **这件事是干嘛的**：对比实验可信的前提是"不改的地方真的没动"。你要读懂随机种子全链路，并在 CPU 上亲手验证：两次完全相同的运行，产出的 loss（损失值，即训练误差）序列逐位一致。这是所有后续实验的地基。
- **你要准备什么**：`pip install -e '.[dev]'`（CPU 版 torch 即可），不需要权重。
- **具体怎么做**：
  1. 先跑现成的种子测试，边跑边读它们各自钉住了哪一层：
     ```bash
     cd OpenWAM
     python -m pytest -q tests/test_seeding.py tests/test_dataloader_seed.py -v
     ```
  2. 把种子调用链画成一张图：从配置里的 `project.seed` 出发，到训练入口脚本下发种子、数据集构造前播种，再到训练器里模型初始化前播种、钉住数据打乱顺序、每个训练步前重新播种。然后回答三个问题：为什么每张卡之间要隔离种子？为什么每步要重新播种，而不是只靠全局种子？`seed=null` 时哪几层会自动失效？
  3. 写一个确定性验证脚本（CPU 上用小假模型，测试文件里有现成的假视频骨干可以参照）：固定 `seed=42`，把"建模型、喂一小批假数据、算 loss"这套流程完整跑两遍，用 `torch.equal`（不是 allclose）断言两遍的 loss 逐位相等；再把种子改成 43 跑第三遍，断言 loss 不一样；最后种子不动、只改一个无关的键（比如 `project.wandb.run_name`），断言 loss 仍然逐位相等。这三连就是"只改一个变量"的操作定义。
  4. 顺带验证 debug 模式的语义（20 步、恒定学习率），写进结论。
- **做完交出什么**：确定性验证脚本（三次断言全部成立）+ 种子调用链图 + 三问的回答。结论里必须写清边界：逐位一致在 CPU 小规模成立；GPU 多卡有非确定性，只能说"同配置同曲线可复现"，不能承诺跨硬件逐位一致。
- **大概要多久**：2-3 天。

### 任务 E-3：亲手做一组对比——两个官方实验模型在 LIBERO 考场上比一比

- **这件事是干嘛的**：从同一组里下载两个官方实验存档（它们只差一个旋钮），用方向 D 的 LIBERO 评测流程各考一遍，比成功率。先小样本验证流程，再扩量出正式数字。这是前四个任务方法论的第一次实战。
- **你要准备什么**：方向 C 的部署服务可用；方向 D 的 LIBERO 评测环境就绪；1 张 24GB 以上显卡；E-1 已完成。
- **具体怎么做**：
  1. 按 E-1 对照表选一对"只改一个变量"的存档，推荐 attention_mask 组（改动最干净）：
     ```bash
     python scripts/download_assets/download_openwam_checkpoints.py \
       --family study --study-type attention_mask --yes
     # 取 robotwin_dual_system_joint_self_attention{,_mutual} 两个（各 ~24.8GB）
     ```
     注意：实验存档多数在 robotwin 数据上训练，如果在 LIBERO 上考表现偏得太厉害，就改用方向 C/D 已支持的对应 benchmark，或改选架构组的一对。选这对的理由要写进报告。
  2. 逐个把存档起成服务，跑小样本评测（每个任务 5–10 次尝试，先验证流程能跑通、结果稳定）：
     ```bash
     bash scripts/deploy.sh assets/openwam_ckpt/openwam_study/attention_mask/<ckpt_a>
     # 另一终端按方向D的 LIBERO 流程起 labtasker 客户端，限制 trials 数
     ```
  3. 流程验证通过后，扩到方向 D 的标准尝试次数。两个存档必须用**同一份 manifest**（评测任务清单，保证同一批种子和初始状态，方向 D 的机制会管住这件事）各跑一遍。
  4. 对比成功率。再打开两个存档目录里的 `config.yaml` 逐键对比，确认它们真的只差目标旋钮。如果差得更多，说明这组对比不成立，回 E-1 重选。
- **做完交出什么**：复现报告，包含：选这对存档的依据、config.yaml 逐键对比的证据、两轮评测记录、带置信说明的对比结论（尝试次数、波动范围）。
- **大概要多久**：1 周（含下载和排队评测）。

### 任务 E-4：写一份"安全网使用手册"——仓库自带的测试怎么用

- **这件事是干嘛的**：仓库自带一批测试，把"改一个变量不会弄坏别的"这条规矩固化成了代码。你要跑通它们、读懂每组测试守的是什么，再写一份手册：做哪类消融之前，该跑哪些测试。
- **你要准备什么**：`pip install -e '.[dev]'`，全部 CPU 可跑，不需要权重。
- **具体怎么做**：
  1. 跑通四组测试：
     ```bash
     cd OpenWAM
     python -m pytest -q tests/test_split_attn_equivalence.py -v
     python -m pytest -q tests/test_train_deploy_consistency.py -v
     python -m pytest -q tests/test_mask_modes.py -v
     python -m pytest -q "tests/test_architecture_variants.py::test_framework_backbone_hydra_compose" -v
     ```
  2. 逐组读懂它们各自守住什么：
     - 第一组：拆开的注意力算路和整块算出来的结果零误差相等——拆层实现不会引入数值偏移。
     - 第二组：训练时和部署时模型表现一致——评测数字才能归因到训练。
     - 第三组：四种掩码模式"谁能看见谁"没搞错——这是掩码消融的语义基线。
     - 第四组：3 种结构 × 3 种骨干共 9 种配置组合都能正常拼出模型——这是 E-1 矩阵每个格子的合法性门禁。
  3. 做一次"故意破坏"实验：临时改一处实现（比如掩码填充方向），确认对应测试变红，再还原。这证明安全网真的在守。
  4. 写手册：按消融维度（换交流方式 / 换掩码 / 换骨干 / 换编码器 / 改训练循环）分别列出"动手前必跑"的测试子集和通过标准。注明 GPU 版测试要设 `OPENWAM_RUN_TRAIN_DEPLOY_GPU=1`，否则会静默跳过。
- **做完交出什么**：四组测试全绿的输出存档 + 一份破坏-变红-还原记录 + 安全网使用手册（按消融维度索引测试，组内公共资产）。
- **大概要多久**：2 天。

### 任务 E-5：做一份组内实验记录模板

- **这件事是干嘛的**：一次实验会留下三条记录通道：wandb 日志（一个实验追踪网站）、debug 的 loss 记录文件、存档目录里的 `config.yaml` 说明书。你要摸清这三条通道，做成一份全组统一的实验记录模板。
- **你要准备什么**：E-1、E-2 已完成；跑过一次 debug 冒烟，或读通了训练器的日志代码。不需要显卡。
- **具体怎么做**：
  1. wandb 通道：配置项 `cfg.project.wandb` 控制开关和 run 名。官方镜像默认离线记录（设了 `WANDB_MODE=offline`），日志落在 `wandb/offline-run-*` 目录，事后用 `wandb sync` 上传；想在线看板，启动前 `export WANDB_MODE=online` 并配 API key。验证方法：设 `project.wandb.run_name=e5_probe` 跑一次冒烟，确认 run 名和配置都传过去了。
  2. csv 通道：debug 运行会把 loss 写进 `debug_loss_history.csv`（首行写表头，列随标签走）。这是轻量留痕，不依赖 wandb。
  3. config 通道：训练时完整配置会写进存档目录的 `config.yaml`（只写一次），任何人拿到目录就能把模型原样重建，不需要口述配置。
  4. 做模板：规定每个对比实验必须记录——完整的命令行改键、`project.seed`、git commit、存档目录路径、与 baseline 的 config.yaml 差异、wandb run 名和 csv 路径、（E-3 类评测实验的）manifest 哈希与尝试次数。模板第一条纪律写死：**记录以存档里的 config.yaml 为准，口述和截图无效**。
- **做完交出什么**：实验记录模板（每个字段都注明由哪条通道产生）+ 用 E-2 或一次冒烟运行实填的样例一份。
- **大概要多久**：1-2 天。

推进顺序一句话：前两周并行做 E-1 和 E-4、读完 E-2 的种子部分；第 3–5 周收尾 E-2 并用它的产物填 E-5 样例；等方向 C/D 就绪后（约第 5–8 周）再做 E-3；小规模重训对比是可选加分项。

## 容易踩的坑

1. **官方没把完整训练配方全发出来**（人视角视频那部分数据没有读取代码），论文 Study 章节的全量训练数字复现不了——只做"评测对比"或"小规模重训对比"，并把这条边界写在报告开头。
2. **配置文件写的掩码和代码默认的掩码不是同一个**（配置写 `mutual`，代码默认 `ACTION_SEES_VIDEO`），绕过配置直接建模型会悄悄换掩码，同一对"对比实验"可能根本不在比同一个变量——一切记录以存档里的 `config.yaml` 为准。
3. **默认只保留最新一个存档，中间存档会被自动删掉**——需要中间步曲线或多步评测时，提前把要留的存档拷出运行目录，或调大 `training.keep_last_k_ckpts`。
4. **GPU 测试会静默跳过，"全绿"可能只是 CPU 绿**——手册里写清要设的环境变量，说"测试绿"时一律注明实际跑了哪些用例。

## 代码地图

```
configs/
├── train.yaml                     # 训练旋钮总清单，所有实验都从改它开始
├── model/
│   ├── single_system.yaml         # 单系统结构的旋钮
│   ├── dual_system.yaml           # 双系统结构的旋钮，消融实验的主战场
│   ├── tri_system.yaml            # 三系统结构的旋钮
│   ├── video_backbone/            # 视频骨干网络的候选名单
│   └── action_backbone/           # 动作模块的候选名单
└── dataloader/                    # 七个数据集的候选名单

scripts/
├── train.py                       # 训练入口，种子就是在这里播下去的
├── deploy.sh                      # 把训练存档起成服务的脚本
└── download_assets/
    └── download_openwam_checkpoints.py  # 官方模型下载清单，34 个实验模型在这里分 5 组登记

openwam/
├── model/architectures/           # 模型结构的登记处，配置在这里被翻译成具体模型
└── train/
    ├── openwam_trainer.py         # 训练主循环，播种、存档、调试模式都在这里发生
    └── utils/
        ├── seeding.py             # 随机种子工具箱，管"两次跑得一模一样"
        ├── checkpointing.py       # 管训练存档的保存、说明书和自动清理
        ├── training_utils.py      # 管训练日志和调试时的损失记录
        └── ckpt_model_loader.py   # 从存档目录把模型原样重建出来

tests/                             # 实验安全网：改一个变量会不会弄坏别的，跑这里就知道
├── test_seeding.py                # 验证种子钉得死不死
├── test_dataloader_seed.py        # 验证数据打乱顺序可以复现
├── test_split_attn_equivalence.py # 验证两种注意力算法算出来完全一样
├── test_train_deploy_consistency.py # 验证训练时和部署时模型表现一致
├── test_mask_modes.py             # 验证四种掩码模式"谁看得见谁"没搞错
└── test_architecture_variants.py  # 验证各种结构组合都能正常拼出模型
```
