# 方向 H：测试与质量基建

> 一句话定位：把 OpenWAM 仓库已有的 ~146 个测试文件当作"契约固化层"来读、跑、扩——维护分环境验证矩阵、解锁 GPU 一致性层、把各方向的复现发现沉淀回测试。读者：第一次接触该模块的同学。
> 前置：先读本目录 README.md 的总览，了解本方向在全局中的位置。

仓库根下文记为 `$R`（`/fact_home/yiyangyuan/workspace/projects/kaiyuan/OpenWAM`），所有路径均为仓库相对路径。行号以 main @ 90e94ae 为准（已逐一核对；若后续漂移，以符号锚点为准）。

## 1. 这个方向做什么、为什么值得做

OpenWAM 的测试体系不追求覆盖率，而是把"如果漂移就会静默出错"的跨模块不变量钉死：注意力拆分前后数值必须逐位相等、train 与 deploy 的 action 输出必须一致、旧 ExecutionPlan 状态机必须 import 失败、每个 backbone 基类必须声明 deploy 钩子。这些断言就是整个仓库的"防回归宪法"。读懂它们，等于拿到了一份由代码强制执行的架构说明书。

本方向在 WAM 技术栈中是**贯穿性的支撑层**：方向 A（环境搭建）的验证矩阵由 H1 持续维护；方向 B（部署与推理复现）跑出的协议矩阵、等价性证据由 H3 沉淀为仓库风格的回归测试；H2/H4 则分别解锁 GPU 一致性层与 benchmarks 离线评测链路。你既是测试体系的"用户"，也是它的"扩建者"。

能学到什么：pytest 的分层门禁设计（marker/skipif/importorskip 三件套）、如何用伪对象在 CPU 上钉住 GPU 系统的契约、资产管理（env 变量 + 默认目录 + skip 降级）的工程范式，以及"把一次复现实验写成可重复测试"的方法论——这是研究工程化最核心的一项技能。

## 2. 最小必要背景

**测试分层设计哲学**：CPU + 伪对象/fake tensor 为主，GPU + 真实权重为显式可选加成。CI（`.github/workflows/ci.yml:40`）只安装 CPU 版 torch（`--index-url .../whl/cpu`），`:44` 只跑 `make check`；GPU 用例一律打 `@pytest.mark.gpu` 并靠 Makefile 过滤掉。含义：默认路径必须在没有 GPU、没有权重的机器上全绿。

**pytest 基础设施**（`tests/conftest.py`）：

- `sys.path` 注入：`tests/conftest.py:9-15` 把 PROJECT_ROOT 与 `third_party/` 插进 `sys.path`，所以测试里可以直接 `from openwam...` / `from tests._assets import ...`。
- marker 注册：`tests/conftest.py:20-21` 注册 `gpu`（需 GPU）与 `data`（需 RoboTwin 数据）两个 marker。
- `stub_reason1` fixture：`tests/conftest.py:24` 的 `_StubReason1Encoder` 把 16GB 的 Qwen/Reason1 文本编码器替换为签名兼容、返回 `(B,512,100352)` 随机张量的桩；`:53` 定义 fixture，monkeypatch 进模块，使 Cosmos-Reason1 相关测试无需下载权重。

**资产解析**（`tests/_assets.py`）：`:23` 的 `_ASSETS` 字典登记 5 个权重键；`:32` 的 `asset_path(name)` 按"规范 env 变量 → 遗留别名 → `assets/video_backbone_ckpt/<...>` 默认目录"的顺序解析。所有权重依赖测试用 `skipif(not os.path.isdir(asset_path(...)))` 降级为 skip——见 `tests/test_video_backbone_consistency.py:29-30,100-101`。

**Makefile 目标**：

- `Makefile:4` `CORE_TEST_ARGS ?= tests --ignore=tests/test_tri_system_smoke.py`
- `Makefile:19-20` `make test` = `pytest -q -m "not gpu" $(CORE_TEST_ARGS)`（CPU 核心集，CI 等价）
- `Makefile:22-23` `make test-full` = `pytest -q tests`（全量，含 gpu marker）
- `Makefile:26` `make tri-system-smoke` = 单独跑被核心集 ignore 的 `tests/test_tri_system_smoke.py`

**dataloader 合成 bucket**（`tests/dataloader/conftest.py`）：`:22` 的 `FakeActionDataset` 继承 `BaseDataset`，按给定条数现场合成假 episode；`:84` 的 `fake_dataset_factory` 是构建器 fixture；`:18` 定义缓存根 `_CACHE_ROOT = tests/dataloader/.cache`——fixture 生成的数据落盘到这里并跨进程复用，**schema 变更后需手工删除该目录**，否则脏缓存造成假失败。

**pytest 配置**：`pytest.ini` 仅设 `testpaths=tests`、`python_files=test_*.py`、`addopts=-ra`——**没有**默认 marker 过滤，GPU 排除完全靠 Makefile/命令行的 `-m "not gpu"`。

## 3. 代码地图

```
tests/
├── conftest.py                    # 【入口】sys.path 注入 / gpu·data marker / stub_reason1（tests/conftest.py:9,20,53）
├── _assets.py                     # 资产解析 _ASSETS:23 / asset_path:32
├── test_split_attn_equivalence.py # ★ 拆分注意力零 atol 等价（tests/test_split_attn_equivalence.py:1-7）
├── test_train_deploy_consistency.py # ★ train↔deploy 数值一致 :50；GPU 开关 _GPU_INTEGRATION_FLAG:429
├── test_wan_pipeline_shape_checker.py # ★ check_resize_height_width 时间维上取整 :20
├── test_save_deploy_assets_hook.py  # ★ deploy 资产钩子无条件分发；:46 断言所有基类声明钩子
├── test_imports.py                # ★ Wan DiTBlock 公开属性契约 :204（MoT 依赖 modulation (1,6,dim)）
├── test_no_execution_plan.py      # ★ 旧 ExecutionPlan/RuntimeState 必须 import 失败 :12
├── test_architecture_smoke.py / _variants.py / test_mask_modes.py / test_mot_driver.py ...  # 架构不变量组
├── test_video_backbone_smoke.py / _consistency.py / _from_scratch.py / test_wan_registry_smoke.py ...  # 视频骨干组
├── test_action_backbone_smoke.py / test_freeze_strategy.py / test_optimizer_groups.py ...  # 动作骨干组
├── test_openwam_trainer.py / test_checkpointing.py / test_seeding.py ...  # 训练组
├── test_deployment_changes.py / test_async_executor.py / test_schedule_equivalence.py ...  # 部署组
├── dataloader/
│   ├── conftest.py                # FakeActionDataset:22 / .cache:18
│   └── test_mixture_*.py / test_lerobotv3_*.py / test_robocoin_*.py / test_interndata_a1_*.py ...  # 31 个，合成数据契约
├── benchmarks/
│   ├── fake_libero_labtasker_runtime.py / fake_robotwin_labtasker.py  # CPU 子进程模拟 labtasker 服务端
│   ├── test_eval_manifest.py      # manifest seal/verify/select（配 benchmarks/utils/eval_manifest.py:56）
│   ├── test_libero_labtasker.py / test_robotwin_labtasker.py / test_task_policy.py  # importorskip("labtasker")
│   └── test_libero_bridge.py / test_ebench_bridge.py / test_vlabench_bridge.py ...   # 共 11 个
├── docker/
│   └── test_docker_tools.py 等 2 个   # docker 工具链（CI 不跑 docker 集成）
└── model/
    └── 2 个                        # 模型级杂项
```

**复现价值最高的 6 个文件及其固化的契约**：

1. `tests/test_split_attn_equivalence.py` — 契约：`pre_attn_at_layer + attention + post_attn_at_layer` 三段组合必须与 `DiTBlock.forward` / `SelfAttnActionDiTBlock.forward` 在 **atol=0** 下数值相等（文件头 docstring :1-7 自述"hard merge gate"）。这是 Wan 拆分注意力推理路径的等价性证明。
2. `tests/test_train_deploy_consistency.py` — 契约：训练态 IDM 的 action 输出与 deploy stage2 逐位一致（`:50`）；tri_system 前向在 VLM attention mask padding 下不变（`:369`）。GPU 变体由 `_GPU_INTEGRATION_FLAG`（`:429`）门控。
3. `tests/test_wan_pipeline_shape_checker.py` — 契约：`check_resize_height_width`（导入于 `:20`）对 num_frames 按 VAE 时间压缩因子上取整，factor=1 时 passthrough。
4. `tests/test_save_deploy_assets_hook.py` — 契约：`BaseWAMArchitecture.save_assets_for_deployment` 无条件分发到各 backbone 的 `save_deploy_assets`；`:46` 用 `vars()` 断言每个 backbone 基类都声明了该钩子（即使是 no-op）。这是"deploy 产物自包含"的门禁。
5. `tests/test_imports.py` — 契约：Wan `DiTBlock` 公开属性面（`:204`）：`modulation` 形状 `(1,6,dim)`、`norm1/norm2` 的 `elementwise_affine=False`、`cross_attn.forward` 二元签名——MoT 拆分注意力直接依赖这些结构假设。
6. `tests/test_no_execution_plan.py` — 契约：旧 dispatch 状态机 `ExecutionPlan`/`RuntimeState` 必须从架构基类中**无法 import**（`:12`），防止历史代码复活。

## 4. 任务书

### 任务 H1：分环境验证矩阵维护（承接方向 A3）

- **目标**：维护"无 GPU 无权重"与"有 GPU 有权重"两套可跑集合的命令清单与绿/跳状态记录；任一方向变更环境（装新包、改 Dockerfile、升 torch）后重跑并更新矩阵。
- **前置**：方向 A 的环境已搭建（见 `方向A-环境搭建与验证矩阵.md` 的 A3 验证矩阵）。
- **步骤**：
  ```bash
  cd $R
  # A 集（CI 等价，任何机器可跑）：
  make test
  # 等价显式形式：
  python -m pytest -q -m "not gpu" tests --ignore=tests/test_tri_system_smoke.py
  # 记录 skip 明细（-ra 已在 pytest.ini addopts 中，自动打印 skip 原因）
  make test-full            # 全量收集，对比哪些用例只在 full 中出现
  make tri-system-smoke     # 核心集 ignore 掉的文件单独跑
  ```
  把结果填入验证矩阵表：每个测试组标注 pass / skip（原因）/ 需要的前置（权重、包、env 开关）。
- **验收标准**：持续全绿记录；每次环境变更后矩阵有更新条目，skip 原因可追溯。
- **产出**：分环境验证矩阵（在方向 A 的文档上追加维护）；skip→原因对照表。
- **难度（估计）**：低。**工作量（估计）**：首次 0.5 天，之后每次环境变更 0.5–1 小时。
- **代码锚点**：`Makefile:4,19-26`、`pytest.ini`、`.github/workflows/ci.yml:40-44`。

### 任务 H2：GPU 一致性层解锁

- **目标**：下载真实权重，令 `tests/test_video_backbone_consistency.py` 等 GPU 用例从 skip 转为 pass，拿到分步接口 bit-exact 的证据。
- **前置**：单卡 L20 级 GPU；~60GB 空闲磁盘；H1 的 A 集已全绿。
- **步骤**：
  ```bash
  cd $R
  # 1. 下载权重（写入默认目录 assets/video_backbone_ckpt/，asset_path 自动命中）
  python scripts/download_assets/download_video_backbone.py --name Wan2.2-TI2V-5B --source huggingface --yes   # ~34.2GB
  python scripts/download_assets/download_video_backbone.py --name Wan2.1-VACE-1.3B --source huggingface --yes  # ~19.0GB
  # 或不用默认目录时，导出规范 env 变量（tests/_assets.py:23 的 5 个键）：
  export OPENWAM_WAN22_TI2V_5B=/path/to/Wan2.2-TI2V-5B
  export OPENWAM_WAN21_VACE_1_3B=/path/to/Wan2.1-VACE-1.3B
  export OPENWAM_COSMOS25_2B=/path/to/Cosmos-Predict2.5-2B        # ~4.6GB（可选）
  export OPENWAM_COSMOS_REASON1_7B=/path/to/Cosmos-Reason1-7B     # ~16.6GB（可选）
  export OPENWAM_COSMOS3_EDGE=/path/to/Cosmos3-Edge               # ~9.2GB（可选）
  # 遗留别名（asset_path 也会识别，见 tests/_assets.py:24-28）：
  #   WAN22_TI2V_5B / WAN_TI2V_5B_PATH / WAN21_VACE_1_3B /
  #   COSMOS25_ASSET_PATH / REASON1_ASSET_PATH / COSMOS3_EDGE_ASSET_PATH
  # 2. 打开 GPU 集成开关并跑 GPU 层：
  OPENWAM_RUN_TRAIN_DEPLOY_GPU=1 OPENWAM_RUN_TRI_SYSTEM_GPU_SMOKE=1 \
    python -m pytest -q tests/test_video_backbone_consistency.py \
      tests/test_train_deploy_consistency.py tests/test_tri_system_smoke.py \
      tests/test_attention_backend_dispatch.py
  # 3. Cosmos real_load 组另需上游依赖（非 pip 可得）：
  git submodule update --init third_party/cosmos-predict2.5
  bash scripts/install_cosmos_predict25.sh    # 需 nvcc + cuDNN
  python -m pytest -q tests/test_cosmos_predict25_real_load.py tests/test_cosmos3_real_load.py
  ```
- **验收标准**：目标用例在 pytest 输出中由 `s`（skip）变为 `.`（pass）；保留前后两次 `-ra` 输出对比作为证据。
- **产出**：GPU 用例通过记录（命令 + 输出摘要 + 权重路径解析方式）。
- **难度（估计）**：中（主要是下载体量与 Cosmos 依赖编译）。**工作量（估计）**：1–2 天（含下载）。
- **代码锚点**：`tests/_assets.py:23-35`、`tests/test_video_backbone_consistency.py:29-30,100-101`、`tests/test_train_deploy_consistency.py:429-435`、`tests/test_tri_system_smoke.py:1287-1288`、`scripts/download_assets/download_video_backbone.py:85-112`。

### 任务 H3：复现过程回归测试化

- **目标**：把方向 B（部署与推理复现，见 `方向B-部署与推理复现.md`）跑出的 B2 协议矩阵、B4 等价性验证等一次性实验脚本，改写为仓库风格的 pytest 用例，合入组内 fork。
- **前置**：方向 B 的 B2/B4 已有可跑通的实验脚本与结论；通读 §3 列出的 6 个高价值文件，内化仓库测试风格。
- **步骤**：
  1. 从 B2 协议矩阵中挑出"漂移即静默出错"的断言（如 ws 协议字段、obs 预处理 shape、action 反归一化路径），每条对应一个测试函数。
  2. 仿照 `tests/test_train_deploy_consistency.py:50` 的写法：CPU + 伪对象构造 cfg（`OmegaConf.create({...})`），断言**行为契约**（shape/数值/异常），不断言实现细节（私有属性、调用次数）。
  3. 需要权重的断言一律走 `asset_path` + `skipif` + `@pytest.mark.gpu` 三件套（照抄 `tests/test_video_backbone_consistency.py:100-101` 的模板），保证 CI 上自动降级为 skip。
  4. 自证有效：故意破坏被钉住的契约（如改掉 mask 布局），确认新测试变红，再还原。
- **验收标准**：新测试在 `make test`（CPU 部分）下绿；破坏对照实验变红；测试断言的是行为契约而非实现细节；合入组内 fork。
- **产出**：若干 `tests/` 风格的新测试文件 + 每个测试钉住的契约一句话说明。
- **难度（估计）**：中。**工作量（估计）**：2–4 天（随 B2/B4 进展滚动进行）。
- **代码锚点**：模板来源 `tests/test_train_deploy_consistency.py:50`、`tests/test_split_attn_equivalence.py:1-7`、`tests/test_save_deploy_assets_hook.py:46`。

### 任务 H4：benchmarks 离线链路

- **目标**：不依赖真实 labtasker 服务端，用仓库自带的 fake labtasker 进程在 CPU 上走通 manifest 构建 → seal（内容哈希封存）→ submit → worker 执行 → 缓存/重试的完整评测编排链路；装 `labtasker==2.6.0` 解锁被 `importorskip` 跳过的用例。
- **前置**：H1 的 A 集已跑过一遍（知道哪些 benchmarks 用例当前是 skip）。
- **步骤**：
  ```bash
  cd $R
  pip install labtasker==2.6.0
  # 1. manifest 契约：seal/verify/select/篡改检测
  python -m pytest -q tests/benchmarks/test_eval_manifest.py
  # 2. fake labtasker 全流程：build_manifest 提交、worker 执行、缓存命中、失败重试
  python -m pytest -q tests/benchmarks/test_libero_labtasker.py \
    tests/benchmarks/test_robotwin_labtasker.py tests/benchmarks/test_task_policy.py
  # 3. 其余 bridge/scheduler 用例
  python -m pytest -q tests/benchmarks/
  ```
- **验收标准**：`tests/benchmarks/` 全目录无 fail（装包前是 skip 的用例装包后转 pass）；产出一份评测编排契约说明：manifest 的 `manifest_hash` 如何生成与校验（`benchmarks/utils/eval_manifest.py:56` 的 `seal_manifest`）、fake 服务端如何被测试以子进程拉起、submit/worker/retry 的状态流转。
- **产出**：`tests/benchmarks/` 全过记录 + 编排契约说明（1–2 页）。
- **难度（估计）**：中。**工作量（估计）**：1–2 天。
- **代码锚点**：`tests/benchmarks/fake_libero_labtasker_runtime.py`、`tests/benchmarks/fake_robotwin_labtasker.py`、`tests/benchmarks/test_eval_manifest.py:20-43`、`tests/benchmarks/test_libero_labtasker.py:96-162`、`benchmarks/utils/eval_manifest.py:56-80`。

## 5. 建议推进顺序与里程碑

1. **W1**：H1 首次建表（跑通 A 集、记录 skip 基线）——这是全组环境的地基，最先做。
2. **W1–W2**：读 §3 的 6 个高价值文件（按 §7 顺序），做"故意破坏→变红"对照实验，内化契约含义。
3. **W2–W3**：H4（离线、无外部依赖，独立性强，可与任何人并行）。
4. **W2 起滚动**：H1 转为轮值维护（每次环境变更触发）。
5. **等 GPU/权重就绪后**：H2（依赖方向 A 的 GPU 环境与下载窗口）。
6. **方向 B 的 B2/B4 出结论后**：H3 滚动沉淀，随复现进度持续合入。

里程碑：M1 末出验证矩阵 v1；M2 末出 H4 编排契约说明；GPU 到位后一周内出 H2 通过记录；M3 起 H3 持续有测试合入。

## 6. 风险与坑

1. **`data` marker 是死代码**：`tests/conftest.py:21` 注册了 `data: test requires RoboTwin data`，但全仓库 `pytest.mark.data` 零命中（已用 grep 核验），Makefile 只过滤 `not gpu`。规避：不要指望 `-m "not data"` 能挡掉任何用例；涉及真实数据的跳过一律看具体文件的 skipif。
2. **`tests/dataloader/.cache` 需手工删除**：缓存根定义在 `tests/dataloader/conftest.py:18`，跨进程复用且不自动失效；fixture schema 变更后旧缓存会造成假失败。规避：改 fixture 后 `rm -rf tests/dataloader/.cache` 再跑。
3. **`OPENWAM_RUN_*` 开关未设时即使有 GPU 也 skip**：`tests/test_train_deploy_consistency.py:434` 与 `tests/test_tri_system_smoke.py:1287` 的 skipif 同时要求 env 开关=1 **且** CUDA 可用。容易把全 skip 误判为"已覆盖"。规避：验证矩阵中单独记录"开关未开"与"硬件缺失"两种 skip 原因。
4. **CI 与本地能力落差**：`.github/workflows/ci.yml:40-44` 只装 CPU torch、只跑 `make check`；GPU 层、`make tri-system-smoke`、docker 集成均无 CI 覆盖。规避："CI 绿"只代表 CPU 核心集绿，复现可行性一律以本方向维护的验证矩阵为准（对应母文档风险 R4）。
5. **CPU 版 train/deploy 一致性的 GPU 变体实为 CPU 桩跑在 CUDA**：`tests/test_train_deploy_consistency.py:441` 注释自承无法加载 Wan2.2-TI2V-5B，只是把 CPU 桩重跑在 CUDA 上验证 device-portability——**真权重端到端 train/deploy 一致性在默认路径缺位**。规避：H2 完成后意识到这层仍是桩；若研究依赖该保证，需在 H3 中自补真权重端到端用例（工作量大，先与导师确认必要性）。
6. **Cosmos GPU 用例依赖非 pip 可得的上游包**：`cosmos_predict2` 需 `git submodule update --init third_party/cosmos-predict2.5` + `bash scripts/install_cosmos_predict25.sh`（需 nvcc/cuDNN 现场编译），否则相关用例永远 skip。规避：H2 步骤 3 提前一周发起（对应母文档风险 R9）。
7. **`tests/benchmarks/` 的字面量路径断言**：部分断言含 `/shared/model`、`/shared/cache` 等绝对路径字面量（多为参数回显，不读盘），易与真实部署路径混淆。规避：读断言时区分"回显校验"与"真实 IO"。

## 7. 推荐阅读顺序

1. 本目录 `README.md` —— 知道方向 H 在全局的位置与交叉依赖。
2. `方向A-环境搭建与验证矩阵.md` —— 前置：H1 承接其 A3 验证矩阵，先搞清环境怎么搭。
3. `tests/conftest.py` 全文（约 60 行）—— sys.path 注入、marker、stub fixture，一切测试的地基。
4. `Makefile:1-40` + `pytest.ini` —— 门禁命令的真实含义。
5. `tests/test_split_attn_equivalence.py` —— 看一个"零 atol 硬门禁"长什么样，建立契约测试的审美基准。
6. `tests/test_train_deploy_consistency.py`（重点 :50 与 :429 附近）—— 学 CPU 桩 + GPU 开关的双层写法，H3 的主要模板。
7. `tests/_assets.py` + `tests/test_video_backbone_consistency.py:1-40` —— 资产解析与 skipif 降级模板，H2 前置。
8. `tests/test_save_deploy_assets_hook.py`、`tests/test_imports.py:204`、`tests/test_no_execution_plan.py` —— 三个小文件，快速看完"钩子存在性 / 公开属性面 / 负向 import"三种契约类型。
9. `tests/dataloader/conftest.py` + 任选一个 `tests/dataloader/test_mixture_*.py` —— 理解合成 bucket 与 .cache 机制。
10. `tests/benchmarks/test_eval_manifest.py` → `test_libero_labtasker.py` → `benchmarks/utils/eval_manifest.py:56` —— 由测试到实现，读懂 manifest/seal/submit 编排，H4 直接前置。
11. `方向B-部署与推理复现.md`（B2/B4 节）—— H3 的服务对象：知道要沉淀什么，再回头设计测试。
