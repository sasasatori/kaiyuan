# 方向 A：环境搭建与验证矩阵

> 一句话定位：搭好全组共享的 OpenWAM 运行环境，并产出"什么能跑、什么不能跑"的权威验证矩阵。读者：第一次接触该模块的同学。
> 前置：先读本目录 README.md 的总览，了解本方向在全局中的位置。
> 所有路径相对仓库根 `$R = OpenWAM/`；行号以 main @ `90e94ae` 为准（本文引用均已实际核对），若上游更新导致行号漂移，以符号锚点为准。

## 1. 这个方向做什么、为什么值得做

OpenWAM 是一个 World–Action Model（WAM）系统预训练研究栈：数据层（13 种注册数据集）→ 模型层（6 架构 × 5 骨干）→ 训练（torchrun + Accelerate + ZeRO-2）→ 部署（WebSocket PolicyServer）→ 评测（8 个 benchmark 客户端）。这一切的前提是一台机器能把代码跑起来——本方向就是这个前提。

好消息是仓库的工程化程度很高：Docker 镜像用 CUDA digest 钉死 + Ubuntu snapshot apt + uv 锁文件三层手段保证可复现（`docker/Dockerfile`）；CPU 测试门禁 `make test` 与 CI 等价（`Makefile:19`、`.github/workflows/ci.yml:44`）；还有一套离线 bundle 交付协议（`docker/offline.py:13 SCHEMA = 4`）。坏消息是 **CI 只覆盖 CPU**：GPU 路径、真实权重、模型端到端都没有 CI 保障，`assets/openwam_usage_docs/docker-validation.md` 诚实列出了未验证项。所以"CI 绿"不等于"复现可行"——本方向产出的验证矩阵才是全组的地面实况。

做完本方向你会得到：Docker 镜像构建与 compose 编排的实战经验、pytest marker 分层测试体系的理解、Hydra 配置回写机制的认识，以及一份全组都会引用的环境 SOP 与排障 wiki。方向 B（部署与推理复现，见 `方向B-部署与推理复现.md`）和方向 C（训练管线复现，见 `方向C-训练管线复现.md`）都直接阻塞在你这里。

## 2. 最小必要背景

**概念 1：Docker 五阶段镜像**。镜像分层构建，每层职责单一，最终产物是 `default` 阶段。
- 锚点：`docker/Dockerfile:1` `ARG CUDA_IMAGE`（基底 `nvidia/cuda:12.8.1-devel-ubuntu24.04` 以 sha256 digest 钉死）；`docker/Dockerfile:6` `ARG UBUNTU_SNAPSHOT=20260910T000000Z`（apt 全部走 snapshot.ubuntu.com 不可变归档，禁用 rolling NVIDIA apt 源）。
- 五个阶段：`docker/Dockerfile:2` `AS system-dependencies`（snapshot apt）→ `:26` `AS dependencies`（venv + torch cu128 + uv 锁）→ `:60` `AS openwam`（`pip install -e .` 可编辑安装、`USER 1000:1000`、`EXPOSE 8848`）→ `:106` `AS cosmos`（可选，跑 submodule 安装脚本）→ `:112` `AS default`（`docker build` 默认目标，即 openwam）。

**概念 2：依赖锁文件**。Python 依赖不看 `pyproject.toml` 的范围约束，而是看 uv 生成的精确锁。
- 锚点：`docker/requirements-cu128.txt`（441 行全精确钉版）：`torch==2.7.1+cu128`（`:357`）、`transformers==5.16.1`（`:383`）、`deepspeed==0.18.9`（`:43`）、`nvidia-cudnn-cu12==9.7.1.26`（`:180`）。顶层范围约束在 `docker/constraints-cu128.txt`。

**概念 3：离线 bundle（schema-4）**。把镜像 + 配置文件打包成 gzip，可在完全离线机器上 verify/load。
- 锚点：`docker/offline.py:13` `SCHEMA = 4`；`docker/offline.py:16` `CONFIG_FILES`（bundle 内含 compose.yaml、compose override、`.env.example`、docker.md、docker-validation.md、offline.py 自身）。导出经 `make docker-export`（`docker/docker.mk:66`）。**注意：bundle 不含任何权重/数据**。

**概念 4：pytest marker 分层**。测试分 CPU 层（默认跑）与 GPU 层（`@pytest.mark.gpu`，默认排除）；GPU 用例经 `asset_path()` 找权重，找不到就 skip 而不是 fail。
- 锚点：`tests/conftest.py:18` `pytest_configure` 注册 `gpu`/`data` 两个 marker；`tests/_assets.py:23` `_ASSETS`（5 个权重键，如 `OPENWAM_WAN22_TI2V_5B`）、`tests/_assets.py:32` `asset_path(name)`（解析顺序：env 变量 → 遗留 env 别名 → `assets/video_backbone_ckpt/<name>` 默认目录）。
- 另有两个 env 开关控制 GPU 集成用例：`OPENWAM_RUN_TRAIN_DEPLOY_GPU=1`、`OPENWAM_RUN_TRI_SYSTEM_GPU_SMOKE=1`，不设则即使有 GPU 也 skip。

**概念 5：`make test` 与 CI 的边界**。`make test` = `pytest -q -m "not gpu" tests --ignore=tests/test_tri_system_smoke.py`（`Makefile:19-20`，`Makefile:4` `CORE_TEST_ARGS`）。CI（`.github/workflows/ci.yml`）只装 **CPU 版** torch（`ci.yml:40` `--index-url .../whl/cpu`），跑 `make check`（`ci.yml:44`）外加一条 import 边界检查（`ci.yml:46-51`：禁止 `openwam/` import `benchmarks`）。**CI 无任何 GPU/model 覆盖**；`docker.yml` 会真实 build CUDA 镜像，但也只跑容器内 CPU 校验。

**概念 6：compose 四种运行形态**。`compose.yaml` 定义 `serve`（起 PolicyServer）、`train`（torchrun 训练）、`dev`（开发 shell）、`gpu-check`（`compose.yaml:107`，GPU 冒烟：BF16 matmul / torch.compile / DeepSpeed FusedAdam / 多卡 NCCL all-reduce，见 `docker/gpu_smoke.py:10` `main`、`:15` `--all-visible`）。入口链：`docker/runtime_user.py`（伪造 NSS 身份）→ `docker/entrypoint.sh:16` `serve)` / `:20` `train)` 分发。

## 3. 代码地图

```
$R/
├── docker/                        # 环境与交付底座（本方向主战场）
│   ├── Dockerfile                 # 五阶段镜像定义【入口文件，A1 必读】
│   ├── requirements-cu128.txt     # uv 锁文件（441 行精确钉版）
│   ├── constraints-cu128.txt      # 顶层版本约束
│   ├── .env.example               # compose 配置模板【A1 第一步要 cp 它】
│   ├── docker.mk                  # make 目标集：docker-build:33 / docker-check:53 / docker-integration-check:63 / docker-export:66
│   ├── compose.dev.yaml / compose.host.yaml / compose.worktree.yaml  # 三种 compose override
│   ├── runtime_user.py            # 容器内身份/home 适配（:1 configure_cuda_compat）
│   ├── entrypoint.sh              # serve/train 分发（:16/:20）
│   ├── gpu_smoke.py               # GPU 冒烟脚本（:10 main，:15 --all-visible 多卡 NCCL）
│   ├── healthcheck.py             # 纯 websockets ping/pong，不 import torch
│   └── offline.py                 # 离线 bundle 协议（:13 SCHEMA=4）
├── compose.yaml                   # serve/train/dev/gpu-check 四服务（:107 gpu-check）
├── Makefile                       # make test:19（CPU 核心集）/ test-full:22 / check:38
├── pytest.ini                     # testpaths=tests；GPU 排除靠 Makefile 而非 ini
├── .github/workflows/
│   ├── ci.yml                     # CPU-only CI（:40 CPU torch，:44 make check）
│   └── docker.yml                 # 真实构建 CUDA 镜像 + docker-check（要求 ≥60GB 盘）
├── tests/
│   ├── conftest.py                # marker 注册（:18）+ _StubReason1Encoder（:24）
│   ├── _assets.py                 # GPU 测试权重解析（:23 _ASSETS，:32 asset_path）
│   ├── test_split_attn_equivalence.py     # A4 门禁①（:106/:142/:195）
│   ├── test_train_deploy_consistency.py   # A4 门禁②（:50/:369；:437 GPU 变体）
│   ├── test_save_deploy_assets_hook.py    # A4 门禁③（:32/:46）
│   ├── test_no_execution_plan.py          # A4 门禁④（:12/:20/:28/:41）
│   └── dataloader/conftest.py     # fixture 缓存 tests/dataloader/.cache（需手工删除刷新）
├── scripts/download_assets/       # 5 个资产下载器【A5 主战场】
│   ├── download_video_backbone.py       # :83 MODELS，5 个视频骨干，回写 model_path
│   ├── download_benchmark_data.py       # :81 BENCHMARKS（8 个），:360 _POST_DOWNLOAD，自动建 stats + 回写 dataset_dir
│   ├── download_visual_encoder.py       # :97 MODELS（4 个编码器，含 gated/直链）
│   ├── download_vlm_backbone.py         # :83 MODELS（Qwen3-VL-2B，回写 checkpoint_path）
│   └── download_openwam_checkpoints.py  # :99 ALPHA（14 个），:120 STUDY_GROUPS（34 个）
├── scripts/install_cosmos_predict25.sh  # Cosmos 可选依赖安装（编 transformer-engine）
├── third_party/cosmos-predict2.5/       # 唯一 submodule，【当前为空，未 init】
└── assets/openwam_usage_docs/
    ├── docker.md                  # 环境/Docker/离线/排障权威文档【先读这个】
    └── docker-validation.md       # 官方验证记录与未验证项清单
```

## 4. 任务书

### 任务 A1：Docker 路线搭建

**目标**：从零 build 出镜像并通过全部容器校验。
**前置**：联网 Linux 机器；Docker Engine + Compose ≥2.30；NVIDIA driver + Container Toolkit（`nvidia-smi` 可见卡）；**空闲磁盘 ≥60GB**（`docker.yml` 对 CI runner 的硬性要求，本地同理；镜像本身约 25GB）。
**步骤**：
```bash
cd $R
cp docker/.env.example .env
# 编辑 .env：把 OPENWAM_UID / OPENWAM_GID 改成 id -u / id -g 的输出（.env.example:7-8）
mkdir -p .cache/docker outputs
make docker-build                            # docker/docker.mk:33；首次拉 CUDA 层最慢，1.5–4h（估计）
make docker-check                            # docker/docker.mk:53；compose 校验 + 容器内 pip check + CPU 测试
make docker-integration-check PYTHON=python3 # docker/docker.mk:63
docker compose run --rm -T gpu-check         # compose.yaml:107；多卡机加 -- --all-visible 验 NCCL
```
**验收标准**：四条命令全部通过；记录各步耗时与 `docker images` 体积、构建前后磁盘占用。
**产出**：Docker 环境 SOP 一页（含耗时/磁盘实测表）。
**难度（估计）**：中。**工作量（估计）**：0.5–1 天（大部分是等待构建）。
**代码锚点**：`docker/Dockerfile:1`、`docker/Dockerfile:6`、`docker/docker.mk:33`、`compose.yaml:107`、`docker/gpu_smoke.py:15`。

### 任务 A2：Native 路线搭建

**目标**：不用 Docker，在 conda 环境直接装出可跑 `make test` 的环境，作为 Docker 路线的对照（也让改代码-跑测试的循环更快）。
**前置**：conda；gcc（deepspeed 是 sdist 需现场编译）；约 20GB 磁盘。
**步骤**：
```bash
cd $R
conda create -n openwam python=3.10 -y && conda activate openwam
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu128
pip install -e '.[dev]'
make test
```
**验收标准**：`make test` 全绿（skip 允许，fail 不允许）；记录与 A1 容器内测试结果的差异。
**产出**：Native 环境 SOP + 与 Docker 路线的差异说明（包版本差异用 `pip freeze | diff` 对 `docker/requirements-cu128.txt`）。
**难度（估计）**：中。**工作量（估计）**：0.5 天。
**代码锚点**：`docker/requirements-cu128.txt:357`（torch 版本以此为基准）、`Makefile:19`、`.github/workflows/ci.yml:40`（CI 同款装法，但 CI 装的是 CPU torch）。

### 任务 A3：CPU 验证矩阵

**目标**：跑 `make test`，统计 pass/skip，并逐项解释 skip 原因，产出"GPU 层 skip 说明表"。
**前置**：A1 或 A2 任一完成。
**步骤**：
```bash
cd $R
python -m pytest -m "not gpu" tests --ignore=tests/test_tri_system_smoke.py -ra | tail -40
# 收集 skip 明细：
python -m pytest -m "not gpu" tests --ignore=tests/test_tri_system_smoke.py -rs 2>&1 | grep SKIPPED | sort | uniq -c | sort -rn
# 再反向收集"被 -m 'not gpu' 排除"的 GPU 用例清单：
python -m pytest tests --collect-only -q -m gpu 2>/dev/null | head -50
```
把 skip 原因归成四类：① 缺 `diffusers`（`test_cosmos3_*` 用 `pytest.importorskip`）；② 缺 `labtasker`（`tests/benchmarks/` 部分用例）；③ 缺权重（`tests/_assets.py:32 asset_path` 解析不到路径）；④ 缺 env 开关（`OPENWAM_RUN_TRAIN_DEPLOY_GPU` / `OPENWAM_RUN_TRI_SYSTEM_GPU_SMOKE`）。
**验收标准**：全绿；skip 说明表每一行都能指到代码里的 skipif/importorskip 位置。
**产出**：验证矩阵表（用例组 × 状态 × skip 原因 × 代码锚点）。
**难度（估计）**：低。**工作量（估计）**：0.5 天。
**代码锚点**：`Makefile:19-20`、`tests/conftest.py:18`、`tests/_assets.py:23`、`tests/dataloader/conftest.py:1-6`。

### 任务 A4：架构不变量门禁复现

**目标**：单独跑通 4 个"漂移即静默出错"的契约测试，理解它们各自冻结什么，并做一次"故意破坏 → 变红 → 恢复"的对照实验，证明门禁真的在守门。
**前置**：A2（native 跑测试循环最快）；无需 GPU、无需权重。
**步骤**：
```bash
cd $R
python -m pytest -q tests/test_split_attn_equivalence.py    # 拆分注意力零 atol 等价：:106 wan_dit_block / :142 action_mot / :195 异构 hidden_dim
python -m pytest -q tests/test_train_deploy_consistency.py  # train↔deploy 数值一致：:50 idm stage2 / :369 tri vlm mask padding
python -m pytest -q tests/test_save_deploy_assets_hook.py   # save_assets_for_deployment 无条件分发：:32 分发 / :46 全骨干声明 hook
python -m pytest -q tests/test_no_execution_plan.py         # 旧 dispatch 状态机必须 import 失败：:12/:20/:28/:41
```
故意破坏对照（示例，任选其一）：
```bash
# 例：破坏拆分等价——临时把 pre_attn/post_attn 拆分路径中一处残差删掉或改 atol
git stash list   # 先确认工作区干净
# 修改后重跑对应测试，确认变红；然后：
git checkout -- <改过的文件>   # 恢复，重跑确认回绿
```
**验收标准**：4 项全过 + 1 份破坏对照实验记录（改了什么 → 哪个测试变红 → 报错信息 → 恢复后回绿）。
**产出**：不变量清单（每个门禁一句话说明"它防止什么回归"）+ 破坏对照记录。
**难度（估计）**：低-中。**工作量（估计）**：0.5–1 天。
**代码锚点**：`tests/test_split_attn_equivalence.py:106`、`tests/test_train_deploy_consistency.py:50`、`tests/test_save_deploy_assets_hook.py:46`、`tests/test_no_execution_plan.py:12`。**注意**：`test_train_deploy_consistency.py:437` 的 GPU 变体注释自承"只是把 CPU 桩重跑在 CUDA"，真正的 Wan 端到端一致性默认路径缺位——不要误认为已被覆盖。

### 任务 A5：资产下载演练

**目标**：逐个跑 5 个下载器，核对它们回写的 yaml 字段与归一化统计生成，产出支持矩阵表。
**前置**：A1 或 A2；联网；磁盘按选下的资产预留（见下表）；DINOv3/FLUX.2 需先在 HF 页面 accept license 并 `hf auth login`。
**步骤**（5 个下载器统一支持交互菜单与 `--name/--source/--root/--yes` 非交互模式）：
```bash
cd $R
python scripts/download_assets/download_benchmark_data.py --name LIBERO --yes   # 最小 1.9GB，先拿它练手；自动建 stats + 回写 dataset_dir
python scripts/download_assets/download_video_backbone.py --name Wan2.2-TI2V-5B --yes  # 34.2GB；可选 --source modelscope
python scripts/download_assets/download_vlm_backbone.py --yes                    # Qwen3-VL-2B，4.3GB
python scripts/download_assets/download_visual_encoder.py --yes                  # 选一个非 gated 的试，如 Wan2.2 VAE
python scripts/download_assets/download_openwam_checkpoints.py                   # 先看菜单选 1 个 alpha ckpt（~17–50GB/个）
```
下载后核对：① 对应 yaml 的 `model_path`/`dataset_dir`/`checkpoint_path` 是否被回写（`configs/model/video_backbone/*.yaml` 等）；② benchmark 数据是否生成了归一化统计（`download_benchmark_data.py:360 _POST_DOWNLOAD` 链路）；③ 体积与预期是否吻合。
**验收标准**：支持矩阵表（模型 × 源(huggingface/modelscope/直链) × 体积 × 回写字段 × 是否 gated）+ 至少 3 个下载器实际下载验证（其中必须含 LIBERO 与 Wan2.2-TI2V-5B，它们是方向 B/C 的入口资产）。
**产出**：资产清单表 + 下载验证记录。
**难度（估计）**：中（受带宽与 gated 授权影响）。**工作量（估计）**：1 天（主要是下载等待）。
**代码锚点**：`scripts/download_assets/download_video_backbone.py:83`、`download_benchmark_data.py:81`/`:360`、`download_visual_encoder.py:97`、`download_vlm_backbone.py:83`、`download_openwam_checkpoints.py:99`/`:120`。

### 任务 A6：已知坑验证

**目标**：在本组 GPU 机上确认两个 NCCL workaround 的有效性，形成本组卡型的适配结论。
**前置**：A1 完成；多卡 GPU 机（H100 或同级）。
**步骤**：
```bash
# ① NCCL_RAS_ENABLE=0（镜像内已默认 ENV，docker/Dockerfile:78）：H100 上不设会 SIGSEGV 退出
docker compose run --rm -T gpu-check -- --all-visible
# ② NCCL_NVLS_ENABLE=0：四卡以上 NVLS 初始化 hang 时需要在 .env / compose 环境追加
# ③ 对照实验（可选，有风险意识再做）：在容器内 unset NCCL_RAS_ENABLE 重跑，观察是否复现 SIGSEGV
nvidia-smi -q | grep "Product Name"   # 记录本组卡型与驱动版本
```
**验收标准**：gpu-check 多卡 NCCL all-reduce 通过；排障记录（卡型、驱动、CUDA、是否触发 RAS/NVLS 问题、workaround 是否必需）并入组内 wiki。
**产出**：本组 GPU 适配结论一页。
**难度（估计）**：中。**工作量（估计）**：0.5 天。
**代码锚点**：`docker/Dockerfile:78`、`docker/gpu_smoke.py:15`、`assets/openwam_usage_docs/docker.md`（排障表）。

## 5. 建议推进顺序与里程碑

1. **第 1 天**：A2（native 先装起来，装的过程读 `docker.md`）→ A3（当天就能出验证矩阵初稿）。
2. **第 1–2 天**：A1 并行挂着 build（耗时最长，先启动），等待期间做 A4。
3. **第 2–3 天**：A5（LIBERO + Wan2.2-TI2V-5B 优先，这是方向 B/C 的入口资产；gated 授权当天就发起申请）。
4. **第 3–4 天**：A6（需要多卡机时间片）。
5. **里程碑**：W1 末交付——环境 SOP（双路线）、验证矩阵、资产清单、GPU 适配结论，全组 unblock。

## 6. 风险与坑

| 坑 | 锚点 | 规避方法 |
|---|---|---|
| `third_party/cosmos-predict2.5` submodule **当前为空**（已核实 90e94ae 工作区） | `third_party/cosmos-predict2.5/`（空目录）；安装脚本 `scripts/install_cosmos_predict25.sh` | Cosmos 路线先 `git submodule update --init third_party/cosmos-predict2.5` 再跑 install 脚本（编 transformer-engine 需 nvcc + cuDNN，约 10 min）；本方向不阻塞，仅记录 |
| DINOv3 / FLUX.2 为 HF **gated** 资产 | `scripts/download_assets/download_visual_encoder.py:97` | 提前一周在 HF 页面 accept license + `hf auth login`；V-JEPA 走官方直链 `dl.fbaipublicfiles.com`，国内可达性需实测 |
| **CI 无 GPU 覆盖**，"CI 绿"≠"复现可行" | `.github/workflows/ci.yml:40-44`；未验证项清单 `assets/openwam_usage_docs/docker-validation.md` | 以本方向 A3/A6 的验证矩阵为准，不引用 CI 状态做结论 |
| `tests/dataloader/.cache` 复用且**需手工删除**才刷新，fixture schema 变更后会假失败 | `tests/dataloader/conftest.py:1-6`（docstring 明说） | 改 fixture/reader 后 `rm -rf tests/dataloader/.cache`；遇到诡异失败先删缓存再排查 |
| `.env.example` 的 `OPENWAM_CHECKPOINT_DIR` 默认指向 `OpenWAM-Alpha-Sim-RoboTwin-Full`，未下载该 ckpt 时 serve 起不来（`create_host_path:false`） | `docker/.env.example:26` | 先跑 A5 下载对应 checkpoint，或先把 `.env` 该字段改指向已存在的目录 |
| Docker 构建吃盘：**≥60GB** 空闲是硬要求 | `.github/workflows/docker.yml`（CI 清理磁盘步骤） | 构建前 `df -h` 确认；`.cache/docker` 与镜像层都在本机盘 |
| H100 多卡 NCCL 坑：RAS SIGSEGV、四卡 NVLS hang | `docker/Dockerfile:78`（默认 `NCCL_RAS_ENABLE=0`）；`docker.md` 排障表 | 保持镜像默认 ENV；hang 时追加 `NCCL_NVLS_ENABLE=0`；结论记入 A6 |
| GPU 集成用例有 env 开关，不设则**有 GPU 也 skip**，易误以为已覆盖 | `tests/test_train_deploy_consistency.py:437`（`_GPU_INTEGRATION_FLAG` 门控） | 跑 GPU 层时显式 `OPENWAM_RUN_TRAIN_DEPLOY_GPU=1 OPENWAM_RUN_TRI_SYSTEM_GPU_SMOKE=1` |
| benchmark 下载器建 stats 需要 openwam 可 import | `scripts/download_assets/download_benchmark_data.py:360` 链路 | 在已完成 A1/A2 的环境里跑下载器，裸 shell 不行 |

## 7. 推荐阅读顺序

1. `assets/openwam_usage_docs/docker.md` — 环境/Docker/离线/排障的官方权威文档，搭建前通读。
2. `docker/Dockerfile`（配 `docker/requirements-cu128.txt` 翻阅）— 弄清五阶段镜像每层装了什么、为什么钉死。
3. `docker/.env.example` + `compose.yaml` — 弄清四个服务与全部 `OPENWAM_*` 旋钮。
4. `assets/openwam_usage_docs/docker-validation.md` — 官方验证过/未验证的边界，校准预期。
5. `Makefile` + `pytest.ini` + `tests/conftest.py` — 弄清 `make test` 到底跑了什么、marker 怎么分层。
6. `tests/_assets.py` + A4 的四个测试文件 — 弄清 GPU 用例的资产解析与四条架构不变量。
7. `scripts/download_assets/` 五个下载器 — 边跑 A5 边读，重点看 `write_config` 回写与 `_POST_DOWNLOAD` 统计链路。
8. 完成后转交：方向 B 见 `方向B-部署与推理复现.md`（部署与推理复现，依赖 A5 下载的 checkpoint），方向 C 见 `方向C-训练管线复现.md`（训练管线复现，依赖 A5 的 LIBERO 数据与 Wan2.2-5B 权重）。
