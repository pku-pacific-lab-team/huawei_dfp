# 多架构 GEMM 仿真工程

本工程提供 Ascend 910、TPU v4、原生 TSP、CIM-TSP 四个独立 INT8 GEMM 分析入口。当前交付以可追溯规格、容量检查、动作计数和硬件库存为主。

技术说明见 [ARCHITECTURE.md](ARCHITECTURE.md)：[四个仿真器结构](ARCHITECTURE.md#simulator-structure)、[公共口径](ARCHITECTURE.md#conventions)、[Ascend](ARCHITECTURE.md#ascend)、[TPU](ARCHITECTURE.md#tpu)、[TSP](ARCHITECTURE.md#tsp)、[CIM Macro](ARCHITECTURE.md#cim-macro)、[CIM-TSP](ARCHITECTURE.md#cim-tsp)、[验证状态](ARCHITECTURE.md#validation)。

文档维护约定：工程内维护使用说明、现行技术说明、参数来源台账和验收证据。教学方案、汇报 PPT、论文原文、历史计划与整理备份放在工程外资料目录，本机统一使用 `D:\IC_paper\huawei_dfp`。运行和自动验收不依赖这些本地材料副本。

## 当前能力

| 入口 | 执行范围 | 当前能力 |
| --- | --- | --- |
| `ascend.py` | 单 AI Core 的 Cube GEMM 路径 | 容量约束、分块及动作计数；支持显式代理下的串行阶段 timing，配置速率优先 |
| `tpu.py` | 单 TensorCore | MXU → TensorCore → Chip 规格与分层库存；单核逻辑计数；支持带明确假设的串行任务 timing，块内最多 4 MXU 并行 |
| `tsp.py` | 单 hemisphere | 220 MiB 整片 MEM 参考规格；单侧驻留计数；支持显式代理下的行流水 timing，配置参数优先 |
| `tsp_cim.py` | 单 Cluster domain | 400 Macro、K320×N640 域计数；支持带明确假设的串行任务 timing；整片库存单独报告 |

四个入口默认 timing，同时输出动作计数和任务时间，共用 workload 精度选择规则。当前参考配置不提供可直接比较的整片执行时间，能耗和面积总量未评估；规格峰值与真实有效吞吐分开。原生 TSP 与 CIM 比较调用同一个参考实现。

四个仿真器的现有参考配置均可运行带明确假设的局部任务 timing。原始硬件 unknown 保留；Ascend/原生 TSP 优先使用配置中的服务率和延迟，只为缺项选择代理，并在报告中逐项披露。

TPU 报告 schema 2 区分整片硬件规格与局部计数：`hardware_hierarchy` 展示 2 TensorCore、每核 4 MXU；`hardware_inventory.per_tensorcore` 与 `.chip` 分别列出每核和整片物理库存。`boundary.hardware_scope=chip`、`counting_scope=single_tensorcore`、`chip_execution_included=false`；单核计数不乘核数，私有 VMEM 不合并成共享容量。硬件配置仍为 schema 1；字段迁移与字符框图见 [TPU 说明](ARCHITECTURE.md#tpu)。

## 安装

使用 Python 3.12+；已有环境验证版本为 3.12.10。仿真、测试和验收脚本仅使用 Python 标准库，运行于 CPU，无需安装第三方包。[requirements.txt](requirements.txt) 保留为无第三方依赖的说明清单。

可直接使用已有 Python，也可建立并激活虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux/macOS 对应使用 `python3 -m venv .venv` 和 `source .venv/bin/activate`。Windows 重定向输出如遇编码错误，先执行 `$env:PYTHONUTF8 = "1"`。

## 快速运行

在工程根目录执行。仅保留默认负载 `configs/workload/tiny_gemm.json`，M/K/N 为 512/256/512，`data_format=INT8`，采用 INT32 累加/输出。四个入口默认使用此文件并计算任务时间，无需指定负载、覆盖精度或添加 `--mode timing`。仍可通过 `--workload` 指定自定义 GEMM 配置。

在 PowerShell 中一次运行全部四个仿真器：

```powershell
.\scripts\run_all.ps1
```

脚本依次运行 Ascend、TPU、原生 TSP、TSP-CIM，优先使用本工程 `.venv` 中的 Python，未找到时使用 PATH 中的 `python`，无需手动激活环境。从其他目录调用也会将报告写到本工程的 `results/`。每次成功运行覆盖对应报告；某个入口失败时停止，后续入口不执行，未更新的旧报告仍保留。清除历史报告使用下方独立清理指令。

也可以单独运行某个入口：

```powershell
python ascend.py
python tpu.py
python tsp.py
python tsp_cim.py
```

直接运行以上四条命令会生成四份包含任务、操作数、搬运量、总周期、时间、阶段分解、规格来源及假设的 JSON 报告；能耗和实际面积仍明确标记未评估。只统计工作量时可显式添加 `--mode count`。

省略 `--workload` 使用默认尺寸。Ascend/TPU 的精度选项是 `--format`，TSP/CIM-TSP 是 `--formats`。硬件和 workload 可使用配置文件名或显式路径；选项完整列表见各入口 `--help`。`tsp_cim.py` 已移除无效的 `--verbose`，旧命令应删去该选项；`tsp.py --verbose` 继续可用。

### 导出报告

成功运行后默认保存到**工程根目录**的 `results/ascend.json`、`results/tpu.json`、`results/tsp.json` 或 `results/tsp_cim.json`，并在终端打印 `[Report]` 路径。从其他工作目录启动时，默认位置仍然是本工程的 `results/`。

目录会自动创建，同一仿真器再次成功运行会覆盖对应文件；任务检查或仿真失败时不写入报告。需要保留多次结果时，用 `--output` 指定不同文件名。显式提供的相对路径以当前工作目录为基准，绝对路径直接使用：

```powershell
python ascend.py --mode count --output results/ascend_count.json
python tpu.py --mode count --output results/tpu_count.json
python tsp.py --mode count --output results/tsp_count.json
python tsp_cim.py --mode count --compare-mxm --output results/tsp_cim_count.json
```

四个入口都会自动创建输出父目录，终端输出继续保留。`results/`、`tmp/` 是 Git 忽略的生成目录。JSON 的任务、范围、计数、来源与缺参含义见 [报告口径](ARCHITECTURE.md#conventions)；各架构 schema 保持独立。

### 清除生成的报告

在工程根目录的 PowerShell 中运行：

```powershell
.\scripts\clear_results.ps1
```

清除本工程 `results/` 顶层的所有 `.json` 报告，包括默认报告和此前的 `_timing`、`_count` 报告。保留 `results` 目录、子目录及其他类型文件，不影响配置、源码或 `validation/issue2/` 验收证据。从其他目录调用时，目标仍是脚本所属工程的 `results/`。可加 `-WhatIf` 预览将删除的文件。

### 分块与复用实验

```powershell
python ascend.py --format INT8 --tile 64 128 64 --l1-k 256
python tpu.py --format INT8 --tile 128 128 128
python tsp.py --formats INT8 --tile 320 320 640 --loop-order KNM
python tsp_cim.py --formats INT8 --cluster 1 1 3 --tile-m 32 --compare-loops
python tsp_cim.py --formats INT8 --sweep-m 1 4 16 --compare-mxm
```

CIM-TSP 的 `--tile-m` 是逻辑行组，不保证 buffer 驻留。`--compare-mxm` 报告原生参考计数；两个入口范围不同且缺少可比系统时序，speedup/energy_ratio 为 null。

CIM-TSP 报告的 `timing.compute_occupancy.value` 给出文档规定的单域 INT8 计算占用：`M×ceil(K/320)×ceil(N/640)×8` cycle。它不包含装载、等待、归约和写回。`hardware_derived` 同时报告 10 拍 LW 扫描、160 拍完整 context 轮转预算和整片 20480 B/cycle 分布式 LW 捕获宽度；这些是设计派生量，不是本次任务总时间。字段与适用范围见 [CIM-TSP 说明](ARCHITECTURE.md#cim-tsp)。

### 任务时间估算

Ascend 和原生 TSP 的时间估算也沿用现有入口：

```powershell
python ascend.py
python tsp.py
```

该任务默认分块下，Ascend 为 **97216 cycle，97.216 μs**，原生 TSP 为 **588 cycle，约 0.653 μs**。二者执行范围、数据起止位置和服务近似不同，这两个数字不能直接作为芯片速度排名。Ascend 采用分形服务和 HBM 均分代理；TSP 采用理想计算 II、命令波传播和单任务加法服务代理，保留原有 MEM/stream 约束。

Ascend/TSP 的 `timing.parameters` 记录实际采用的值、单位、来源、条件和 `selection`；`proxy_parameters` 列出本次替代的缺项。`missing_parameters` 仍表示原硬件配置的缺项，不代表本次估算未成功。已有配置数值优先，阶段求和见 `stages_cycles`。时间服务代理不改写 count 报告或硬件参数台账。

TPU 可直接用现有参考配置运行单 TensorCore 时间模型：

```powershell
python tpu.py
```

报告中的 `timing.total_cycles` 是总周期，`time_ns` 是纳秒，`stages_cycles` 分解输入搬运、部分和读取、矩阵计算、阵列填充/排空及输出写回。该命令的默认分块下为 **20016 cycle，约 19.06 μs**。模型采用 1.05 GHz 参考时钟，单核 HBM 带宽暂按整片均分；VMEM 服务率和填充/排空延迟属于显式假设，全部列在 `timing.parameters`、`assumptions`，不修改硬件配置中的 unknown。

默认逻辑块上界仍为 128/128/128，每块只用一个 MXU；`--tile 256 128 256` 可在块内形成四个输出小块，由四个 MXU 并行计算。逻辑块之间和搬运/计算阶段之间串行执行，未模拟 XLA 调度或整片执行。这是可解释的近似时间，不是实测性能或可靠的上下界。公式与手算例见 [TPU 说明](ARCHITECTURE.md#tpu-timing)。

CIM-TSP 也可直接用现有参考配置运行简化任务时间模型：

```powershell
python tsp_cim.py
```

报告默认写入 `results/tsp_cim.json`。查看 `timing.total_cycles`、`time_ns` 和 `stages_cycles`，分别是总周期、纳秒和阶段分解；`timing.parameters`、`assumptions` 说明全部近似。模型将装载、切换、计算、排空、跨 K 部分和读取/累加、写回串行相加，保留四套 context 的复用，不模拟重叠。时间换算沿用原生参考 0.9 GHz；这不是芯片实测。完整公式与范围见 [CIM-TSP 说明](ARCHITECTURE.md#cim-tsp)。

四个入口默认采用 timing；显式 `--mode timing` 与默认行为相同。Ascend 缺 GM 路径服务且缺 HBM 代理依据时，或 TSP 缺 MEM 基础服务率时，仍会明确报错。TPU 若缺时钟、阵列几何、MAC 速率或 HBM 带宽，也会报错；count 不因此补时间。TSP/CIM-TSP 拒绝 `--scope chip`。Macro 局部 timing API 仍要求其自身校准参数，不能把上层串行估算当作 Macro 标定。

## 检查与维护

```powershell
python -m unittest discover -s tests -v
```

按 issue #2 的四类交付物和五条条件执行完整验收：

```powershell
python scripts/verify_issue2.py
```

交付物位置及通过判据见[逐条验收表](ARCHITECTURE.md#validation)，最新结果见 [acceptance.json](validation/issue2/run/acceptance.json)。脚本保存测试明细、四份 count 报告、四份 timing 正例与缺参负例；只有自动检查通过且来源/迁移审阅记录仍匹配当前文件时才退出 0。

当前测试覆盖与最近验证记录集中维护在 [验证状态](ARCHITECTURE.md#validation)，参数来源结论见 [核验结果](ARCHITECTURE.md#source-audit)与[逐项台账](calibration/parameter_source_audit.json)，剩余缺口见 [参数清单](ARCHITECTURE.md#remaining)。修改参数时应同时更新来源及状态，硬件配置路径见 [文件职责表](ARCHITECTURE.md#migration)。旧能耗/面积模型、DSE 和报告 API 的迁移同样记录在那里。

```text
huawei_dfp/
├── README.md / ARCHITECTURE.md       # 使用入口 / 统一技术说明
├── ascend.py / tpu.py                # Ascend / TPU 独立入口
├── tsp.py / tsp_cim.py               # TSP / CIM-TSP 入口
├── configs/hardware/                # 架构参考规格与共享 Macro profile
├── configs/workload/                # 公共 GEMM 任务
├── src/hardware/                    # 规格、校验与派生库存
├── src/core/                        # 按架构组织执行内核
│   ├── ascend/                     # simulator.py + tiling.py
│   ├── tpu/                        # simulator.py + tiling.py
│   ├── tsp/                        # simulator.py + tiling.py
│   ├── tsp_cim/                    # simulator.py + tiling.py
│   └── cim/                        # Macro 聚合 simulator + macro_model
├── src/utils/                       # 按架构前缀命名的 config/report；公共 workload
├── calibration/                     # 参数来源核验台账
├── tests/                           # 与执行内核对应的回归测试
│   ├── ascend/ / tpu/              # Ascend / TPU
│   ├── tsp/ / tsp_cim/             # 原生 TSP / CIM-TSP
│   ├── cim/                        # Macro 与 CIM-TSP 集成
│   └── common/                     # 任务、输出路径、库存和参数来源
├── scripts/verify_issue2.py          # issue #2 可复现验收入口
├── scripts/run_all.ps1               # 一次运行四个仿真器并生成完整报告
├── scripts/clear_results.ps1         # 清除 results 顶层生成的 JSON 报告
└── validation/issue2/               # 审阅记录、逐条件结果与实际输出
```

目录约定：`src/core/<架构>/simulator.py` 负责执行统计，`tiling.py` 负责分块；共享 Macro 实现归入 `src/core/cim/macro_model.py`。硬件规格继续放在 `src/hardware/<架构>.py`，共享字段校验为 `parameters.py`；配置和报告按 `<架构>_config.py`、`<架构>_report.py` 命名，Macro 引用由 `tsp_cim_config.py` 解析。各架构按实际需要提供报告模块，CIM-TSP 入口直接输出 JSON。

通用实验入口 `main.py` 及其 `default_core.json` 场景已移除；当前只保留四个架构入口。硬件/workload JSON 和报告内容保持不变，`--output` 现在默认写入工程 `results/`。旧的内部模块路径与新位置见 [目录迁移表](ARCHITECTURE.md#directory-migration)。
