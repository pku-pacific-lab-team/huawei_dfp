# 多架构 GEMM 仿真工程

本工程提供 Ascend 910、TPU v4、原生 TSP、CIM-TSP 四个独立 INT8 GEMM 分析入口。当前交付以可追溯规格、容量检查、动作计数和硬件库存为主。

技术说明见 [ARCHITECTURE.md](ARCHITECTURE.md)：[四套执行路径字符框图](ARCHITECTURE.md#execution-diagrams)、[四个仿真器结构](ARCHITECTURE.md#simulator-structure)、[公共口径](ARCHITECTURE.md#conventions)、[Ascend](ARCHITECTURE.md#ascend)、[TPU](ARCHITECTURE.md#tpu)、[TSP](ARCHITECTURE.md#tsp)、[CIM Macro](ARCHITECTURE.md#cim-macro)、[CIM-TSP](ARCHITECTURE.md#cim-tsp)、[验证状态](ARCHITECTURE.md#validation)。

首次阅读建议先看[四套框图及粒度对照](ARCHITECTURE.md#execution-diagrams)，按存储边界、搬运、计算与累加、写回及调度约束理解当前实现，再进入各架构的参数和公式章节。

文档维护约定：工程内维护使用说明、现行技术说明、参数来源台账和验收证据。教学方案、汇报 PPT、论文原文、历史计划与整理备份放在工程外资料目录，本机统一使用 `D:\IC_paper\huawei_dfp`。运行和自动验收不依赖这些本地材料副本。


四份配置统一提供 `execution` 区域，显式记录分块、缓冲、固定延迟、共享资源和装载策略。CLI 未覆盖时读取 JSON；报告 `execution_contract` 记录实际策略，`work_counts.logical/issued/internal` 区分有效工作、补齐发出工作与已建模内部动作。完整约定及版本迁移见 [执行配置](ARCHITECTURE.md#execution-contract)。

TPU 默认 tile 为 256/128/256，可使用 4 MXU；双输入槽允许 K 预取，容量不足退回一槽。原生 TSP 固定采用单待用块权重预取；CIM 默认采用后台 LW，两边默认装载策略类别均为 background。各自的权重缓冲、安装和切换机制仍不同。以下对照保持配置边界，不代表真实芯片加速比：

```bash
python tpu.py --schedule resource_pipeline --buffers 2
python tpu.py --schedule serial
python tsp_cim.py --compare-mxm --compare-load-policies
python tsp_cim.py --load-overlap serialized
```

## 当前能力

| 入口 | 执行范围 | 当前能力 |
| --- | --- | --- |
| `ascend.py` | 单 AI Core 的 Cube GEMM 路径 | 容量约束、分块及动作计数；资源/依赖流水 timing，容量允许时双缓冲 |
| `tpu.py` | 单 TensorCore | MXU → TensorCore → Chip 规格与分层库存；单核逻辑计数；支持容量约束 K 预取流水与串行对照，块内最多 4 MXU 并行 |
| `tsp.py` | 单 hemisphere | 220 MiB 整片 MEM 参考规格；单侧驻留计数；支持显式代理下的行流水 timing，配置参数优先 |
| `tsp_cim.py` | 单 hemisphere | 2 Plane、8 Cluster domains、3200 Macro；K/N 静态 wave、MEM 容量检查、共享带宽、默认后台 LW 和行流水；整片库存单独报告 |

四个入口默认 timing，同时输出动作计数和任务时间，共用 workload 精度选择规则。当前参考配置不提供可直接比较的整片执行时间，能耗和面积总量未评估；规格峰值与真实有效吞吐分开。原生 TSP 与 CIM 比较调用同一个参考实现。

四个仿真器的现有参考配置均可运行带明确假设的局部任务 timing。已有代理服务直接写入硬件 JSON，标为 assumed，并在 conditions 中说明取法。TSP 配置只保存固定服务上限和固定延迟，实际分配带宽和逐行耗时由模拟器计算。null 表示尚无本模型可用值，必需服务缺失时明确报错。

TPU 报告 schema 13 区分整片硬件规格与局部计数：`hardware_hierarchy` 展示 2 TensorCore、每核 4 MXU；`hardware_inventory.per_tensorcore` 与 `.chip` 分别列出每核和整片物理库存，SparseCore/SPMEM 已移出当前模型。`boundary.hardware_scope=chip`、`counting_scope=single_tensorcore`、`chip_execution_included=false`；单核计数不乘核数，私有 VMEM 不合并成共享容量。硬件配置现为 schema 12；字段迁移与字符框图见 [TPU 说明](ARCHITECTURE.md#tpu)。

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

### 硬件 JSON 的维护注释

当前活动参数为 Ascend 29、TPU 21、TSP 30、CIM-TSP 10 项。11 项仅用于标称规格或库存展示的数值已按 TPU 的方式移入 [四架构归档](configs/archive/README.md)，运行时不加载归档，也不据此补默认值。硬件/报告版本分别为 **9/12、12/13、9/10、7/10**；旧自定义配置需先移出字段再升级版本，报告不再输出相应库存或标称字段。

四份参考 JSON 的根层 `_documentation` 使用统一中文注释，保持标准 JSON 格式，无需支持 `//` 的解析器：

| 注释分类 | 阅读方式 |
|---|---|
| `hardware_parameters` | 容量、数量、时钟、吞吐、带宽等数值；区分计算、校验、库存及独立服务用途 |
| `execution_policy` / `execution_fields` | `execution` 及其逐字段说明；槽数、分块和固定延迟属于执行策略，其来源文字属于依据 |
| `descriptive_metadata` / `parameter_fields` | 描述、路径依据以及参数内的单位、作用域、来源等字段说明；单位/作用域仍参与校验 |
| `configuration_control` | 单列模型身份、版本、格式限制及实际加载的配置引用，避免误当成纯注释 |

正文统一按“配置控制 → 硬件数值 → 执行策略 → 说明依据”排列。Ascend 的 `output_path` 只记录和报告路径假设，修改文字不会改变实际通路，也不是输出文件名；CIM 的 `native_reference` / `macro_reference` 则会实际加载配置。每份文件的 `maintenance_rules` 列出修改规则。

`_documentation` 是可选的阅读注释；四个 Spec 在严格校验前只剔除这个根层对象，CIM 也处理其引用的原生 TSP 注释。它不进入仿真报告或运行时 `reference_id`，其余未知字段仍报错。增加注释本身不改变数值、策略、来源元数据和 schema 版本；配置无需补注释也能读取，但仍须满足当前归档后的字段约束；来源台账的完整 JSON 哈希仍覆盖注释，因此编辑注释后也需同步台账哈希。

### 服务周期与总时间分解

四架构报告统一拆分原 `timing.stages_cycles`：

- `timing.service_cycles`：可重叠的阶段服务工作量，`by_stage` 之和为该对象的 `total_cycles`，不能直接当作任务耗时。
- `timing.elapsed_cycles`：互斥的总时间分解；有分解时，`by_stage` 之和必须等于 `timing.total_cycles`。

两个对象都用 `status` 区分 `available`、`not_decomposed`、`not_evaluated`。Ascend/TPU 流水提供服务分解，时间分解标 `not_decomposed` 并保留总耗时；TPU serial 两者都提供。TSP/CIM 提供时间分解，服务分解标 `not_decomposed`。count 模式两者均为 `not_evaluated`。不可提供的分项为 null，并非零周期。

统一时间视图最初在报告 Ascend/TPU **11**、TSP/CIM **9** 中引入；归档后的当前版本见上方清单。旧 `service_cycles_sum` 改读 `service_cycles.total_cycles`；硬件配置版本、调度结果与动作计数不变。完整结构、各模型可用性和迁移规则见 [统一时间口径](ARCHITECTURE.md#timing-breakdowns)。

### 分块与复用实验

```powershell
python ascend.py --format INT8 --tile 64 128 64 --l1-k 256
python tpu.py --format INT8 --tile 128 128 128
python tsp.py --formats INT8 --tile 320 320 640 --loop-order KNM
python tsp_cim.py --hemisphere 1 --tile-m 32 --mem-slices 0 1 2 3 --streams 8 --compare-loops
python tsp_cim.py --formats INT8 --sweep-m 1 4 16 --compare-mxm
```

CIM-TSP 固定执行 `single_hem`，`--hemisphere 0/1` 选择半片；`--tile-m` 默认 min(M,320)。完整 A/B/C 按 K16/N32 补齐后检查所选 MEM 容量，超过容量明确拒绝。`--mem-slices`、`--streams` 和 `--reserved-bytes` 控制本侧公共资源。单域执行及其串行计时已删除，CLI 不再接受 `--scope`、`--cluster`；API 不再接受 `scope`、`plane`、`cluster`。调用方删除这些参数，通过 `hemisphere` 选择半片。

`--compare-mxm` 按本次模式和资源分配运行原生 TSP：count 比较工作量，timing 另给 `analytical_cycle_ratio_native_over_cim`。该值是两种解析策略的周期比；原生固定后台预取、CIM 可选后台 LW，缓冲和切换约束各自不同，尚非硬件实测，`speedup/energy_ratio` 仍为 null。原生不支持的循环顺序会明确标记不可比。

半片的 `timing.compute_occupancy.value` 按静态 wave 的最大域占用累加，另以 `domain_cycles_sum` 报告域占用之和；两者都不等于总延迟。`schedule.waves` 报告实际任务、域、LW/流水起止时间及 writer 映射；不足 8 个任务的 wave 不自动补满，更不会将单域周期直接除以 8。

TPU 硬件配置现为 **schema 12**，报告为 **schema 13**。21 项活动硬件参数全部有值：9 项 documented、12 项 assumed。默认 `cmem_mode="tiled"`、`cmem_groups_mn=[4,4]`、`cmem_buffer_slots=2`：CMEM 保存完整 K 的 A/B 面板，最多供 4×4 个 VMEM 输出块复用。容量允许时分配第二个面板槽，预取下一外层块；最后一次 CMEM 读取结束即可释放当前槽，允许装入与尾部计算重叠。HBM 装入和 C 导出共享带宽，CMEM 读写采用独立服务。首次装入及最终导出均计时。整芯片读/写代理约为 2.151 / 1.075 TB/s：将官方 libtpu v4 的 CMEM 单值配置作为读基准，写按 v4i 文献的 2:1 比例归约。两个方向映射仍为 assumed，未在 v4 标定；完整数值与复核证据见 [CMEM 说明](ARCHITECTURE.md#tpu-cmem-proxy)。

```powershell
python tpu.py --cmem-mode tiled --cmem-groups 4 4 --cmem-buffers 2 --output results/tpu_cmem.json
python tpu.py --cmem-mode tiled --cmem-buffers 1 --output results/tpu_cmem_one_slot.json
python tpu.py --cmem-mode bypass --output results/tpu_bypass.json
```

CMEM 容量按核均分，不借用闲置份额。先保证一份面板可放下，再决定是否分配第二槽；不为双缓冲强制缩小面板。单槽也可以在最后一次读取后提前复用。`--buffers` 控制 VMEM，`--cmem-buffers` 控制 CMEM；`--schedule serial` 关闭全部重叠。报告 `memory.cmem` 给出请求/有效槽数、单槽/总分配及回退原因；`timing.cmem_outer_overlap_saved_cycles` 单独比较本次外层重叠与原串行外层装入。`input_transfer_bytes` 是送入 VMEM 的载荷，HBM 读取使用 `hbm_input_read_bytes`。

默认 tiny 仅有一个外层面板，仍需首次装入：tiled 为 9244 cycle，bypass 为 8936；输入读取分别为 256/512 KiB。2048×4096×2048 使用相同分块和当前带宽代理，从关闭外层重叠的 2,044,600 降至 1,995,454 cycle；这是模型结果，非硬件实测。schema 7 自定义配置增加 `execution.cmem_buffer_slots=1` 或 `2` 后完成历史 schema 8，再显式添加 `vector_fp32_exp_cost`、`vector_fp32_rsqrt_cost` 的完整元数据完成 schema 9，最后移除 `scalar_units_per_tensorcore` 完成 schema 10，再移出五项规格参数及 interconnect 完成 schema 11，最后按[归档清单](configs/archive/README.md)移出标称 TOPS 和三项 VPU 几何参数，升级到当前 schema 12；schema 6 还需先补外层分组和模式。其他可能重叠的位置及保留的串行边界见 [CMEM 模型](ARCHITECTURE.md#tpu-cmem-proxy)。

Vector/Nonlinear 参考 TSP 的共享工作量模型：`vector_base_ops_per_cycle=2048`、`vector_int32_add_cost=1`、`vector_fp32_basic_op_cost=1`、`vector_fp32_exp_cost=15`、`vector_fp32_rsqrt_cost=12`，均为 assumed。`TPUSpec.vector_service_cycles()` 支持 INT32 ADD、FP32 ADD/MUL/EXP/RSQRT，按总等价工作量统一取整；2048 次 EXP 对应 15 cycle。EXP/RSQRT 是算法工作量代理，不执行数值运算或模拟完整 Softmax/RMSNorm，推导见 [Vector 服务](ARCHITECTURE.md#tpu-vector-service)。各操作共用同一预算；FMA、DIV、归约仍拒绝。TPU GEMM 跨 K 累加在 MXU 内，Vector API 不参与 GEMM 计时。SMEM 已移除；寄存器容量、ICI 两项、工艺和面积共五项规格及拓扑说明单独保存在 [TPU 归档 JSON](configs/archive/tpu_v4_reference.json)，运行时不加载该文件。Scalar 数量及两级库存已移除，旧执行字段不再接受；历史证据存于 `retired_parameters`。标称 TOPS 和三项 VPU 几何参数也已移入该归档，仅供参考，不再进入活动配置和报告；归档与迁移规则见 [参数检查](ARCHITECTURE.md#tpu-parameter-usage)。

### 任务时间估算

Ascend 和原生 TSP 的时间估算也沿用现有入口：

```powershell
python ascend.py
python tsp.py
```

该任务默认分块下，Ascend 为 **43588 cycle，43.588 μs**；原生 TSP 为 **601 cycle，约 0.668 μs**。二者执行范围、数据起止位置和服务近似不同，这些数字不能直接作为芯片速度排名。Ascend 采用分形服务、L2 输入块缓存、HBM/L2 均分代理及有缓冲生命周期约束的任务调度；TSP 采用理想计算 II、命令波传播和单任务加法服务代理，保留原有 MEM/stream 约束。

```powershell
python ascend.py --buffers 2
python ascend.py --buffers 1
python ascend.py --trace
```

Ascend 仅保留流水调度，旧 `--timing-policy`/API `timing_policy` 参数已移除。`--buffers 2` 保持原分块，在每类缓冲能容纳两份时使用两个槽，否则该缓冲退回一个槽。HBM 读写共享服务，L1→L0A/B 共用 MTE1；计算和独立搬运资源可以重叠，L0C 保留完整 K 归约后才导出。`timing.total_cycles` 是最后任务完成时间；`service_cycles.by_stage` 是服务工作量，不能相加当作总耗时。`service_cycles.total_cycles` 统计服务周期之和，`overlapped_service_cycles` 为其与总耗时的差，另报资源利用率及实际存活数据峰值。`memory.*.peak_bytes` 表示固定槽的预留容量，`peak_payload_bytes` 是单次载荷峰值。`--trace` 仅用于 timing，增加任务起止、依赖和缓冲生命周期；未打开时不输出完整轨迹。

Ascend/TSP 的 `timing.parameters` 保留实际读取的固定配置值，`selection=configured`；Ascend 的 `timing.derived_services.hbm` 报告均分后的 HBM 速率；TSP 的 `timing.derived_services` 单独报告本次分配带宽、LW 空闲峰值、内部 IW 带宽、权重缓冲容量，以及实际尾块的 VXM 行计算/首行时间范围。`proxy_parameters` 列出已配置的 assumed 服务，`missing_parameters` 仅列本任务必需但仍为空的时间参数。修改服务直接改 JSON 的 value，不再使用 derivation。TSP 当前硬件 schema 9、报告 schema 10；此前 schema 3 将 install_first_cycles/vxm_add_first_cycles 分别拆为固定传播/额外流水延迟，不能直接复制旧值。迁移及公式见 [显式模型参数](ARCHITECTURE.md#explicit-model-parameters)。

TSP schema 4 移除 6 项无执行用途的参数：`spare_superlanes`、`physical_sram_bytes`、`local_sram_bytes`、`offchip_capacity_bytes`、`offchip_bandwidth_bpc`、`interchip_bandwidth_bpc`。schema 6 替换旧 add/nonlinear 速率，schema 7 再删除 `mem_shared_bpc`，schema 8 增加有限待用权重缓冲；当前共 30 项且无 null，18 项 documented、12 项 assumed。共享 VXM 预算为 320 op/cycle，INT32 add/FP32 add/mul 代价为 1，exp 为 15、rsqrt 为 12。`TSPSpec.vxm_service_cycles()` 支持独立计算占用估算；原生/CIM GEMM 只消费 INT32 add。特殊函数值是算法工作量代理，不是原生指令时序。迁移及 API 示例见 [共享 VXM 服务](ARCHITECTURE.md#tsp-vxm-service)。自定义 schema 3 配置先删除上述 6 项完成历史 schema 4，再按[迁移说明](ARCHITECTURE.md#tsp-parameters)逐步迁移到当前 schema 9；更早版本还需先迁移固定服务字段。原生库存同步移除备用数量和物理/局部 SRAM 汇总，该次六项精简保持 MEM 容量与 GEMM 公式不变；当前读写并发公式见下文。CIM 共享来源不再转录三项外部系统参数，CIM 自身库存的后续精简见下文；历史配置和来源保存在台账 `retired_parameters`。具体迁移见 [TSP 参数说明](ARCHITECTURE.md#tsp-parameters)。

Ascend 硬件配置为 **schema 9**，报告为 **schema 12**。HBM 容量为 32,000,000,000 B，带宽为 1.2e12 B/s；单核均分后为 37.5 B/cycle。L2 使用 32 MiB 全芯片容量和已选的 **3 TB/s**（1 GHz 下 3000 B/cycle），均为 `assumed`，单核份额为 **1 MiB、93.75 B/cycle**。A/B 输入采用冷启动的整块 LRU；缺失依次经过 `hbm_to_l2 → l2_fill → l2_to_l1`，命中只走 L2→L1。C 输出和超过单核容量的输入块绕过缓存。报告分别统计 HBM 缺失读取、L2 填充/读取、输入请求及命中率；当前配置无 null 参数。策略、来源与局限见 [L2 说明](ARCHITECTURE.md#ascend-l2-bandwidth)，旧配置须按 [迁移说明](ARCHITECTURE.md#migration) 补全 execution.l2 和资源映射。缓存行、组相联、bank/控制器细节及真实硬件仲裁仍未模拟；3 TB/s 不是实测持续带宽。

Ascend 新增与 TSP 同口径的共享 Vector/Nonlinear 计算服务：每核预算 64 op/cycle，INT32 ADD 与 FP32 ADD/MUL 代价为 1，FP32 EXP 为 15、RSQRT 为 12；新增 5 项均为 assumed，现有 29 项数值硬件参数均非 null。`AscendSpec.vector_service_cycles()` 提供独立计算占用估算，GEMM 的 Cube/L0C 累加不调用它。64 的每周期假设尚未标定，Exp/Rsqrt 是算法工作量代理；没有独立 SFU，也未实现完整 Softmax/RMSNorm 的搬运、归约和依赖调度。旧 schema 7 须补齐五项配置完成 schema 8，再移出 Cube/Vector 单元数量，升级到当前 schema 9。来源、公式和 API 示例见 [Ascend Vector 服务](ARCHITECTURE.md#ascend-vector-service)。

TPU 可直接用现有参考配置运行单 TensorCore 时间模型：

```powershell
python tpu.py
```

报告中的 `timing.total_cycles` 是总周期，`time_ns` 是纳秒，`service_cycles.by_stage` 记录输入搬运、部分和读取、矩阵计算、阵列填充/排空及输出写回的服务周期；流水的 `elapsed_cycles` 保留总耗时并标记 `not_decomposed`。该命令默认启用 CMEM，为 **9244 cycle，约 8.80 μs**；`--cmem-mode bypass` 为 8936 cycle，约 8.51 μs。模型采用 1.05 GHz 参考时钟，单核 HBM 带宽暂按整片均分；VMEM 服务率已在硬件 JSON 中显式配置为 1075200000000 B/s、status=assumed；填充/排空等阶段假设继续列在 `timing.parameters`、`assumptions`。

默认逻辑块上界为 256/128/256，完整块使用四个 MXU；容量允许时使用两个输入槽，使下一段 K 的 DMA 与当前计算重叠。输出块之间仍串行，未模拟 XLA 调度或整片执行。这是可解释的近似时间，不是实测性能或可靠的上下界。公式与手算例见 [TPU 说明](ARCHITECTURE.md#tpu-timing)。

CIM-TSP 也可直接用现有参考配置运行简化任务时间模型：

```powershell
python tsp_cim.py
```

报告默认写入 `results/tsp_cim.json`。`timing.elapsed_cycles.by_stage` 为不重叠的 `load_wait + row_pipeline`，后台 LW 不重复相加；`timing.resources` 报告 MEM 与方向 stream 的预算和最高预约速率。默认任务为 **4307 cycle，约 4.786 μs**，仅有一个 K/N 分片，不能占满 8 个域。`M1/K320/N5120` 使用全部 8 个域，为 **202 cycle**；两者均为当前假设下的分析结果。完整公式见 [CIM-TSP 说明](ARCHITECTURE.md#cim-tsp)。

四个入口默认采用 timing；显式 `--mode timing` 与默认行为相同。Ascend 缺 Cube、核内搬运服务、HBM 带宽或时钟时，或 TSP 缺本次任务必需服务时，明确报错；Ascend count/timing 均检查 HBM 可用容量。TPU 若缺时钟、阵列几何、MAC 速率或 HBM/VMEM 带宽，也会报错；count 不因此补时间。TSP 拒绝 `--scope chip`；CIM-TSP 仅提供半片执行，不提供 scope 选项。Macro 只提供动作计数，任务时间由上层执行模型估算；其显式假设不代表 Macro 物理时序已经标定。

CIM 参数精简历史：Macro profile 升至 **schema 2**，删除 `compute_first_cycles`、`write_first_cycles`、`write_wordline_ii_cycles`、`context_switch_cycles`；Macro API 不再接受 `mode`，直接调用即计数。当时 CIM 硬件升至 **schema 3**，删除 `physical_sram_bytes`、`additional_local_sram_bytes`、`collector_buffer_bytes`；报告升至 **schema 4**，同步移除相应库存和四项无消费路径的实体/端点/domain 占位。旧配置需删字段并升级版本，历史依据保存在来源台账 `retired_parameters`。该次精简曾保留原生 `nonlinear_ops_per_cycle`；在原生 schema 6 的 VXM 迁移时被共享操作代价替换，CIM 同步来源与服务，当时报告升至 schema 7。该次精简保持动作和旧 Cluster 时间公式不变；已知 SRAM 仍只是小计，collector 容量仍未验证。具体迁移及后续 Issue #3–#6 的边界见 [CIM 说明](ARCHITECTURE.md#cim-tsp)。

CIM 当前硬件为 **schema 7**、报告为 **schema 10**，Macro profile 仍为 2。自定义硬件需添加 `execution`，将 `nim_output_bpc` 改名为 `int32_writer_bpc` 并标 assumed，scope 使用 `writer_superlane_plane`；数值仍为 256。该口径依据详稿 §3.4.6 的分布式 writer，不扩大 FP 转换 lane 库存，聚合吞吐仍受共享带宽限制。这些是早期迁移步骤；迁移到当前 schema 7 还须采用独立 MEM 读写规则，并按[归档清单](configs/archive/README.md)移出四项 NIM 库存字段。旧 schema 明确拒绝。

TSP 和 CIM 当前采用**简化 MEM 读写并发模型**：每 slice 读、写分别为 320 B/cycle，假设 bank 布局无冲突；同方向动作仍共享预算，后台 LW 与激活/旧部分和读取竞争读带宽。删除合计上限 `mem_shared_bpc`，不新增 640 B/cycle 单向服务。原生 schema 6→7、CIM schema 5→6 的配置必须显式增加 `execution.rules.mem_read_write="independent_assumed_conflict_free"`，报告分别升至 7/8；旧配置拒绝。计数和容量不变，MEM 受限场景的周期可能减少；该次迁移时 tiny 为原生 588、CIM 4307 cycle；后续权重预取拆分后原生为 601 cycle。公式和边界见 [MEM 并发模型](ARCHITECTURE.md#tsp-mem-concurrency)。

## 检查与维护

在工程根目录执行，使用 `-t .` 指定测试导入的顶层目录，避免 `tests/tpu` 等测试包与根目录的同名入口模块冲突：

```powershell
python -m unittest discover -s tests -t . -v
```

按 issue #2 的四类交付物和五条条件执行完整验收：

```powershell
python scripts/verify_issue2.py
```

交付物位置及通过判据见[逐条验收表](ARCHITECTURE.md#validation)，当前完整验收统一使用 [acceptance.json](validation/issue2/run/acceptance.json)。脚本保存测试明细、四份 count 报告、四份 timing 正例与缺参负例；只有自动检查通过且来源/迁移审阅记录仍匹配当前文件时才退出 0。查看结果时须核对文件快照，不能把历史通过状态用于已修改的工程。

历史验证、迁移基线和一次性脚本已集中归档，位置与保留规则见 [验证记录说明](validation/README.md)。自动检查通过后，A1/A5 仍可能因语义审阅快照过期而显示 `needs_review`；清理文件或重新运行脚本不会代替审阅。

当前测试覆盖与最近验证记录集中维护在 [验证状态](ARCHITECTURE.md#validation)，参数来源结论见 [核验结果](ARCHITECTURE.md#source-audit)与[逐项台账](calibration/parameter_source_audit.json)，剩余缺口见 [参数清单](ARCHITECTURE.md#remaining)。修改参数时应同时更新来源及状态，硬件配置路径见 [文件职责表](ARCHITECTURE.md#migration)。旧能耗/面积模型、DSE 和报告 API 的迁移同样记录在那里。

```text
huawei_dfp/
├── README.md / ARCHITECTURE.md       # 使用入口 / 统一技术说明
├── ascend.py / tpu.py                # Ascend / TPU 独立入口
├── tsp.py / tsp_cim.py               # TSP / CIM-TSP 入口
├── configs/hardware/                # 架构参考规格与共享 Macro profile
├── configs/archive/                 # 已退出运行配置的硬件规格与来源
├── configs/workload/                # 公共 GEMM 任务
├── src/hardware/                    # 规格、校验与派生库存
├── src/core/                        # 按架构组织执行内核
│   ├── ascend/                     # 分块、L2 缓存与流水调度
│   ├── tpu/                        # 分块、CMEM 与两级预取流水
│   ├── tsp/                        # 分块与权重预取流水
│   ├── tsp_cim/                    # 半芯片 wave 调度与资源约束
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
└── validation/                      # 当前验收与历史材料索引
    ├── issue2/                     # 审阅依据及 run/ 当前完整验收
    └── archive/                    # 历史验证、基线与过程材料归档
```

目录约定：`src/core/<架构>/simulator.py` 负责执行统计；Ascend、TPU、原生 TSP 的 `tiling.py` 负责分块，CIM-TSP 的半芯片映射和调度位于 `hemisphere.py`；共享 Macro 实现归入 `src/core/cim/macro_model.py`。硬件规格继续放在 `src/hardware/<架构>.py`，共享字段校验为 `parameters.py`；配置和报告按 `<架构>_config.py`、`<架构>_report.py` 命名，Macro 引用由 `tsp_cim_config.py` 解析。各架构按实际需要提供报告模块，CIM-TSP 入口直接输出 JSON。

通用实验入口 `main.py` 及其 `default_core.json` 场景已移除；当前只保留四个架构入口。硬件/workload JSON 和报告内容保持不变，`--output` 现在默认写入工程 `results/`。旧的内部模块路径与新位置见 [目录迁移表](ARCHITECTURE.md#directory-migration)。

原生 TSP 现使用 [有限权重预取模型](ARCHITECTURE.md#tsp-weight-prefetch)：LW 利用当前行流水剩余的 MEM 读和同方向 stream 带宽，IW 在旧 tile 排空后通过内部端口安装权重。仅保留一个待用块，假设缓冲容量为 204800 byte（200 KiB）；超过容量明确拒绝。30 项参考参数全部有值，18 documented、12 assumed。缓冲容量和保守 IW 时序均为建模假设，公开论文仅支持机制；没有保留旧串行运行选项。

报告以 `timing.elapsed_cycles.by_stage` 中的 `weight_load_stall + install + row_service` 分解总周期，另列已隐藏的 `weight_load_overlap_cycles` 和各 tile/LW 段。首次装载不会隐藏，内部 IW 不再次读取 MEM。默认 tiny 为 **601 cycle**，原有动作计数和 MEM 驻留容量保持不变。自定义 schema 7 配置先补全权重缓冲参数及预取策略完成 schema 8，再移出 `nominal_int8_tops`，升级到当前 schema 9。
