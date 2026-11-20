# 多架构算子仿真工程

本工程提供 Ascend 910、TPU v4、原生 TSP、CIM-TSP 四个独立分析入口，支持 INT8 GEMM、独立 INT32/FP32 Vector/Nonlinear、FP32 RMSNorm/Softmax，以及 Residual、SwiGLU/FFN、Decode Attention/GQA 和 KV Cache 复合块，以及 INT8 RMSNorm、Q/K/V、Attention 输出投影、RoPE、Embedding 和 FP32 词表输出投影。当前交付以可追溯规格、容量检查、动作计数和局部解析时间为主。

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

四个入口默认 timing，同时输出动作计数和任务时间。GEMM 共用 workload 精度选择规则；Vector/复合块在输入 Tensor 中声明 dtype。当前参考配置不提供可直接比较的整片执行时间，能耗和面积总量未评估；规格峰值与真实有效吞吐分开。原生 TSP 与 CIM 比较调用同一个参考实现。

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

在工程根目录执行。默认 GEMM 负载为 `configs/workload/tiny_gemm.json`，M/K/N 为 512/256/512，`data_format=INT8`，采用 INT32 累加/输出。四个入口默认使用此文件并计算任务时间，无需指定负载、覆盖精度或添加 `--mode timing`。可通过 `--workload` 指定自定义配置；其他算子示例在 `configs/workload/operators/`。

在 PowerShell 中按范围一次运行仿真与验证：

```powershell
.\scripts\run_all.ps1 -Scope issue2  # 4 架构原生 GEMM + Issue #2 验证
.\scripts\run_all.ps1 -Scope issue3  # 4 架构 × 22 类算子 + Issue #3 验证
.\scripts\run_all.ps1 -Scope all     # 两组全部执行，并核对全仓库测试覆盖
.\scripts\run_all.ps1                # 默认等同于 -Scope all
```

| 范围 | 日常 timing 报告 | 独立验证 |
| --- | --- | --- |
| `issue2` | 4 份原生 GEMM，`results/<架构>.json` | 原生测试及 17 项 CLI 检查，`validation/issue2/run/` |
| `issue3` | 88 份：统一 GEMM 报告及 21 个 Vector/复合块样例 | 算子/验证工具测试及 184 项 count/timing/负例 CLI 检查，`validation/issue3/run/` |
| `all` | 两组共 92 份，分别保存 | 两套判据分别汇总；核对单元测试集合并集覆盖全仓库且无重复 |

Issue #3 另输出每算子一行的 `results/run_all_<范围>_operators.csv`；可加 `-Details` 保留独立详细文件。Issue #3 的 GEMM 适配报告使用 `results/<架构>_gemm_operator.json`，其他算子使用 `<架构>_<算子>.json`。每个范围另生成 `results/run_all_<范围>.json` 总结和 `results/run_all/<范围>/` 命令日志，均以脚本所属工程为根目录。Python 优先使用工程 `.venv`，否则使用 PATH。Linux/macOS 可运行等价命令 `python scripts/run_all.py --scope all`。

某一步失败后会继续尝试其余选中步骤，最后统一返回：`0`=所选范围全部通过，`1`=执行/验证/覆盖检查失败，`2`=自动检查通过但 Issue #2 的 A1/A5 本地审阅证据需对齐（不等同于 GitHub PR 待审）。该状态不会阻止 Issue #3 执行，也不会被改成通过。旧报告保留时会标记本次未生成新报告，不能算作成功。`all` 指当前已登记的实现与测试，不表示尚未实现的 Issue #3 复合块已完成。

日常报告重跑成功后覆盖；只想快速生成单个仿真报告时，使用下方架构入口，无需运行全部测试。

也可以单独运行某个入口：

```powershell
python ascend.py
python tpu.py
python tsp.py
python tsp_cim.py
```

直接运行以上四条命令会生成四份包含任务、操作数、搬运量、总周期、时间、阶段分解、规格来源及假设的 JSON 报告；能耗和实际面积仍明确标记未评估。只统计工作量时可显式添加 `--mode count`。

省略 `--workload` 使用默认尺寸。Ascend/TPU 的精度选项是 `--format`，TSP/CIM-TSP 是 `--formats`。硬件和 workload 可使用配置文件名或显式路径；选项完整列表见各入口 `--help`。`tsp_cim.py` 已移除无效的 `--verbose`，旧命令应删去该选项；`tsp.py --verbose` 继续可用。

### 算子结果接口（Issue #3 第一批）

四个入口新增可选 `--report-format operator`，将本次 GEMM 的原生结果转换为统一的 `OpResult` JSON。默认 `legacy` 保持现有报告格式和终端输出，硬件配置、分块、计数及计时不变。例如：

```powershell
python tpu.py --report-format operator --output results/tpu_operator.json
python tsp_cim.py --mode count --report-format operator --output results/tsp_cim_operator.json
```

统一报告使用独立 **schema 2**：保留算子输入输出、`operator.attributes`、顶层 `latency_s` / `cycle_count` / `timing_status`、容量需求、执行边界和一份 `action_counts`。count 模式的时间为 null；缺时钟但可计算周期时为 `cycles_only`。能耗保持未评估。容量按各资源的 scope 和 requirement_basis 分列，不能跨层相加；动作保留单位、dtype、basis 和 role，reference/summary 不能作为新增工作量重复累加。

默认不保留或导出完整原生报告。需要分块、调度、来源或完整连接描述时，显式指定另一个输出文件：

```powershell
python ascend.py --workload operators/fp32_add.json --details-output results/details/ascend_fp32_add.json
python tpu.py --report-format operator --details-output results/details/tpu_gemm.json
.\scripts\run_all.ps1 -Scope issue3 -Details
```

详细文件独立使用 `kind=operator_details`，包含 `native_report`、`io_contract` 和动作来源，以 `result_sha256` 绑定精简结果。两个文件路径不得相同；`--output ""` 仍可关闭摘要文件。GEMM 的 trace/比较/扫描选项配合 operator 格式时必须同时指定详细输出。原生 legacy 报告保持原格式。

批量运行自动生成 `results/run_all_<范围>_operators.csv`，每个 Issue #3 架构与算子一行，列出形状、周期、延迟、各资源容量和动作计数。默认不生成详细文件；`-Details` 才额外写入 `results/details/`，不影响 Issue #2 输出。

Python API 可直接复用已有硬件 Spec：

```python
from src.operators import Gemm, TensorSpec, simulate_gemm
from src.utils.tpu_config import TPUConfigLoader

task = Gemm.from_tensors(TensorSpec("A", (1, 17), "INT8"),
                         TensorSpec("B", (17, 33), "INT8"))
result = simulate_gemm(task, "tpu", TPUConfigLoader().load_hardware(), mode="timing")
report = result.to_dict()  # compact schema 2
print(result.latency_s, result.cycle_count)
```

第一批提供 INT8 GEMM → INT32 输出、公共类型和计数接口；第二批补充独立 Vector，第三批接入 RMSNorm/Softmax，第四批接入 Residual、SwiGLU/FFN 和粗粒度 INT8 存储边界工作量，第五批接入 Decode Attention/GQA 与 KV Cache，第六批 A 补齐 INT8 RMSNorm、Q/K/V 和 Attention 输出投影，第六批 B 补齐 RoPE、Embedding 和 FP32 词表输出边界，第七批 A 接入 MLA 低秩前处理与共享压缩缓存，第七批 B 补齐 absorbed MLA Attention 和输出投影集成。Tensor 能描述 FP32 等存储类型，不表示这些 GEMM 输入格式已可执行。MoE 等块及整片执行仍待实现；完整模型拼装属于 Issue #4，能耗计算另行接入。容量不足和缺少必需服务继续报错；TSP/CIM 的片外流量标记为执行边界之外。接口、计数依据和已知限制见[算子契约](ARCHITECTURE.md#operator-contract)。

`Gemm.from_tensors()` 保留调用方的 Tensor 名称；GEMM、Vector、RMSNorm、Softmax 的 Python API 均可用 `output_name=` 指定输出。未指定时使用默认名称并避开输入名，显式输入输出重名会拒绝。Vector/复合 workload 同样接受可选 `output_name`。名称只在一次算子调用内标识张量，后续图层须用节点 ID 限定，不能以同名推断共用存储。

RMSNorm/Softmax 的规范化正轴 `axis`、RMSNorm 的 `epsilon` 保留在统一 `operator.attributes`；属性使用不可变标量键值对，导出为 JSON 对象。算子描述包含计算语义与 Tensor 身份，但不包含硬件、分块和参数来源，不能单独作为仿真结果缓存键。

`plan_connection(producer_result, consumer_result, output_index=0, input_index=0)` 检查一个拟连接边：形状不符直接拒绝；dtype 不同列出显式转换需求及两端逻辑字节数；布局、设备和驻留尚待验证时明确列出要求。它仅规划边界，不执行计算图，不从独立算子的账本中扣除流量；转换/传输的时间与新增动作均为 null，不能当成零。例如 GEMM 的 INT32 输出接 FP32 RMSNorm 时，转换方法、比例与舍入策略尚未建模，即使两端均为 4 B/元素也不能直接视为兼容。

```python
from src.operators import RMSNorm, plan_connection

norm = RMSNorm(TensorSpec("projection_fp32", (1, 33), "FP32"),
               TensorSpec("weight", (33,), "FP32"), output_name="normalized")
from src.operators import simulate_composite
consumer = simulate_composite(norm, "tpu", TPUConfigLoader().load_hardware())
edge = plan_connection(result, consumer)  # result 为上例 GEMM 的 OpResult
assert "explicit_conversion_model" in edge["requirements"]
```

### 独立 Vector / Nonlinear（Issue #3 第二批）

四个入口通过 `operation="vector"` workload 独立执行 INT32 ADD、FP32 ADD/MUL/EXP/RSQRT。一元操作一个输入，二元操作两个同形状、同精度输入；不进行广播、原地覆盖或数值计算。示例位于 `configs/workload/operators/`，均含 1025 个元素以覆盖尾块。

```powershell
python ascend.py --workload operators/fp32_add.json
python tpu.py --workload operators/fp32_exp.json
python tsp.py --workload operators/int32_add.json --mem-slices 0 1 --streams 2
python tsp_cim.py --workload operators/fp32_rsqrt.json --hemisphere 1
```

Vector 默认 timing，`--mode count` 只计工作量。输出固定使用统一 operator report，可省略 `--report-format` 或显式选 `operator`，显式 `legacy` 会报错。`--vector-tile-elements` 覆盖块上限，Ascend/TPU 的 UB/VMEM 不足时自动缩小块；HBM 或 TSP 本侧完整输入输出容量不足会拒绝任务。`--format(s)`、GEMM tile、buffers、schedule、CMEM、loop-order、trace、比较与扫描选项不适用于 Vector，显式提供会报错。GEMM 任务不接受 Vector 专用选项。

独立配置 [configs/execution/vector.json](configs/execution/vector.json) 当前使用 schema 2（schema 1 仍可用于独立 Vector），记录默认 1024 元素块、逐块逐阶段串行、显式零启动开销以及 Ascend UB 读写各 256 B/cycle 的未标定代理。后者是每周期 64 个 32-bit 字的建模假设，不是实测带宽。可用 `--vector-config` 指定替代配置；缺文件不回退。既有五类计算服务、HBM/VMEM/MEM/stream 参数继续来自所选硬件配置，GEMM 配置及 schema 无需迁移。新配置来源与哈希登记于独立[执行参数台账](calibration/operator_execution_audit.json)的 `vector_execution_profiles`；硬件规格台账不包含这些算子策略。

```python
from src.operators import TensorSpec, Vector, simulate_vector
from src.utils.tpu_config import TPUConfigLoader

task = Vector("fp32_exp", (TensorSpec("x", (1025,), "FP32"),))
report = simulate_vector(task, "tpu", TPUConfigLoader().load_hardware(),
                         mode="timing", tile_elements=80).to_dict()
```

EXP/RSQRT 复用现有等价工作量代理，仍与基础操作共享一个计算预算。内存搬运和计算均进入时间，count 不要求带宽或计算速率有值。TSP 缺时钟时可以输出周期和 `cycles_only`；Ascend/TPU 的 byte/s 服务换算需要时钟。CIM Vector 调用原生 VXM 路径，不产生 Macro MAC/编程动作。公式、路径及限制见[独立 Vector 模型](ARCHITECTURE.md#vector-operators)。

### RMSNorm / Softmax（Issue #3 第三批）

四个入口支持 `operation="rmsnorm"` 或 `"softmax"` 的 schema 1 workload，输入固定 FP32，沿最后一个轴归约，前导维度合并为独立行。RMSNorm 的 inputs 顺序为 x、形状 `[width]` 的 weight，epsilon 默认 `1e-5` 且必须为有限正数；Softmax 只有 x 输入，采用减行最大值的稳定公式。不执行真实数值计算；不包含 mask、量化转换或任意轴布局转换。

```powershell
python ascend.py --workload operators/rmsnorm.json
python tpu.py --workload operators/softmax.json --vector-tile-elements 80
python tsp.py --workload operators/rmsnorm.json --mem-slices 0 1 --streams 2
python tsp_cim.py --workload operators/softmax.json --hemisphere 1
```

复合块沿用 `--mode count|timing`、`--vector-config`、`--vector-tile-elements` 和统一 operator report；GEMM 专用选项继续拒绝。schema 2 profile 的 `composite` 区域显式记录 SUB/MAX/RCP 代价 1/1/7 和每轮归约至少 1 cycle 的未标定假设。schema 2 的该区域缺失、null 或非完整字典都会在加载时拒绝，包括只运行独立 Vector 的情况。旧 schema 1 不足以运行复合块；迁移须完整添加该区域的策略、四个参数及来源元数据。

```python
from src.operators import RMSNorm, Softmax, TensorSpec, simulate_composite
from src.utils.tpu_config import TPUConfigLoader

op = RMSNorm(TensorSpec("x", (2, 8192), "FP32"),
             TensorSpec("weight", (8192,), "FP32"), epsilon=1e-5)
report = simulate_composite(op, "tpu", TPUConfigLoader().load_hardware(),
                            mode="timing", tile_elements=1024).to_dict()
```

RMSNorm 两遍读取输入，Softmax 用输出区暂存指数值再归一化。块内树形归约、跨块合并、行标量广播和中间存储访问均进入模型；不会将独立 Vector 的完整 HBM 往返直接拼接。超过 UB/VMEM 的行可分块，TSP/CIM 仍要求完整输入输出及块内临时区驻留 MEM。详见[复合块公式与容量边界](ARCHITECTURE.md#normalization-operators)。

每次复合块执行还会检查符号缓冲生命周期，结果在 `memory.lifetime_validation`：包括初始化后才读取、局部槽覆盖顺序、奇数归约存活值、完整指数行写完后才归一化，以及临时容量。重复块最多检查首轮和稳定轮，检查量不随 batch 或块数线性增长。这不是物理地址、bank 或寄存器级验证。

### Residual / SwiGLU / FFN（Issue #3 第四批）

四个入口支持三种 schema 1 workload：`residual` 接受两个同形状 INT8 或 FP32 输入，输出保持相同存储类型；`swiglu` 接受两个同形状 INT8、INT32 或 FP32 输入，内部使用 FP32，输出默认为 INT8，可指定 `output_dtype="FP32"`；`swiglu_ffn` 输入顺序为 INT8 的 `x[M,H]`、`gate_weight[H,I]`、`up_weight[H,I]`、`down_weight[I,H]`，输出为 INT8 `[M,H]`。当前 FFN 不含 bias 或外部残差连接，Residual 是单独的块。

```powershell
python ascend.py --workload operators/residual.json
python tpu.py --workload operators/swiglu.json --vector-tile-elements 80
python tsp.py --workload operators/swiglu_ffn.json --mem-slices 0 1 --streams 2
python tsp_cim.py --workload operators/swiglu_ffn.json --hemisphere 1 --details-output results/details/tsp_cim_ffn.json
```

Python API 使用 `Residual`、`SwiGLU`、`SwiGLUFFN` 与 `simulate_block(op, architecture, spec, mode="timing")`。三个 GEMM 复用原生内核，FFN 按 gate → up → SwiGLU → down → INT8 输出依次执行。每个子块在 HBM（Ascend/TPU）或本侧 MEM（TSP/CIM）中物化结果；父块保留全部子块读写，不额外加收“连接流量”。输入与权重在整块期间常驻，中间张量在最后一次使用后释放，容量按同时存活峰值检查。

独立 [blocks.json](configs/execution/blocks.json) 保存串行组合、符号补齐布局和未标定的类型边界代价，可用 `--block-config` 替换；`--vector-config` 继续控制所复用的 Vector 服务，`--vector-tile-elements` 控制点算 tile。INT32→FP32→INT8 不再被当作免费连接，但这里仅有块级工作量代理，不实现独立量化/反量化指令、量化数值或物理布局转换。详见[公式、生命周期与执行边界](ARCHITECTURE.md#decode-blocks)。

执行范围仍为单 AI Core、单 TensorCore 或单半片，`chip_execution_included=false`。尚未实现整片 M/N 分片、权重/激活复制与跨核通信，不能直接把单核时间除以库存核数。

### Decode Attention / GQA 与 KV Cache（Issue #3 第五批）

四个入口新增 `kv_cache` 和 `decode_attention`，支持单 token、统一历史长度的 Decode。Query 为 INT8 `[B,Hq,1,D]`；预分配 K/V 缓存各为 INT8 `[B,Hkv,C,D]`，要求 `Hq` 是 `Hkv` 的整数倍，覆盖 MHA、GQA 和 MQA。`length` 是本次执行前的有效 token 数，`capacity` 是预留 token 数；`append_tokens` 可为 0 或 1，缓存满时拒绝追加。长度可以为零，此时首次追加后才能做 Attention。

```powershell
python ascend.py --workload operators/kv_cache.json
python tpu.py --workload operators/decode_attention.json --details-output results/details/tpu_decode_attention.json
python tsp.py --workload operators/decode_attention.json --mem-slices 0 1 --streams 2
python tsp_cim.py --workload operators/decode_attention.json --hemisphere 1
```

```python
from src.operators import TensorSpec, KVCacheSpec, KVCacheAccess, DecodeAttention, simulate_attention
from src.utils.tpu_config import TPUConfigLoader

cache = KVCacheSpec(batch=2, kv_heads=2, head_dim=7, length=5, capacity=9)
op = DecodeAttention(TensorSpec("query", (2, 4, 1, 7), "INT8"), cache)
result = simulate_attention(op, "tpu", TPUConfigLoader().load_hardware(), mode="timing")
next_cache = op.cache_access.cache_after  # length=6；只推进不可变描述，不保存真实 K/V 值
```

本块接受已经投影、位置处理后的 Q 和新 K/V，执行缓存追加、`QKᵀ → 1/sqrt(D) 缩放 → Softmax → PV → INT8 context`；Q/K/V 与输出投影由第六批 A 提供独立 block，RoPE 由第六批 B 提供独立 block，尚未组装完整 Attention 层。新 K/V 输入由缓存形状生成对应 Tensor 描述，追加只写本轮 token。`kv_cache` 独立入口默认追加后读取有效前缀一次；Attention 内部仅调用追加阶段，K/V 读取由两个原生 GEMM 负责，不额外再收一次缓存读取。

缓存容量按 `B*Hkv` 份 K/V head 分配，GQA 不创建复制后的 K/V 张量。当前每个 Query head 按原生 GEMM 冷启动策略读取 K/V，且完整执行六个阶段后才处理下一 head；多个 head 共享一个临时工作区，计数与时间以重复模板压缩汇总。`--attention-config` 指定独立 [attention.json](configs/execution/attention.json)，`--vector-config`、`--block-config` 分别控制复用的 Vector/Softmax 与存储边界成本；单独 `kv_cache` 不接受无计算用途的 `--block-config`。

缓存状态输出与原缓存共享 backing，详细报告的 `cache_state` 和 `io_contract.state_aliases` 明确记录长度变化、追加区间和别名；不会为“更新后的缓存”重复申请整块空间。容量包含预留的未填充位置及原生补齐，读写量只计实际执行的访问。形状视图和转置仍是符号布局假设，无物理布局转换指令、数值计算、缓存淘汰或整片执行。概率和 context 的 INT8 存储也属于性能 what-if，不表示量化精度已验证。

公式和容量口径见 [Decode 模型说明](ARCHITECTURE.md#decode-attention)。[目标模型覆盖矩阵](validation/issue3/model_coverage.json) 已登记已实现、部分实现和待实现块；常规 GQA/KV Cache 不替代 DeepSeek-R1 的 MLA 压缩缓存，整个 Issue #3 仍未完成。

### INT8 RMSNorm 与 Attention 投影（Issue #3 第六批 A）

四个入口增加三种 schema 1 workload，对应 Python API `RMSNormINT8`、`QKVProjection`、`AttentionOutputProjection`，均通过 `simulate_block` 执行：

| workload | 输入 → 输出 |
| --- | --- |
| `rmsnorm_int8` | INT8 x、FP32 weight → FP32 RMSNorm → INT8；最后轴、epsilon 语义沿用原实现 |
| `qkv_projection` | INT8 x `[B,H]`、Wq `[H,Hq*D]`、Wk/Wv `[H,Hkv*D]` → INT8 Q `[B,Hq,1,D]`、K/V `[B,Hkv,1,D]` |
| `attention_output_projection` | INT8 context `[B,Hq,1,D]`、Wo `[Hq*D,Hout]` → INT8 `[B,Hout]` |

```powershell
python ascend.py --workload operators/rmsnorm_int8.json
python tpu.py --workload operators/qkv_projection.json --details-output results/details/tpu_qkv.json
python tsp.py --workload operators/attention_output_projection.json --mem-slices 0 1 --streams 2
python tsp_cim.py --workload operators/rmsnorm_int8.json --hemisphere 1
```

QKV workload 还须提供 `query_heads`、`kv_heads`、`head_dim`，Hq 必须是 Hkv 的整数倍；可用 `output_names=["q","k","v"]` 指定三个互异输出名，其他两种块使用 `output_name`。矩阵权重按 `[in,out]` 描述，无 bias。INT32 累加结果经过显式边界工作代理存为 INT8；该投影块的 Q/K 输出尚未经过 RoPE，需连接独立的 RoPE 块。

继续使用 `--block-config`、`--vector-config` 和 `--vector-tile-elements`。子块串行物化，Q/K/V 最终输出同时保留；head 拼接只是明确记录的符号存储视图，连接仍需布局/分配校验。归一化峰值包括 FP32 中间张量、行状态和临时区。完整 Attention 层与模型尚未拼装；详见[计数和容量说明](ARCHITECTURE.md#decode-projections)。

### RoPE、Embedding 与词表输出（Issue #3 第六批 B）

三个独立 block 继续通过四个 CLI 或 `simulate_block` 执行，使用同一组 `--mode`、`--vector-config`、`--block-config` 和本侧 MEM 分配选项：

| workload / Python API | 输入与输出约定 |
| --- | --- |
| `rope` / `RoPE` | INT8 Q `[B,Hq,1,D]`、K `[B,Hkv,1,D]`，FP32 cos/sin 各 `[C,D/2]`，`position` 为 `[0,C)` 内整数；输出两个同形状 INT8 张量 |
| `embedding` / `Embedding` | INT32 indices `[B]`、INT8 weight `[V,H]` → INT8 `[B,H]` |
| `vocabulary_projection` / `VocabularyProjection` | INT8 x `[B,H]`、weight `[H,V]` → FP32 logits `[B,V]` |

```powershell
python ascend.py --workload operators/rope.json --vector-tile-elements 4
python tpu.py --workload operators/embedding.json
python tsp.py --workload operators/vocabulary_projection.json --mem-slices 0 1 --streams 2
python tsp_cim.py --workload operators/rope.json --hemisphere 1 --details-output results/details/tsp_cim_rope.json
```

RoPE 只处理单 token、整条 head 的相邻通道对，D 必须为偶数，整个 batch 使用同一 position；V 不旋转。sin/cos 表由调用方预先提供，本块统计完整表驻留和每个 head 的行读取，不计算三角函数或表生成。`--vector-tile-elements` 至少为 2，奇数上限向下取完整通道对；内部 FP32 旋转后显式转换回 INT8。`output_names` 可指定两个互异输出名。

Embedding 的 token 值保持符号描述，假设索引合法，不执行真实查表；同一行重复出现时按重复读取计算。完整词表、索引和输出都参加容量检查，访问量只计算本次索引与选中行。词表投影保留原生 INT32 累加，显式转换为 FP32 logits，不含最终 Softmax 或采样。这两种块使用可选 `output_name`。

这些模型仍是单核/单半片的局部执行。TSP/CIM 无法驻留完整词表时明确拒绝，后续分片/片外供给不能靠缩小 tile 替代。当前 [覆盖矩阵](validation/issue3/model_coverage.json) 的 LLaMA 独立块已覆盖，MoE、整片调度和完整模型拼装仍待推进。详细公式、容量与假设见 [第六批 B 说明](ARCHITECTURE.md#position-vocabulary)。

### MLA 前处理与压缩缓存（第七批 A）

```powershell
python tpu.py --workload operators/mla_preprocess.json --mode timing
python tsp.py --workload operators/mla_cache.json --mode timing --mem-slices 0 1 2 --streams 2
```

四架构均提供 `MLASpec`、`MLAPreprocess`、`MLACacheSpec`、`MLACacheAccess`。前处理使用 `simulate_block`，缓存使用 `simulate_mla_cache`。`dimensions` 显式区分 hidden、Q rank、KV rank、head 数、非位置 QK 维度、RoPE 维度和 V 维度；本批只支持正 Q rank 的低秩路线。样例为小尺寸性能验证，INT8 存储是项目 what-if 假设，不是 DeepSeek 官方 FP8/BF16 数值复现。

前处理产生 `query_nope`、旋转后的 `query_rope`、归一化后的 `latent_kv` 和旋转后的共享 `key_rope`，四者均为 INT8。Q 和 KV 投影后的分支拆分有独立读写、转换与存储分配；只对位置分支执行 RoPE。缓存以 `[B,1,C,KV_rank]` 和 `[B,1,C,RoPE_dim]` 预分配，不随 query head 数复制；追加量仅为 0/1，读取只覆盖已填充长度，输出状态描述复用原缓存。

前处理和独立缓存接口不包含 Attention 核心；第七批 B 的 `MLADecodeAttention` 接续这两者，显式计入 absorbed 路线中的投影、两路打分和输出工作。完整依赖、容量口径和执行假设见 [MLA 说明](ARCHITECTURE.md#mla-preprocess-cache)。

### Absorbed MLA Attention（第七批 B）

```powershell
python tpu.py --workload operators/mla_decode_attention.json --mode timing
python tsp_cim.py --workload operators/mla_decode_attention.json --mode timing --mem-slices 0 1 2 --streams 2 --hemisphere 1
```

`MLADecodeAttention` / `simulate_mla_attention` 接收已完成前处理的两路 Query、压缩缓存、新 token 和 `kv_up_weight[Hq,dn+dv,rk]`。每个 head 依次执行 Q→latent、内容打分、位置打分、latent 加权求和及 latent→V 五次原生 GEMM；两路 score 分别转 FP32，再合并、缩放和 Softmax。缓存追加一次；latent 缓存被每个 head 用于两次矩阵读取，位置缓存读取一次。

样例提供 `output_weight[Hq*dv,H]`，因此输出 INT8 `[B,H]`；省略时输出 INT8 context `[B,Hq,1,dv]`。API `MLADecodeAttention.from_preprocess(preprocess, kv_up_weight, cache, output_weight=...)` 可绑定第七批 A 的四个输出，并核对 position 与更新后缓存的最新 token；该方法只建立描述，不隐式执行前处理或承诺跨算子驻留复用。`append_tokens=0` 读取已有缓存时不接受新 token，position 对应 `length-1`。

默认缩放为 `1/sqrt(dn+dr)`。长上下文修正可通过正的 `scale_multiplier` 显式提供，非默认值必须附 `scale_source`；例如调用方可提供经过确认的 YaRN mscale²。不会根据缓存 capacity 自动推断修正。INT8 精度、串行执行和符号布局继续沿用项目假设；完整容量与手算说明见 [第七批 B](ARCHITECTURE.md#mla-attention)。

### 导出报告

成功运行后默认保存到**工程根目录**的 `results/`，并在终端打印 `[Report]` 路径。从其他工作目录启动时，默认位置仍然是本工程的 `results/`。

- GEMM：`ascend.json`、`tpu.json`、`tsp.json` 或 `tsp_cim.json`。
- Vector / RMSNorm / Softmax：`<架构>_<算子>.json`，例如 `ascend_fp32_add.json`、`tpu_softmax.json`、`tsp_cim_rmsnorm.json`。不同算子分别保存，不覆盖原有 GEMM 报告。

目录会自动创建，同一架构与算子再次成功运行会覆盖对应文件；count/timing 共用该默认文件名，任务检查或仿真失败时不写入报告。需要保留多次结果时，用 `--output` 指定不同文件名。显式提供的相对路径以当前工作目录为基准，绝对路径直接使用：

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

清除本工程 `results/` 顶层的所有 `.json` 报告，包括默认报告和此前的 `_timing`、`_count` 报告。保留 `results` 目录、子目录及其他类型文件，不影响配置、源码或 `validation/issue2/`、`validation/issue3/` 验收证据。从其他目录调用时，目标仍是脚本所属工程的 `results/`。可加 `-WhatIf` 预览将删除的文件。

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

Vector/Nonlinear 参考 TSP 的共享工作量模型：`vector_base_ops_per_cycle=2048`、`vector_int32_add_cost=1`、`vector_fp32_basic_op_cost=1`、`vector_fp32_exp_cost=15`、`vector_fp32_rsqrt_cost=12`，均为 assumed。`TPUSpec.vector_service_cycles()` 支持 INT32 ADD、FP32 ADD/MUL/EXP/RSQRT，按总等价工作量统一取整；2048 次 EXP 对应 15 cycle。EXP/RSQRT 是算法工作量代理，该 Spec API 不执行数值运算或独立模拟完整 Softmax/RMSNorm；第三批复合层负责搬运与归约调度，服务推导见 [Vector 服务](ARCHITECTURE.md#tpu-vector-service)。各操作共用同一预算；FMA、DIV、归约仍拒绝。TPU GEMM 跨 K 累加在 MXU 内，Vector API 不参与 GEMM 计时。SMEM 已移除；寄存器容量、ICI 两项、工艺和面积共五项规格及拓扑说明单独保存在 [TPU 归档 JSON](configs/archive/tpu_v4_reference.json)，运行时不加载该文件。Scalar 数量及两级库存已移除，旧执行字段不再接受；历史证据存于 `retired_parameters`。标称 TOPS 和三项 VPU 几何参数也已移入该归档，仅供参考，不再进入活动配置和报告；归档与迁移规则见 [参数检查](ARCHITECTURE.md#tpu-parameter-usage)。

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

Ascend 新增与 TSP 同口径的共享 Vector/Nonlinear 计算服务：每核预算 64 op/cycle，INT32 ADD 与 FP32 ADD/MUL 代价为 1，FP32 EXP 为 15、RSQRT 为 12；新增 5 项均为 assumed，现有 29 项数值硬件参数均非 null。`AscendSpec.vector_service_cycles()` 提供独立计算占用估算，GEMM 的 Cube/L0C 累加不调用它。64 的每周期假设尚未标定，Exp/Rsqrt 是算法工作量代理；没有独立 SFU，该 Spec API 仅估算计算占用；第三批复合层另外实现 RMSNorm/Softmax 的搬运、归约和依赖调度。旧 schema 7 须补齐五项配置完成 schema 8，再移出 Cube/Vector 单元数量，升级到当前 schema 9。来源、公式和 API 示例见 [Ascend Vector 服务](ARCHITECTURE.md#ascend-vector-service)。

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

Issue #3 已实现算子的独立验证：

```powershell
python scripts/verify_issue3.py
```

输出到 `validation/issue3/run/`，只运行算子及验证工具测试，另执行 184 项算子 CLI 检查。范围、逐项覆盖和剩余缺口见 [Issue #3 验证说明](validation/issue3/README.md)及[覆盖清单](validation/issue3/scope.json)；退出 0 只表示已实现范围通过，不表示整个 Issue #3 已完成。

按 Issue #2 的四类交付物和五条条件执行原生模型验收：

```powershell
python scripts/verify_issue2.py
```

交付物位置及通过判据见[逐条验收表](ARCHITECTURE.md#validation)，Issue #2 使用独立的 [acceptance.json](validation/issue2/run/acceptance.json)，只发现原生架构及 common 测试，不执行 `tests/operators`。脚本保存测试明细、四份 count 报告、四份 timing 正例与缺参负例；只有自动检查通过且来源/迁移审阅记录仍匹配当前文件时才退出 0。查看结果时须核对文件快照，不能把历史通过状态用于已修改的工程。

历史验证、迁移基线和一次性脚本已集中归档，位置与保留规则见 [验证记录说明](validation/README.md)。自动检查通过后，A1/A5 仍可能因语义审阅快照过期而显示 `needs_review`；清理文件或重新运行脚本不会代替审阅。

PR #7 已由师兄审阅、修复并合入 main（`7c84187`）。[审阅状态标注](validation/issue2/review_context.json)单独记录这一事实，验证报告通过 `review_context` 展示；不要把旧的本地 `needs_review` 解释成仍在等待师兄审阅该 PR。该记录不覆盖后续 Issue #3 修改，也不代替 A1/A5 的逐项证据。

Issue #2 测试覆盖与验证记录维护在 [原生模型验证状态](ARCHITECTURE.md#validation)；Issue #3 另见[独立验证目录](validation/issue3/README.md)，参数来源结论见 [核验结果](ARCHITECTURE.md#source-audit)与[逐项台账](calibration/parameter_source_audit.json)，剩余缺口见 [参数清单](ARCHITECTURE.md#remaining)。修改参数时应同时更新来源及状态，硬件配置路径见 [文件职责表](ARCHITECTURE.md#migration)。旧能耗/面积模型、DSE 和报告 API 的迁移同样记录在那里。

```text
huawei_dfp/
├── README.md / ARCHITECTURE.md       # 使用入口 / 统一技术说明
├── ascend.py / tpu.py                # Ascend / TPU 独立入口
├── tsp.py / tsp_cim.py               # TSP / CIM-TSP 入口
├── configs/hardware/                # 架构参考规格与共享 Macro profile
├── configs/archive/                 # 已退出运行配置的硬件规格与来源
├── configs/workload/                # 公共 GEMM 任务
├── configs/workload/operators/      # Vector / 归一化任务
├── configs/execution/               # 算子执行策略与代理参数
├── src/hardware/                    # 规格、校验与派生库存
├── src/operators/                   # 算子描述、公共 API、结果适配
├── src/runtime/                     # 后端选择、执行编排、CLI 选项与输出
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
│   ├── common/                     # 任务、输出路径、库存和参数来源
│   ├── operators/                  # Issue #3 算子与兼容性测试
│   └── validation/                 # 验证工具范围和目录隔离
├── scripts/verify_issue2.py          # Issue #2 原生模型验收
├── scripts/verify_issue3.py          # Issue #3 已实现算子独立验证
├── scripts/verification.py          # 两者共用的无验收策略工具
├── scripts/run_all.ps1 / run_all.py  # issue2 / issue3 / all 仿真与验证总入口
├── scripts/clear_results.ps1         # 清除 results 顶层生成的 JSON 报告
└── validation/                      # 当前验收与历史材料索引
    ├── issue2/                     # 原生审阅依据及 run/ 验收
    ├── issue3/                     # 算子覆盖清单及 run/ 验证
    └── archive/                    # 历史验证、基线与过程材料归档
```

目录约定：`src/core/<架构>/simulator.py` 负责执行统计；Ascend、TPU、原生 TSP 的 `tiling.py` 负责分块，CIM-TSP 的半芯片映射和调度位于 `hemisphere.py`；共享 Macro 实现归入 `src/core/cim/macro_model.py`。硬件规格继续放在 `src/hardware/<架构>.py`，共享字段校验为 `parameters.py`；配置和报告按 `<架构>_config.py`、`<架构>_report.py` 命名，Macro 引用由 `tsp_cim_config.py` 解析。各架构按实际需要提供报告模块，CIM-TSP 入口直接输出 JSON。

算子调用按 `CLI / operators.simulate_* → runtime → core → 结果适配` 分层。Vector/归一化内核返回原生结果，外层统一生成 `OpResult`；`utils/operator_workload.py` 只解析任务。调用入口、仿真公式和报告路径保持兼容；结果 JSON 已迁移为 schema 2，具体职责见[算子执行分层](ARCHITECTURE.md#operator-layering)。

通用实验入口 `main.py` 及其 `default_core.json` 场景已移除；当前只保留四个架构入口。硬件/workload JSON 和报告内容保持不变，`--output` 现在默认写入工程 `results/`。旧的内部模块路径与新位置见 [目录迁移表](ARCHITECTURE.md#directory-migration)。

原生 TSP 现使用 [有限权重预取模型](ARCHITECTURE.md#tsp-weight-prefetch)：LW 利用当前行流水剩余的 MEM 读和同方向 stream 带宽，IW 在旧 tile 排空后通过内部端口安装权重。仅保留一个待用块，假设缓冲容量为 204800 byte（200 KiB）；超过容量明确拒绝。30 项参考参数全部有值，18 documented、12 assumed。缓冲容量和保守 IW 时序均为建模假设，公开论文仅支持机制；没有保留旧串行运行选项。

报告以 `timing.elapsed_cycles.by_stage` 中的 `weight_load_stall + install + row_service` 分解总周期，另列已隐藏的 `weight_load_overlap_cycles` 和各 tile/LW 段。首次装载不会隐藏，内部 IW 不再次读取 MEM。默认 tiny 为 **601 cycle**，原有动作计数和 MEM 驻留容量保持不变。自定义 schema 7 配置先补全权重缓冲参数及预取策略完成 schema 8，再移出 `nominal_int8_tops`，升级到当前 schema 9。
