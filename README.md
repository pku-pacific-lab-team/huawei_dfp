# Digital CIM Simulator

Digital Compute-in-Memory (CIM) 架构性能与能耗仿真器，支持多种数据格式的设计空间探索 (DSE)。

## 项目结构

```
huawei_dfp/
├── configs/                          # 配置文件
│   ├── energy/                       # 能量模型配置
│   │   └── default_22nm.json
│   ├── hardware/                     # 硬件参数配置
│   │   └── default_core.json
│   └── workload/                     # 工作负载配置
│       └── default_gemm.json
│
├── src/                              # 源代码
│   ├── hardware/                     # 硬件规格定义
│   │   ├── macro.py                  # CIM Macro 规格
│   │   └── dataformat.py             # 数据格式定义
│   ├── core/                         # 核心仿真器
│   │   ├── simulator.py              # CIM Core 仿真器
│   │   └── tiling.py                 # Tiling 策略引擎
│   ├── dse/                          # 设计空间探索
│   │   └── explorer.py               # DSE 引擎
│   └── utils/                        # 工具模块
│       ├── config.py                 # 配置加载器
│       └── report.py                 # 结果报告生成
│
├── main.py                           # 主入口
└── model_fp.py                       # 原始单文件版本 (备份)
```

## 快速开始

### 环境要求

- Python 3.8+
- pandas

### 安装依赖

```bash
pip install pandas
```

### 运行仿真

```bash
# 使用默认配置运行所有数据格式的 DSE
python main.py

# 仅测试 INT8 格式
python main.py --formats INT8

# 测试多种格式
python main.py --formats INT8 BF16 MXINT8

# 使用自定义配置
python main.py --energy default_22nm.json \
               --hardware default_core.json \
               --workload default_gemm.json

# 详细输出模式
python main.py --verbose
```

## 配置文件说明

### 能量模型 (`configs/energy/*.json`)

定义各组件的能耗参数 (单位: pJ):

| 参数 | 说明 |
|------|------|
| `tpu_flop_power` | CIM MAC 操作能耗 |
| `vect_flop_power` | 向量单元操作能耗 |
| `l1_sram_read/write_power` | L1 Buffer 读写能耗 |
| `l0_sram_read/write_power` | L0 Buffer 读写能耗 |
| `dram_read/write_power` | DRAM 访问能耗 |
| `cim_local_read_power` | CIM 本地读出能耗 |

### 硬件参数 (`configs/hardware/*.json`)

定义 CIM Core 硬件配置:

| 分组 | 参数 | 说明 |
|------|------|------|
| **macro** | `n_banks_internal` | Bank 数量 |
| | `n_subarrays_per_bank` | 每 Bank 的 Sub-array 数量 |
| | `n_loc_per_subarray` | 本地读出开关数 |
| | `n_cols_per_bank` | 每 Bank 的 Column 数 |
| **core** | `n_macros` | Macro 数量 |
| | `clock_freq_ghz` | 时钟频率 (GHz) |
| **memory** | `size_l1_input/output_buffer_bytes` | L1 Buffer 大小 |
| | `size_l0_input/output_buffer_bytes` | L0 Buffer 大小 |
| **bandwidth** | `bw_noc_to_l1_gbps` | NoC → L1 带宽 |
| | `bw_l1_to_l0_gbps` | L1 → L0 带宽 |
| **vector_unit** | `throughput_gops` | 向量单元吞吐量 |

### 工作负载 (`configs/workload/*.json`)

定义 GEMM 工作负载:

| 参数 | 说明 |
|------|------|
| `gemm.M/K/N` | 矩阵维度 |
| `data_format` | 数据格式 (INT8/BF16/MXINT8/FP8/MXFP8_M4) |
| `custom_overheads` | 自定义开销参数 (可选) |

## 支持的数据格式

| 格式 | 存储 (Bytes) | 尾数位 | 保护位 | 说明 |
|------|-------------|--------|--------|------|
| INT8 | 1.0 | 8 | 0 | 标准 8 位整数 |
| BF16 | 2.0 | 8 | 7 | Brain Floating Point 16 |
| MXINT8 | 1.03125 | 8 | 2 | 共享指数 INT8 |
| FP8 | 1.0 | 4 | 5 | 8 位浮点 |
| MXFP8_M4 | 1.03125 | 4 | 5 | 共享指数 FP8 |

## 输出指标

- **Latency**: 总延迟 (Cycles)
- **Energy**: 总能耗 (pJ)
- **Throughput**: 吞吐量 (GOPS)
- **EDP**: Energy-Delay Product (能耗×延迟)
- **Efficiency**: 能效 (TOPS/W)
- **Bottleneck**: 系统瓶颈阶段

## API 使用示例

```python
from src.hardware import CIMMacroSpec
from src.core import CIMCoreSimulator
from src.utils import ConfigLoader

# 加载配置
loader = ConfigLoader("configs")
energy_params = loader.load_energy_config()
hardware_config = loader.load_hardware_config()

# 创建 Macro 规格
macro_spec = loader.create_macro_spec(hardware_config)

# 创建仿真器
sim = CIMCoreSimulator(
    n_macros=32,
    macro_spec=macro_spec,
    size_l1_input_buffer_bytes=256*1024,
    size_l1_output_buffer_bytes=256*1024,
    size_l0_input_buffer_bytes=4096,
    size_l0_output_buffer_bytes=4096,
    bw_noc_to_l1_gbps=64.0,
    bw_l1_to_l0_gbps=128.0,
    throughput_vector_unit_gops=128,
    energy_params=energy_params,
    data_format='INT8'
)

# 运行仿真
result = sim.simulate(M=1024, K=4096, N=1024,
                      partition_strategy=(2, 16),
                      loop_order='MKN')

print(f"Throughput: {result['throughput_gops']:.2f} GOPS")
print(f"Energy: {result['total_energy_pj']:.2f} pJ")
```
