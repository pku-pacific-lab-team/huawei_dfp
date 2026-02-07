# Digital CIM Simulator

Digital Compute-in-Memory (CIM) 架构性能与能耗仿真器，支持多种数据格式的设计空间探索 (DSE)。
同时包含 Ascend 910 Cube 数字基线仿真器，用于 CIM vs 传统数字架构的对比分析。

## 项目结构

```
huawei_dfp/
├── configs/                              # 配置文件
│   ├── energy/                           # 能量模型配置
│   │   ├── default_22nm.json             #   CIM: 22nm 工艺
│   │   └── 12nm.json                     #   Cube: 12nm 工艺
│   ├── hardware/                         # 硬件参数配置
│   │   ├── default_core.json             #   CIM Core 配置
│   │   └── ascend.json                   #   Ascend 910 配置
│   └── workload/                         # 工作负载配置
│       ├── default_gemm.json             #   默认 GEMM (1024×4096×1024)
│       └── tiny_gemm.json                #   小规模 GEMM (512×256×512)
│
├── src/                                  # 源代码
│   ├── hardware/                         # 硬件规格定义
│   │   ├── macro.py                      #   CIM Macro 规格
│   │   ├── cube.py                       #   Cube Unit 规格
│   │   └── dataformat.py                 #   数据格式定义 (INT8/BF16/FP16 等)
│   ├── core/                             # 核心仿真器
│   │   ├── simulator.py                  #   CIM Core 仿真器
│   │   ├── tiling.py                     #   CIM Tiling 策略引擎
│   │   ├── cube_simulator.py             #   Cube Core 仿真器
│   │   └── cube_tiling.py                #   Cube Tiling 策略引擎
│   ├── dse/                              # 设计空间探索
│   │   └── explorer.py                   #   DSE 引擎
│   └── utils/                            # 工具模块
│       ├── config.py                     #   CIM 配置加载器
│       ├── cube_config.py                #   Cube 配置加载器
│       └── report.py                     #   结果报告生成
│
├── main.py                               # CIM 仿真主入口 (含 DSE)
├── ascend.py                             # Cube 仿真主入口
└── model_fp.py                           # 原始单文件版本 (备份)
```

## 快速开始

### 环境要求

- Python 3.8+
- pandas

### 安装依赖

```bash
pip install pandas
```

### 运行 CIM 仿真

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

### 运行 Cube 基线仿真

```bash
# 使用默认 Ascend 910 配置
python ascend.py

# 指定数据格式和循环顺序
python ascend.py --format INT8 --loop-order KNM

# 使用自定义配置
python ascend.py --energy configs/energy/12nm.json \
                 --hardware configs/hardware/ascend.json \
                 --workload configs/workload/tiny_gemm.json
```

## CIM 架构建模

### 硬件架构概览

```
                          Core #0
┌───────────────────────────────────────────────────┐
│                                                   │
│  ┌─────────┐    ┌──────────────────────────────┐  │
│  │ L1 Input│───▶│  L0 Input Buffer (per Macro) │  │
│  │ Buffer  │    └──────────┬───────────────────┘  │
│  └────┬────┘               │                      │
│       │              ┌─────▼─────┐                │
│  DRAM │              │ WL Driver │                │
│   ▲   │              └─────┬─────┘                │
│   │   │     ┌──────────────▼──────────────┐       │
│   │   │     │    CIM Macro (×32)          │       │
│   │   │     │  ┌────────────────────────┐ │       │
│   │   │     │  │ Bank (×64, N方向)      │ │       │
│   │   │     │  │ ┌──────────────────┐   │ │       │
│   │   │     │  │ │SubArray (×64, K) │   │ │       │
│   │   │     │  │ │ 4 switches ×     │   │ │       │
│   │   │     │  │ │ 8 columns(bits)  │   │ │       │
│   │   │     │  │ └───────┬──────────┘   │ │       │
│   │   │     │  │     Adder Tree (K方向) │ │       │
│   │   │     │  └─────────┬──────────────┘ │       │
│   │   │     │      Shift-Accumulator      │       │
│   │   │     └────────────┬────────────────┘       │
│   │   │           ┌──────▼───────┐                │
│   │   │           │ L0 Output    │                │
│   │   │           │ (PSUM Buffer)│                │
│   │   │           └──────┬───────┘                │
│   │   │           ┌──────▼───────┐                │
│   │   └──────────▶│ L1 Output    │                │
│   │               │ Buffer       │                │
│   │               └──────┬───────┘                │
│   └──────────────────────┘                        │
└───────────────────────────────────────────────────┘
```

### Macro 物理参数

每个 CIM Macro 的维度由以下参数决定：

- **block_k_dim** = `n_subarrays_per_bank × n_loc_per_subarray` = 64 × 4 = **256**
  - 每个 Sub-array 存 4 个权重 (n_loc_per_subarray=4)
  - 64 个 Sub-array 串联 (K 方向)
- **block_n_dim** = `n_banks_internal` = **64**
  - 64 个 Bank 并行 (N 方向)
- 每个 Sub-array 有 8 个 Column，对应权重的 8-bit 存储

### 两级 Tiling 策略

仿真器使用两级 Tiling 将 GEMM 问题 (M×K×N) 切分到硬件上：

```
GEMM (M × K × N)
  └─ L1 Tile (M_l1 × K_l1 × N_l1)    ← 受限于 L1 Buffer 容量
       └─ L0 Tile (M_l0 × K_l0 × N_l0)  ← 受限于 L0 Buffer 容量 + Macro物理维度
```

**L0 Tile 大小**由两个约束的最小值决定：
- **L0 Input 约束**: `M_l0 = L0_input_bytes / (block_k_dim × storage_bytes)`
- **L0 Output 约束**: `M_l0 = L0_output_bytes / (block_n_dim × 4)`，PSUM 固定 32 位

例如 INT8 (storage=1B)、L0 各 4KB：
- Input: 4096 / (256 × 1) = **16 行**
- Output: 4096 / (64 × 4) = **16 行** → M_l0 = 16

### 流水线模型 (7 级)

```
时间轴 ────────────────────────────────────────────────────→

NoC:         ████████                    ████████
             DRAM→L1 加载                (miss时)

Weight_Prog:      ████████                    ████████
                  权重编程到 SRAM bitcell      (miss时)

Bus_L1_L0:             ████  ████  ████       ████
                       L1→L0 Input传输        (miss时跳过)

Pre_Proc:                 ███  ███  ███
                          输入对齐/预处理

Macro:                       ████  ████  ████
                             CIM 阵列位串行计算

Post_Proc:                      ██  ██  ██
                                Shift-Acc 累加

L1_Output:                        ████████████████
                                  PSUM 驱逐/完成tile写回 L1
```

各级之间的依赖关系：
- `Bus_L1_L0` 必须等 `NoC` 完成（数据需先到 L1）
- `Macro` 必须等 `Pre_Proc` **和** `Weight_Prog` 同时完成
- `Post_Proc` 必须等 `Macro` 完成
- `L1_Output` 在驱逐 PSUM 时必须等 `Post_Proc` 完成
- 最终延迟 = max(所有阶段时间戳)

### PSUM 空间管理

L0 Output Buffer 容量有限（默认 4KB/Macro），仿真器精确追踪 PSUM 的分配和驱逐：

- **空间占用**: 每个 C tile 的 PSUM = `M_l0 × block_n_dim × 4B`。默认配置下 16×64×4=4096B，恰好占满一个 Macro 的 L0 output，即**同一时间只能存一个 C tile 的 PSUM**
- **K 维度累加**: 同一 C tile `(m, n)` 的多次 K 迭代共享同一块 PSUM 空间（read-modify-write）
- **驱逐策略**: 当新 C tile 需要写入但空间已被占用时，将旧 tile 驱逐到 L1 Output（产生读 L0 + 写 L1 的延迟和能耗）
- **完成转移**: 当 C tile 的所有 K 维度计算完毕，自动从 L0 Output 转移到 L1 Output

### L1 缓存追踪

L1 Buffer 采用多 tag 字典追踪，支持数据复用：

- **Input 缓存**: 追踪 `{(m, k): bytes}`，同一 input tile 在不同 N 迭代中可复用
- **Weight 缓存**: 追踪 `{(k, n): bytes}`，同一 weight tile 在不同 M 迭代中可复用
- **驱逐策略**: FIFO，空间不足时驱逐最早加载的 tile
- **效果**: 不同 loop order 产生不同的缓存命中模式，直接影响 DRAM 访问能耗

### 权重重编程

CIM 架构的权重存储在 SRAM bitcell 中，当 K 维度迭代（新权重列）时需通过 Weight I/O 端口重新编程：

- **时机**: 仅在 L1 weight cache miss 时触发
- **延迟**: `bytes_per_macro / weight_prog_bpc` 周期
- **依赖**: Macro 计算必须等待权重编程完成
- **向后兼容**: 若 `weight_prog_bw_gbps` 未配置则自动禁用（零开销）

### Loop Order 的影响

6 种循环顺序（MKN/KMN/MNK/KNM/NKM/NMK）对外层 L1 和内层 L0 循环同时生效。
不同顺序影响缓存命中率和能耗：

| Order | L1 Input 复用 | L1 Weight 复用 | 典型能效 |
|-------|:------------:|:-------------:|:-------:|
| MKN   | M→N时 input 可复用 | 无 | 中 |
| KNM   | 无 | K→M时 weight 可复用 | 高 |
| MNK   | 无 | 无 | 低 |

## 配置文件说明

### 能量模型 (`configs/energy/*.json`)

定义各组件的能耗参数 (单位: pJ/byte 或 pJ/op):

| 参数 | 说明 |
|------|------|
| `tpu_flop_power` | CIM MAC 操作能耗 |
| `vect_flop_power` | 向量单元操作能耗 |
| `sram_write_power` | SRAM 写入能耗 (权重加载) |
| `sram_read_power` | SRAM 读取能耗 |
| `l1_sram_read/write_power` | L1 Buffer 读写能耗 |
| `l0_sram_read/write_power` | L0 Buffer 读写能耗 |
| `cim_local_read_power` | CIM 本地读出电路能耗 |
| `dram_read/write_power` | DRAM 访问能耗 |
| `weight_prog_power` | 权重编程能耗 (pJ/byte) |
| `weight_prog_bw_gbps` | Weight I/O 端口带宽 (Gbps) |

### 硬件参数 (`configs/hardware/default_core.json`)

| 分组 | 参数 | 说明 | 默认值 |
|------|------|------|--------|
| **macro** | `n_banks_internal` | Bank 数量 (→ block_n_dim) | 64 |
| | `n_subarrays_per_bank` | 每 Bank 的 Sub-array 数量 | 64 |
| | `n_loc_per_subarray` | 每 Sub-array 权重数 | 4 |
| | `n_cols_per_bank` | 每 Bank 的 Column 数 (权重 bit 宽) | 8 |
| **core** | `n_macros` | Macro 数量 | 32 |
| | `clock_freq_ghz` | 时钟频率 (GHz) | 1.0 |
| **memory** | `size_l1_input_buffer_bytes` | L1 Input Buffer | 256 KB |
| | `size_l1_output_buffer_bytes` | L1 Output Buffer | 256 KB |
| | `size_l0_input_buffer_bytes` | L0 Input Buffer (per Macro) | 4 KB |
| | `size_l0_output_buffer_bytes` | L0 Output / PSUM Buffer (per Macro) | 4 KB |
| **bandwidth** | `bw_noc_to_l1_gbps` | NoC → L1 带宽 | 64.0 |
| | `bw_l1_to_l0_gbps` | L1 ↔ L0 带宽 (双向) | 128.0 |
| **vector_unit** | `throughput_gops` | 向量单元吞吐量 | 128 |

### 工作负载 (`configs/workload/*.json`)

| 参数 | 说明 |
|------|------|
| `gemm.M/K/N` | 矩阵维度 |
| `data_format` | 数据格式 (INT8/BF16/MXINT8/FP8/MXFP8_M4) |
| `custom_overheads` | 自定义开销参数 (可选) |

## 支持的数据格式

### CIM 格式

| 格式 | 存储 (Bytes) | 尾数位 | 保护位 | 说明 |
|------|-------------|--------|--------|------|
| INT8 | 1.0 | 8 | 0 | 标准 8 位整数 |
| BF16 | 2.0 | 8 | 7 | Brain Floating Point 16 |
| MXINT8 | 1.03125 | 8 | 2 | 共享指数 INT8 |
| FP8 | 1.0 | 4 | 5 | 8 位浮点 |
| MXFP8_M4 | 1.03125 | 4 | 5 | 共享指数 FP8 |

### Cube 格式

| 格式 | 字节 | 说明 |
|------|------|------|
| INT8 | 1 | 标准 8 位整数 |
| INT16 | 2 | 16 位整数 |
| FP16 | 2 | 16 位浮点 |

## 输出指标

### 基础指标

| 指标 | 说明 |
|------|------|
| `total_latency_cycles` | 总延迟 (Cycles) |
| `total_energy_pj` | 总能耗 (pJ) |
| `throughput_gops` | 吞吐量 (GOPS) |
| `energy_efficiency_tops_per_watt` | 能效 (TOPS/W) |
| `bottleneck` | 系统瓶颈阶段 |

### 流水线各阶段延迟 (`latencies`)

| Key | 说明 |
|-----|------|
| `NoC` | DRAM → L1 数据加载 |
| `Weight_Prog` | 权重编程到 SRAM bitcell |
| `Bus_L1_L0` | L1 → L0 Input 传输 |
| `Pre_Proc` | 输入预处理/对齐 |
| `Macro` | CIM 阵列计算 |
| `Post_Proc` | Shift-Accumulator 累加 |
| `L1_Output` | PSUM 驱逐 / 完成 tile 写回 L1 |

### 能耗分解 (`energies_pj`)

| Key | 说明 |
|-----|------|
| `dram_read` / `dram_write` | DRAM 访问 |
| `load_X_NoC` / `load_W` | Input/Weight 加载到 L1 |
| `weight_prog` | 权重重编程 |
| `read_L1` / `write_L0` | L1→L0 数据搬运 |
| `pre_process` | 输入对齐 |
| `read_L0` / `local_readout` | L0 读取 / CIM 本地读出 |
| `compute_macro` | CIM 阵列计算 |
| `accum_vector_unit` | 向量单元累加 |
| `read_psums` / `write_psums` | PSUM 读写 (L0 Output) |
| `write_l1_output` | PSUM 写回 L1 Output |

### 缓存统计

| Key | 说明 |
|-----|------|
| `l1_cache_stats.input_hits/misses` | L1 Input 缓存命中/未命中 |
| `l1_cache_stats.weight_hits/misses` | L1 Weight 缓存命中/未命中 |
| `l0_input_cache_stats.hits/misses` | L0 Input 数据复用命中/未命中 |
| `l0_to_l1_output_stats.total_evict_count` | PSUM 驱逐次数 |
| `l0_to_l1_output_stats.total_evict_bytes` | PSUM 驱逐总字节 |

## API 使用示例

```python
from src.hardware.macro import CIMMacroSpec
from src.core.simulator import CIMCoreSimulator
from src.utils.config import ConfigLoader

# 加载配置
loader = ConfigLoader("configs")
energy_params = loader.load_energy_config("default_22nm.json")
hardware_config = loader.load_hardware_config("default_core.json")

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
result = sim.simulate(
    M=1024, K=4096, N=1024,
    partition_strategy=(4, 8),
    loop_order='KNM'
)

print(f"Throughput:  {result['throughput_gops']:.2f} GOPS")
print(f"Energy:      {result['total_energy_pj']/1e6:.1f} uJ")
print(f"Efficiency:  {result['energy_efficiency_tops_per_watt']:.2f} TOPS/W")
print(f"Bottleneck:  {result['bottleneck']}")
print(f"L1 weight hits: {result['l1_cache_stats']['weight_hits']}")
```

## Cube 基线仿真器

Ascend 910 的 Cube Unit 采用传统数字脉动阵列架构 (16×16×16)，作为 CIM 的对比基线。
Cube 仿真器独立运行，拥有自己的 Tiling 引擎、配置加载器和三缓冲区模型 (L0_A / L0_B / L0_C)。

```bash
# 运行 Cube 仿真
python ascend.py --format INT8 --loop-order MKN

# 输出示例
# Total Latency :      627,231 Cycles
# Total Energy  :   57,131,663 pJ
# Throughput    :       213.98 GFLOPS
# Efficiency    :         2.35 TFLOPS/W
```
