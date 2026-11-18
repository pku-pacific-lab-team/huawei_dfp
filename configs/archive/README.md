# 仅供参考的硬件规格归档

这里保存已经移出运行配置的原始参数、单位、作用域、来源、版本及条件。归档不是可执行硬件配置，运行时不会读取或合并这些文件，也不会用它们补齐缺失的计算服务。归档文件的 `schema_version=1` 是归档格式版本；`archived_from` 保留首次迁入时的配置身份，追加批次见 `archival_history`。

| 归档 | 本轮迁入参数 | 当前活动参数数 | 硬件 / 报告 schema |
|---|---|---:|---|
| [Ascend](ascend910_reference.json) | `cube.units_per_core`、`vector.units_per_core` | 29 | 9 / 12 |
| [TPU](tpu_v4_reference.json) | `nominal_int8_tops`、`vpus_per_tensorcore`、`vpu_lanes`、`vpu_alus_per_lane` | 21 | 12 / 13 |
| [TSP](tsp.json) | `nominal_int8_tops` | 30 | 9 / 10 |
| [CIM-TSP](tsp_cim.json) | `nim_units_per_plane`、`nim_tiles_per_plane`、`nim_channels_per_tile`、`nim_conversion_lanes` | 10 | 7 / 10 |

TPU 归档继续保留此前迁入的寄存器容量、ICI 链接数量和报告带宽、工艺节点、面积上界五项参数，以及 interconnect 描述；本轮追加后共九项数值参数。Ascend 使用原始点分路径作为归档参数名，避免两个 `units_per_core` 重名。其余三个归档保留原 `parameters` 下的名称。

本轮 11 项只服务标称规格或库存展示。活动报告移除 Ascend 的 `area_inventory.compute_units`、TPU 的标称 TOPS 和 VPU/ALU 库存、TSP 的标称 TOPS、CIM 的四项 NIM 库存；不以 0 或 null 占位。运行报告仍保留矩阵几何派生峰值、实际计算服务和其余库存，因此剩余库存属于部分库存，不代表完整芯片面积。

迁移旧自定义配置时，先按表移出对应字段，再升级硬件 schema；仅改版本号会因多余字段而报错。归档不改变 GEMM 动作、容量、调度和耗时，也不改变独立 Vector/VXM 服务公式；配置内容变化会更新原生 TSP/CIM 的 `reference_id`。六项几何重复输入、SR 坐标校验、bank 容量整除校验、physical/usable 容量区分均保留。

每项历史审阅依据保存在 [来源台账](../../calibration/parameter_source_audit.json) 的 `retired_parameters`；归档文件哈希在 `archives` 中。修改归档后须同步哈希，不能通过编辑归档改变运行模型。字符框图中保留的物理结构可引用归档事实，但不意味着它们仍是活动仿真输入。
