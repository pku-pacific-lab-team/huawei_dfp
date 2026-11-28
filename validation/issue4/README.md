# Issue #4 · 单片负载与张量并行

## 实现与验收入口

[Issue #4](https://github.com/pku-pacific-lab-team/huawei_dfp/issues/4) 的验收对象是 Decode 的计算图、逐片容量与内部计数。计算关键路径按单请求回放，片内交接、跨片传输和外部选词的服务时间取零。通信清单给出后续系统仿真所需的依赖、参与芯片和张量字节数。能耗系数和链路时延仍需单独建模。

| 原始要求 | 实现与核对方式 |
| --- | --- |
| 两份可追溯的模型配置，支持 batch、初始 KV 长度和 Decode 步数 | [配置加载与请求校验](../../src/models/config.py) 校验上游 Git blob、正整数参数和上下文上限。 |
| 生成 Decoder 层与整模依赖 | [LLaMA](../../src/models/llama_model.py) 组装 80 层 GQA。[DeepSeek-R1](../../src/models/deepseek_model.py) 组装 61 层 MLA，前 3 层使用 Dense FFN，后 58 层执行 MoE。 |
| 按片核算存储、动作和局部时间，拒绝超容量放置 | [逐片计算图](../../src/models/block_graph.py) 核算常驻权重、KV、活跃张量、工作区和后端私有存储。[模型测试](../../tests/models/) 包含小形状手算、容量边界及专家路由用例。 |
| 四架构公开入口可复现 | [专项脚本](../../scripts/verify_issue4.py) 从四个入口分别运行两个模型，核对默认放置、两步 Decode、容量拒绝、通信依赖及固定请求下的软件分块搜索。 |

在项目根目录运行 `python scripts/verify_issue4.py`。默认汇总为 `results/issue4/acceptance/acceptance.json`。`status` 为 `passed` 表示模型测试、八组整模用例、四组放置对齐以及运行前后的源码快照检查均通过。`run_all --scope all` 包含模型单测，八组整模用例由专项脚本单独运行。

**如何对齐版本。** `source_version.git_commit` 记录启动时的工程 HEAD，`git_dirty` 记录当时工作树是否有待提交变更。在源码导出目录也可运行，缺少的 Git 信息用 `null` 表示。`source_version.files` 按相对路径列出四个入口、工程源码、脚本、模型单测以及参考配置的 SHA-256，计算前统一文本换行。`source_version.sha256` 标识整份文件清单，`snapshot_unchanged_during_run` 比较验收前后的清单。有本地修改时，可据此定位同一提交上实际参与验收的文件版本。

## 里程碑四 · DeepSeek-R1 的 MLA 与专家并行

以下命令在项目根目录运行。四个架构入口默认取 batch=1、两步 Decode。`--sequence-length` 表示首步追加前已有的 KV 长度。LLaMA3-70B 默认取 8,190，两步的有效长度为 8,191 和 8,192，刚好到达模型上限。DeepSeek-R1 默认取 131,072（128k），两步的有效长度为 131,073 和 131,074，低于固定配置的 163,840 上限。计算报告和通信清单默认保存到 `results/issue4/`，可用 `--output` 和 `--communication-output` 指定路径。

```powershell
python ascend.py --model deepseek_r1
python tpu.py --model deepseek_r1 --routing concentrated
python tsp.py --model deepseek_r1 --mode count
python tsp_cim.py --model llama3_70b --mode timing
python scripts/verify_issue4.py
```

### 整模软件分块搜索

四个架构入口的 `--config-dir` 同时指定硬件与执行配置目录。整模计数、计时和候选分块共用该目录中的 `execution/vector.json` 与 `execution/blocks.json`。

四个架构入口的 `--auto-tile` 开关在 timing 模式下生效。沿用默认配置只需一条命令，也可与已有的请求、放置或路由参数组合使用。

```powershell
python tpu.py --model llama3_70b --auto-tile
python tsp.py --model deepseek_r1 --routing concentrated --auto-tile
```

一次运行固定请求、各片持有的权重、KV 和专家命中，对同一组算子比较默认与候选分块。向量算子尝试默认元素数的一半和两倍，跳过被局部宽度裁成相同大小的请求。GEMM 从当前形状下的默认块出发，一次将一个可调维度减半或加倍，遵守后端的对齐和上界。TSP-CIM 只允许改变 M，batch=1 的单行投影沿用原生块。每个候选都按整节点的完整执行阶段计时，包含各片内单元的并行波次。容量检查覆盖活跃张量、常驻分片、临时区及后端私有存储，默认块仍可使用原有的容量回退。

对每个局部算子选出周期最少的合法候选，再把这组选择放回同一张 Decode 依赖图。回放时，同一物理片的计算按既定顺序串行，独立芯片可并行。PP 按发送和接收片传递就绪时间，TP 归约及专家合并等待所需输入。跨片事件和外部选词的耗时取零，原生存储服务已包含在算子周期中。整条路径同速时保留默认方案。若默认块超容量而候选可执行，报告给出默认块的容量诊断。

默认文件名为 `results/issue4/<架构>_<模型>_timing_auto_tile.json`。其中 `profiles`、`chips` 反映实际选中方案，`tile_search` 提供以下对照。

| 字段 | 读法 |
| --- | --- |
| `baseline`、`per_node_candidate`、`selected` | 分别为默认方案、逐节点选块组合与最终整模选择。各自报告计算关键路径周期、秒数、整片驻留峰值及工作区。 |
| `peak_chip_workspace_bytes`、`peak_private_bytes_per_unit` | 前者统计同一节点在整片上的临时区与显式选择工作区，后者分别统计各私有存储层级在单元上的峰值。原生 TSP 的 `weight_buffer` 取单次待装载权重的最大字节数。各层峰值可出现在不同节点。 |
| `selected_policy`、`saved_cycles`、`latency_reduction_fraction` | 选中策略、节省周期、相对默认时延的下降比例。默认方案不可执行时后两项为空。 |
| `profile_trials[profile][节点]` | 有不同候选的节点记录分块参数、默认及选中的请求值、各候选周期或超容量原因。请求值经原生后端按实际尺寸、局部容量和尾块裁剪。 |

运行 `python scripts/verify_issue4.py` 可复现四架构、两模型的对照和容量检查。汇总在 `results/issue4/acceptance/acceptance.json` 的各个 `cases.*.tile_search` 中。

### 默认放置

**整模默认分片。** PP 按连续层分段，注意力 TP 和 EP 可以共用芯片。表内片数按每段实际使用的物理芯片并集累加。DeepSeek-R1 的专家内部 TP 均为 1。执行沿用工程的 INT8 GEMM、INT32 累加和 INT8 压缩缓存假设，归一化、路由权重、专家返回和带权合并使用 FP32。

| 架构 | LLaMA3-70B（初始 KV 8,190） | DeepSeek-R1（初始 KV 131,072） |
| --- | --- | --- |
| Ascend | PP3、TP1，共 3 片 | PP31、TP1、EP1，共 31 片 |
| TPU | PP3、TP1，共 3 片 | PP31、TP1、EP1，共 31 片 |
| TSP | PP78，首尾 TP16，中间 76 段 TP4，共 336 片 | PP61，第 0 层 TP16，第 1–2 层 TP8，第 3–59 层 TP32，第 60 层 TP128，MoE 层 EP52，共 4,942 片 |
| TSP-CIM | PP78，首尾 TP16，中间 76 段 TP4，共 336 片 | PP61，第 0 层 TP16，第 1–2 层 TP8，第 3–59 层 TP32，第 60 层 TP128，MoE 层 EP52，共 4,942 片 |

Ascend 和 TPU 使用相同的 DeepSeek-R1 层边界。前 30 段各负责连续两层，尾段独占第 60 层和词表投影。两种 TSP 均按一层一段放置 DeepSeek-R1，第 0 层含 Embedding。四种 TP 宽度将 128 个注意力头均分为每片 8、16、4、1 个头。每个 MoE 段的注意力首片同时负责路由、结果合并、共享专家和路由专家 0，另有 51 片专用于其余专家。中间 MoE 段各用 83 片，尾段用 179 片，全模型共 16 + 2 × 8 + 57 × 83 + 179 = 4,942 片。

默认请求的容量峰值来自两步 Decode 的逐片报告，下表按对应的整片 HBM 或单个半片 MEM 统计。

| 架构与存储 | 可用容量（B） | LLaMA3-70B 峰值（B） | DeepSeek-R1 峰值（B） |
| --- | ---: | ---: | ---: |
| Ascend 整片 HBM | 32,000,000,000 | 27,953,668,096 | 31,919,136,496 |
| TPU 整片 HBM | 34,359,738,368 | 24,618,213,376 | 23,400,833,856 |
| TSP 半片 MEM | 115,343,360 | 112,038,520 | 115,153,104 |
| TSP-CIM 半片 MEM | 115,343,360 | 112,038,520 | 115,162,696 |

省略所有放置参数时使用表中策略。`--plan` 可指定各段独立的物理芯片组。显式指定任一并行度参数会使用均匀分片生成器，复现自定义实验时可写全 `--pp`、`--tp`，以及 R1 的 `--ep`、`--etp`。例如 `python ascend.py --model deepseek_r1 --sequence-length 127 --pp 4 --tp 4 --ep 8 --etp 1` 可复现早期短上下文放置。修改 batch、长度或放置后会重新预检容量。

**MLA 与 Dense。** 61 层都执行吸收式 MLA。每个注意力 rank 保留完整的 Q 下投影、latent/位置下投影、压缩归一化权重和两份压缩缓存。Q 上投影、吸收矩阵、值上投影按头切，输出投影按输入通道切并做 INT32 all-reduce。为分别分配内容和 RoPE 通道，模型将 Q 上投影和 KV 下投影分别执行为两个本地 GEMM。片内按输出列分 GEMM，原生补齐和输入读流量随该选择计入。一个 latent backing 同时服务内容打分和加权求和，物理容量取两种原生视图及追加布局的最大值，片内重排交接计零。RoPE 表预先生成；打分比例使用固定配置的 YaRN 系数平方除以 `sqrt(128+64)`。前 3 层执行 Dense SwiGLU，后 58 层进入 MoE。[固定模型配置](../../configs/models/deepseek_r1.json) · [上游注意力实现](https://huggingface.co/deepseek-ai/DeepSeek-R1/blob/56d4cbbb4d29f4355bab4b9a39ccb717a14ad5ad/modeling_deepseek.py)

每层、每 token 逻辑追加 512 B latent 和 64 B 位置分量。默认请求全模型每步逻辑追加 35,136 B，物理追加量再计各层 TP 副本。计算报告中可通过 `chips[].steps[].runs` 找到具体层对应的 `profiles`，查看 `kv.append` 的位置、缓冲区、动作与时间。

**专家放置与路由。** Ascend 和 TPU 的每个 MoE 层将 256 个路由专家及 1 个共享专家全部放在该段的同一片上。TSP 和 TSP-CIM 将共享专家和路由专家 0 放在注意力首片，其余 255 个路由专家按编号连续分到另外 51 组，每片 5 个。共享专家每 token 执行一次，分派和返回均在路由器本片完成。自定义 `--plan` 使用完整的 PP stage 数组，每段含 `layers`、`chips`，可指定有序的 `attention_chips` 和 `expert_groups`（芯片 ID 的数组，每组首片负责返回）。可选 `expert_owners` 是长度 256 的组号数组，`shared_group` 是共享专家组号。分配给一片的专家权重跨步常驻，未命中的专家仍占容量。开启专家 TP 后，gate/up 沿中间列分片，down 的部分和归约完毕再由组首片输出 FP32。

每层由注意力组首片计算门控投影、sigmoid、带偏置的分组选取、top-8 和归一化权重的算子成本。命中 ID 来自固定路由场景，可选 `spread`、`concentrated` 或 JSON 文件。基础 `assignments` 的每一行对应一个 batch 元素，需含 8 个不同 ID，最多涉及 4 个路由组。下面的文件适用于 B=1，覆盖项的 `step`、`layer` 从 0 起算。固定场景用于形状与负载核算，模型不计算数值 logits 或预测命中 ID。

```json
{
  "assignments": [[0, 1, 2, 3, 4, 5, 6, 7]],
  "overrides": [
    {"step": 1, "layer": 3, "assignments": [[0, 16, 64, 80, 128, 144, 192, 208]]}
  ]
}
```

发送方按专家聚合被选中的 token 行，向该专家组各片发送 INT8 激活。组首片按原选择槽位返回 FP32 结果。共享专家单列，注意力组首片执行带权 combine、共享输出相加、存储转换和残差，然后将结果广播到同段注意力片。相同专家组依次收取和返回各专家的批次。路由索引和槽位顺序由场景提供，通信清单按张量有效载荷计字节，协议头和控制消息尚无链路模型。预检为首片保留独立的跨算子交换区（skip、归一化输入、索引、权重、全部路由返回和共享返回），可在 `resident_bytes_per_unit.moe_exchange_buffers` 对账。逐片时长累计本地计算与存储服务。选词、跨片传输和全系统端到端时延留给系统层。

**独立验收。** `scripts/verify_issue4.py` 从四个架构入口读取默认请求和放置，逐项核对两步 KV、层边界、每片头区间、专家归属、count/timing 动作、容量、MAC、通信依赖和超容量拒绝。它还逐段对比 Ascend 与 TPU、TSP 与 TSP-CIM 的放置。汇总保存在 `results/issue4/acceptance/acceptance.json`，其中 `placement_alignment` 记录架构间的对齐结果，`cases` 记录各架构独立的容量峰值。R1 的首个 MoE 层在 Ascend 和 TPU 上的 8 次分派全部落在本片。在两种 TSP 上，内置集中和分散场景都命中专家 0 一次，其余 7 次为远端分派。单层单步的远端路由激活和返回合计 7 × 7,168 × (1 + 4) = 250,880 B，最忙专家组的路由命中数分别为 5 和 1。通信清单只记录有效载荷与依赖，网络时延尚未建模。

## 里程碑三 · LLaMA3-70B 全模型 Decode

独立脚本与四个架构入口使用同一套 LLaMA 默认值，初始 KV 为 8,190。Ascend 和 TPU 的三段分别负责 27、27、26 层。TSP 和 TSP-CIM 将第 0–1 层交给首段，第 2–77 层每层一段，第 78–79 层交给尾段。首段执行 Embedding，尾段执行最终 RMSNorm 和词表投影，共用 16 + 76 × 4 + 16 = 336 片。局部执行沿用各架构的单元数、带宽份额及原生算子计数。片内交接耗时为零。

两种 TSP 的首尾 TP16 各片分到 4 个 Q 头和 1 个 KV 头，每个逻辑 KV 头在两片上各有一份。中间 TP4 各片持有 16 个 Q 头和 2 个独占的 KV 头。默认请求全模型每步逻辑追加 163,840 B，加上首尾四层的副本后，物理追加量为 172,032 B。

```powershell
foreach ($arch in 'ascend', 'tpu', 'tsp', 'tsp_cim') {
    foreach ($mode in 'count', 'timing') {
        .\.venv\Scripts\python.exe scripts/run_llama_model.py --architecture $arch --mode $mode
        if ($LASTEXITCODE -ne 0) { throw "Model failed for $arch ($mode)" }
    }
}
.\.venv\Scripts\python.exe -m unittest tests.models.test_llama_model -v
```

均匀分段时可直接指定 `--pp` 和 `--tp`。用 `python scripts/run_llama_model.py --architecture tsp --pp 20 --tp 16 --sequence-length 127` 可复现下文的短上下文容量失败。通过 `--plan plan.json` 可指定连续、左闭右开的层区间和有序 TP 芯片组。例如下面的放置将前 40 层交给芯片 10 和 12，将后 40 层交给芯片 20 和 22。各段允许有不同的 TP 片数。

```json
[
  {"layers": [0, 40], "chips": [10, 12]},
  {"layers": [40, 80], "chips": [20, 22]}
]
```

**逐片计算与驻留。** 默认计算文件是 `results/issue4/<架构>_llama_model_<模式>.json`。在 `chips` 中按芯片查看层区间、Q/KV 头、FFN 列和端点权重分片。Embedding 表按隐藏列切分，每个 rank 用同一 token ID 只读取所选行的本地列，再 all-gather 完整隐藏向量。尾段在各 rank 计算最终 RMSNorm，按词表列投影，在 rank 0 收齐 FP32 logits。Embedding 和输出权重分别驻留。每层独立持有权重和最终长度的 KV，片内各层共享同一对预计算 RoPE 表。首段、中间段、尾段的常驻字节分类列在 `resident_bytes_per_unit`。

先对所有放置检查常驻分配与活跃张量，再用原生 count 模式检查各算子的工作区和私有存储，全部通过后才开始 timing。容量失败时标准错误中的 JSON 给出芯片、半片或 HBM、stage、层或端点、节点以及所需与可用字节。上述 TSP 短上下文示例使用 PP20、TP16，首段芯片 0 的半片 0 在 lookup 时需要 142,124,400 B，可用 115,343,360 B。

**层级对账。** `schedule` 用 `step.0/layer.17/ffn.up` 这样的 ID 标出每次执行。按 stage 找到对应芯片，使用 `chips[].steps[步号].block` 查 `profiles` 的本地节点，即可取得这一层自己的权重分片和动作次数。具有相同局部形状、KV 有效长度和剩余容量的层可引用同一剖面。累加全模型动作时，每个芯片、每步、每层都计一次，首尾端点各按实际执行次数计入。每片每步的计算时间已按该片层数展开。`step_totals` 给出有效 MAC 和 KV 追加字节的逻辑量及含副本的执行量。

**跨片依赖。** 同目录的 `*.communication.json` 记录事件 ID、生产者、消费者、参与组、形状、dtype 和有效载荷。每层两次 TP all-reduce 使用 INT32 部分和。每次 PP 交接将完整 INT8 激活送达下一段的每个 rank，清单列出每份副本的发送片和接收片。载荷按逻辑张量计，物理链路流量和通信时间由 Issue #6 的拓扑模型确定。第一步收齐 logits 后，在尾段首片保留外部选词依赖，再把 INT32 token ID 送回首段；第二步的每层 KV append 同时依赖该层上一步的更新。选词和跨片事件均保留待计时字段，逐片计算秒数仅覆盖本地算子与原生存储服务。

## 里程碑二 · LLaMA3-70B Decoder Block

以下命令运行一个真实层，分别输出四个 TP rank 的放置、逐算子动作和单片计算时间。

```powershell
foreach ($arch in 'ascend', 'tpu', 'tsp', 'tsp_cim') {
    foreach ($mode in 'count', 'timing') {
        .\.venv\Scripts\python.exe scripts/run_llama_block.py --architecture $arch --mode $mode --layer 0 --tp 4 --batch 1 --sequence-length 127 --decode-steps 2
        if ($LASTEXITCODE -ne 0) { throw "Block failed for $arch ($mode)" }
    }
}
.\.venv\Scripts\python.exe -m unittest tests.models.test_llama_block -v
```

报告位于 `results/issue4/<架构>_llama_block_<模式>.json`。`chips` 给出全局 Q 头、KV 头和 FFN 中间列的左闭右开区间，`query_to_kv` 逐头记录绑定。形状相同的芯片共用一份 `profiles`，其中按 Decode step 存放执行结果。启动时预留终态 KV，每步按有效长度执行冷启动局部算子。测试覆盖小尺寸逐节点手算、真实形状、跨片 KV 复制、容量边界以及错误输出保护。

- **GQA。** 每片按所需 KV 头生成 K、V 投影，按通道对旋转 Q、K，再向原缓存追加一个 token。QK 按通道在片内切 K，以 INT32 树归约后按全局 head dimension 缩放。Softmax 的 MAX、SUM 分别收齐全行局部结果，再计算倒数。PV 在片内按输出通道切分；读缓存的 Q 头共享 KV backing。每个阶段都物化本片所有 query head 的输出，头内工作区复用。
- **行切归约。** Attention 输出投影和 FFN Down 都生成 INT32 部分和。两个 `*.reduce` 事件原地更新部分和，均带参与片数、每片逻辑载荷和整组加法数。消费者等待归约，再做 INT8 转换和残差。网络拓扑与集合通信耗时留给系统层，`local_compute_cycles` 只累计单片本地计算和原生存储服务。
- **片内分配。** 大块 GEMM 沿可分轴均分，余数分给编号较小的单元。两次 RMSNorm 在各 TP 片计算完整隐藏向量，片内分通道求平方和，再由一个单元计算全行比例并广播。RoPE 按完整通道对分配，残差与 SwiGLU 按元素分配。每个并行阶段取最晚单元完成时间，片内的交接和视图重排额外耗时为零。树的末级与标量倒数各自保留真实的参与单元数。
- **容量。** 各片持有本层的权重、最终长度的 KV 和预计算 RoPE 表；分片间重复的 KV 头会增加相应权重、缓存及计算量。每个具名张量的物理分配取各原生视图需求的最大值，计入补齐、条带化和必要的输入复制。各节点记录活跃张量、复用工作区及核内存储。先检查所有常驻张量及活跃张量，再执行原生内核并核对含 scratch 的峰值；MEM 空间紧张时尝试缩小元素块。HBM 按片检查，MEM 逐半片检查。容量失败以退出码 2 和标准错误中的 JSON 给出位置、层级、所需与可用字节及占用明细。

以上述初始 KV 为 127 的请求对账，`TP=4` 每片持有 16 个 Q 头和 2 个 KV 头，每片每步追加 512 B。两步的全组有效 MAC 分别为 857,735,168 和 857,751,552。将 `--tp` 改为 16 可检查 KV 复制：8 个逻辑 KV 头占用 16 个物理头副本，全组每步追加量从 2,048 B 增至 4,096 B。仅校验一个 Decoder Block 的驻留与计算。

## 里程碑一 · Dense FFN 上投影

在工程根目录运行以下命令，可以从四个已有架构后端获取真实投影的逐片结果。

```powershell
foreach ($arch in 'ascend', 'tpu', 'tsp', 'tsp_cim') {
    foreach ($mode in 'count', 'timing') {
        .\.venv\Scripts\python.exe scripts/run_projection.py --architecture $arch --mode $mode --model llama3_70b --layer 0 --tp 4 --batch 1 --sequence-length 127 --decode-steps 2
        if ($LASTEXITCODE -ne 0) { throw "Projection failed for $arch ($mode)" }
    }
}
.\.venv\Scripts\python.exe -m unittest tests.models.test_projection -v
```

报告默认写入 `results/issue4/`。用 `--output` 可指定其他输出文件。全仓测试发现和 `scripts/run_all.py --scope all` 均覆盖本里程碑的测试。

### 配置与范围

两个模型保留上游的完整 `config.json`，加载时校验固定版本的 Git blob。LLaMA3-70B 使用公开镜像获取文件，Meta 同版本对应的配置与镜像文件的 blob ID 均为 `2f26ee5614fe1336b244da1655ba55436f3f1499`。[官方文件目录](https://huggingface.co/api/models/meta-llama/Meta-Llama-3-70B/tree/c82494877ce7f6d7d317c56ec081328e382c72fe) · [镜像配置](https://huggingface.co/NousResearch/Meta-Llama-3-70B/raw/fdbf133d993d2b8157e518bfc8bc4570b2d31c45/config.json) · [DeepSeek-R1 官方配置](https://huggingface.co/deepseek-ai/DeepSeek-R1/raw/56d4cbbb4d29f4355bab4b9a39ccb717a14ad5ad/config.json)

计算范围是指定层的一次 Dense FFN up GEMM，输入和权重采用项目的 INT8 假设，输出为 INT32。LLaMA 可选第 0 至 79 层，DeepSeek-R1 的 Dense 层为第 0 至 2 层。`sequence_length` 表示首步前已有的 KV 长度，`effective_lengths` 记录各步追加一个 token 后的有效长度。上投影形状只依赖 batch 和模型维度，各步复用同一份冷启动估算。报告的 MAC、动作量和耗时均为每步值。

### 逐片对账

1. 在 `chips` 中按芯片编号读取全局输出列区间，所有区间均为左闭右开。每片拥有对应的权重列和输出列，输入激活完整复制到每片、每个执行单元。单元的输入、权重、输出按各自原生分配计入驻留。HBM 架构还对该投影的所有单元分配求和并检查芯片上限。
2. 按 `chip_profile` 查找片内划分。`unit_slices` 的列区间相对于本片起始列，`local_run` 指向一个真实的单核、TensorCore 或单半片结果。形状相同且硬件配置一致的单元复用这一结果。余列按编号分给前面的执行单元，最多相差一列；每个单元至少分得一列。
3. 在 `resources` 中核对单元数和原生带宽份额。单片耗时取参与单元完成时间的最大值。片内交接取零，局部 GEMM 内的存储服务、补齐和流水仍按原生模型计时。
4. 在 `chip_profiles` 中按动作 ID 检查汇总，保留原生动作的单位、计数依据和角色。CIM 的 reference 和 summary 各自单列。为核对有效计算量，按芯片实际引用次数累加 `useful_macs`。LLaMA 示例每片 7168 列、58,720,256 MAC，四片合计 234,881,024 MAC。

本阶段报告的时间范围是一个投影的单片计算。Decoder Block 生命周期、KV 驻留、PP/EP 和跨片通信计时属于后续里程碑。
