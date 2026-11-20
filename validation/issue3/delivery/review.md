# Issue #3 本地最终审查与交付记录

本轮未发现需要修改实现的阻断性问题。第八批 B 已保存为 `59c83310733d31f1ea6219ee454d252eb5d7716b`，随后完成了相对 `main` 的分支审查、源码快照复核和补充跨形状检查。这里记录的是本地代码审查与解析验收，不代替维护者对本 PR 的审阅，也不改变 GitHub 状态。

## 版本与范围

| 项目 | 核对结果 |
| --- | --- |
| 目标基线 | `7c8418707f24d8e6b7148a4990b39e88b8d4d344`，PR #7 合并提交 |
| 实现交付 | `59c83310733d31f1ea6219ee454d252eb5d7716b`，共领先基线 9 个提交 |
| 远端核对 | `git ls-remote origin refs/heads/main` 与上述基线一致；本轮仅只读访问远端 |
| 第八批 A | `8533f2c22a851f17eff2b99b5176f4375d2123c1`，历史 CLI 对账基线 |
| 实现分支差异 | 1016 个文件，其中 126 个为代码/入口/脚本/配置/测试，881 个为 run/baselines 证据，其余为文档/审计材料；不含本次交付说明文件 |
| 原生边界 | 原有四架构 GEMM simulator、硬件配置/规格与 `validation/issue2/review.json` 无差异 |

## 阅读与审查记录

| 顺序 | 入口 | 本轮核对重点与结果 |
| --- | --- | --- |
| 1 | `src/operators/types.py`、`actions.py`、`adapters.py`、`connections.py` | 存储 dtype、Tensor 身份、输出别名、动作来源与单位；summary/reference 不作为新增动作重复合计；连接不自动承诺转换、布局或驻留复用。 |
| 2 | 四个顶层 CLI、`src/runtime/cli.py`、`simulation.py`、`workloads.py` | 原生 GEMM 路径兼容；独立 Vector 无需 GEMM；非法选项/输入在报告写入前拒绝；摘要与详情使用不同路径及哈希绑定。 |
| 3 | `src/core/vector*`、各架构新增 vector/composite、`composite*.py` | 独立算术、归约树与行状态，尾块及单元素边界；本地 DMA 与 MEM/stream 分开计数；缺少所用服务时 timing 明确失败，count 可独立执行。 |
| 4 | `src/runtime/blocks.py`、`src/core/block_schedule.py`、投影/RoPE/Embedding/FFN | 输入持久保留、中间结果最后使用后释放、输出与输入重叠；native padding 和按分配条带化；广播/视图与真实存储转换区分。 |
| 5 | `src/runtime/attention.py`、`mla.py` 及对应 operators | GQA 的共享 KV backing，MLA latent/位置缓存共享，追加一次、原生读取按 head 重复；INT32 score 先转 FP32 再相加与缩放；输出投影使用整块存活容量。 |
| 6 | `src/runtime/moe.py`、`moe_block.py`、`src/core/block_moe.py` | 空专家不执行但全权重占容量；偏置只影响选择；权重取原始 sigmoid；route weights 跨专家阶段保留，indices 在合并前释放；检查整块全局峰值；嵌套动作保留角色和来源。 |
| 7 | `scripts/verify_issue3.py`、`verification.py`、`run_all.py`、两份 Issue #3 清单 | 完成标记需实际测试、完整验收映射、实现文件、模型清单和无内部待办；缺失/跳过/失败/未登记证据不能通过；旧报告不能满足新运行；Issue #2 与 #3 判据保持分离。 |

审查以运行契约、工作量/存储公式和已有反例测试为依据。JSON 生成证据通过绑定与快照核验，不把逐行阅读大量重复日志作为正确性证明。

## 验证证据

| 项目 | 结果 |
| --- | --- |
| 单元测试 | 474/474：215 个原生测试、259 个算子与验证工具测试 |
| Issue #3 CLI | 216/216：208 个正例和 8 个负例 |
| 日常仿真 | 108/108：4 份原生 GEMM、104 份统一算子报告 |
| 历史 API | 64 组，原生结果/Operator/账本基线保持一致 |
| 第八批 A 兼容性 | 184 份 CLI 摘要与原生报告均一致，见 [batch8a_comparison.json](../run/batch8a_comparison.json) |
| 模型覆盖 | LLaMA 10 个块、DeepSeek 13 个块，全部有实现与实际测试证据 |
| 交付判据 | [acceptance.json](../run/acceptance.json) 的 `implemented_scope_passed=true`、`full_issue_complete=true` |
| 补充审查探针 | 88 组形状、176 次仿真；两种模式动作和容量相同，时间有效，解析公式与结果相符 |

补充探针的范围如下，四种架构均执行 count/timing，tile 上限为 3：

- RMSNorm/Softmax：batch 为 1、3，行宽为 1、7、33；核对逻辑运算 `B*(4D+2)` 和 `B*(5D-1)`。
- FFN：`(B,H,I)=(1,3,5)`、`(3,17,7)`，分别输出 INT8/FP32；核对 `3BHI` MAC 与 `6BHI+6BI` 逻辑运算。
- GQA：`(B,Hq,Hkv,D,Lbefore)=(1,1,1,3,0)`、`(3,6,2,17,17)`，追加一个 token；核对 `2*B*Hq*Lafter*D` MAC。
- MLA：`(B,Hq,Lbefore,Rkv,Dnope,Drope,Dv,H)=(1,1,0,3,1,2,5,7)`、`(3,2,17,17,5,6,3,19)`，含输出投影；核对五个 head GEMM 和输出投影的独立 MAC 公式。
- MoE：`H=5,I=7,E=6,G=3,Lgroups=2`，分别为无 shared、`assignments=[[1]]`，以及一个 shared、`[[0,2],[3,4]]`；核对 Gate 与实际专家分配量对应的 MAC。

这些探针是本轮补充审查，没有增加正式单元测试数量或修改既有基线。其检查只支持上述解析模型条件，不是数值精度、真实设备吞吐或布局正确性的实测证明。

## 源码快照与提交的对应关系

全量证据生成时 HEAD 是 `8533f2c`，第八批 B 当时仍在工作区；验收快照包含这些未提交文件。第八批 B 提交为 `59c8331` 后，已重新比较三个源码快照，均一致。`repository_head` 保留运行时真实值，不改写成后来的提交，也不只用 HEAD 判断证据有效性。

本轮未修改模型、测试、执行配置或原有验收文档；新增 delivery 材料是审查说明，不进入当前仿真源码快照。因而复用了已经通过的全量验收，只额外运行针对最终审查的跨形状探针和快照检查。

在仓库根目录可直接核对已保存证据与当前源码，无需依赖本机 tmp 文件：

```powershell
@'
import json
from pathlib import Path
from scripts.run_all import source_snapshot
from scripts.verify_issue2 import snapshot as issue2_snapshot
from scripts.verify_issue3 import snapshot as issue3_snapshot
read = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))
i2 = read("validation/issue2/run/acceptance.json")
i3 = read("validation/issue3/run/acceptance.json")
assert i2["snapshot"] == issue2_snapshot()
assert i3["snapshot"] == issue3_snapshot()
assert i3["implemented_scope_passed"] and i3["full_issue_complete"]
batch = Path("results/run_all_all.json")
if batch.exists():
    assert read(batch)["snapshot"] == source_snapshot()
print("Saved source snapshots match")
'@ | python -B -X utf8 -
```

`results/` 是本地忽略输出，干净克隆中没有它时，仍可验证已提交的两份 Issue 快照；运行下方全量命令会重新生成 batch 结果。

## 复现与后续交付

使用 Python 3.12+，无需第三方库。下列命令从仓库根目录执行；Windows 可将 `python` 替换为 `.venv\Scripts\python.exe`。

```powershell
# 独立验收 Issue #3，写入独立复现目录，预期退出 0。
python -B -X utf8 scripts/verify_issue3.py --output-dir validation/issue3/reproduction

# 两套验收及全部日常报告，包含详细诊断。
python -B -X utf8 scripts/run_all.py --scope all --details

# 检查分支代码差异，优先阅读实现，生成证据另行查看。
git diff origin/main...HEAD -- src scripts configs tests ascend.py tpu.py tsp.py tsp_cim.py
git diff --check origin/main...HEAD
```

第二条命令当前预期退出 2：所有自动检查通过，但保留 Issue #2 的 A1/A5 旧本地审阅证据状态。PR #7 已审阅、修复并合入；该返回码不是其待审标记，也不否定 Issue #3 的独立结论。完整执行将更新既有 run 日志和证据，属于复现生成文件。

当前交付边界为单资源、INT8 性能假设、全权重驻留和显式路由分布场景。完整模型拼装、整片分片/通信、专家卸载、真实数值计算及硬件/能耗标定保留为后续工作。

本地 PR 标题见 [pr_title.txt](pr_title.txt)，正文见 [pr_body.md](pr_body.md)。本轮完成本地提交和材料整理；远端推送、创建 PR、合并或关闭 Issue 均未执行。
