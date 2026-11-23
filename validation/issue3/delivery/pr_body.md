# Issue #3：四架构 Decode 粗粒度算子与正式交付验收

Refs #3

## 修改内容

- 为 Ascend、TPU、TSP 和 CIM-TSP 增加统一 Tensor/Operator/Result 与动作计数接口，支持独立向量、非线性、Decode Attention、MLA 和粗粒度 MoE 等算子。覆盖 LLaMA 的 10 个、DeepSeek 的 13 个计算块。
- 将模型预期 block ID 集合与实现证据清单分开，明确拒绝缺项、替换项、额外项、重复项和模型缺失/重复；保留逐块实现文件与实际测试证据校验。
- 新增 `--require-complete` / `-RequireComplete` 正式交付模式：除已实现范围回归通过外，还要求 `full_issue_complete=true`；子脚本和批量汇总均校验，失败返回非零并列出原因。普通回归语义保留，Issue #2 判定独立。
- 修正共享 VXM：整片峰值 5120 op/cycle；根据用户转述师兄当面澄清的半片因素，采用两个半片固定均分、单任务 2560 op/cycle 的模型假设。TSP/CIM 独立向量与 GEMM 跨 K 累加共用此预算；访存和依赖另行计时，操作代价不变。
- 保留 `tile_elements=1024`，澄清其为可覆盖的软件默认策略；受约束的自动搜索归入 Issue #4。
- 沿用 `4802166` 的清理：生成报告和日志按需生成，`validation/issue3/run/` 保持 Git 忽略，固定回归基线继续跟踪。

接口组织参考 Issue #3 指定的 LLMCompass_mod 提交 `98e79ee5b34991f09c65efe395e1e84d58589fab`。

## 验证与复现

Python 3.12+，仓库根目录，仅需标准库：

```powershell
# 普通回归：只要求已实现范围通过
python -B -X utf8 scripts/verify_issue3.py --output-dir results/issue3-regression
# 正式交付：还必须满足完整 Issue #3 算子范围
python -B -X utf8 scripts/verify_issue3.py --require-complete
# 两套验收和全部日常仿真；严格交付只作用于 Issue #3
python -B -X utf8 scripts/run_all.py --scope all --require-complete --details
```

本轮 480 项单测（215 原生 + 265 算子/验收工具）全部通过；Issue #3 普通与正式模式分别通过 216 项 CLI 和 18 项条件，日常仿真 108/108 通过。全量汇总退出 2 仅保留 Issue #2 历史 A1/A5 标记，无自动检查失败。详细验证记录见 [PR #8 修订记录](pr8_followup.md)。验收报告中的时间、运行时 HEAD、源码快照和工作区内容共同标识被验证版本；未提交修改不会体现在 HEAD 中。

64 组 API 基线逐项核对后更新：周期、算子身份、动作和容量保持不变；60 份原生报告的差异来自来源哈希、预算元数据或策略文字。较大形状的计算/访存受限案例另由针对性测试覆盖；历史基线快照和原始产物字段保留。

Issue #2 的 A1/A5 仍按历史本地审阅快照判定；PR #7 已审阅并合入，该状态不表示重新等待其审阅。旧人工签核保持原样。

## 模型边界

当前为 INT8 矩阵/INT32 累加及显式 FP32 非线性和存储转换代理，按单核/TensorCore/半片执行。固定半片均分、全权重驻留、串行物化、符号布局和路由分布均为声明的建模条件；不执行数值路由。完整模型拼装、整片调度/通信/卸载、自动 tiling search、数值精度和硬件/能耗标定归入后续工作。

建议最终使用 Squash and merge，以免把早期提交过的生成报告带入 main 历史。
