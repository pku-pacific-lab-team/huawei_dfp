现有四架构入口以原生 GEMM 为主，缺少描述 LLaMA3-70B 与 DeepSeek-R1 Decode 大块计算的统一算子层。本 PR 补齐独立 Vector/Nonlinear、复合块及其资源、延迟和动作报告，使目标模型清单中的 23 个粗粒度块均有实现与测试。

Refs #3。目标分支为 `main`，工作分支为 `dfp-issue3-pr`；实现基于已合入 PR #7 的 `7c84187`，实现交付提交为 `59c8331`。

## 实现范围

- 增加 Tensor/Operator/OpResult 与 ActionLedger，区分逻辑工作、补齐工作、内部动作和 summary/reference，能耗系数与动作计数分离。
- 四架构支持独立 Vector、RMSNorm/Softmax、残差、SwiGLU/FFN、归一化与投影边界、RoPE、Embedding 和词表投影。
- 增加 Decode GQA/KV Cache，以及 DeepSeek absorbed MLA 的低秩前处理、共享压缩缓存、双路打分、加权 latent 和输出投影。
- 增加 MoE Gate/Sigmoid、分组与专家选择代理、原始分数归一化、实际分配量对应的专家执行、FP32 加权/共享分支合并，以及整块生命周期校验。
- 增加独立 Issue #3 验收和模型覆盖清单；紧凑报告与详细诊断分文件保存，哈希绑定。既有原生 GEMM 内核、硬件参数和 Issue #2 旧审阅快照保持原样。

## 验证与复现

在 Python 3.12+ 环境、仓库根目录运行；仿真与测试只需标准库：

```powershell
python -B -X utf8 scripts/verify_issue3.py --output-dir validation/issue3/reproduction
python -B -X utf8 scripts/run_all.py --scope all --details
```

已保存的结果：

- 474 个测试通过：Issue #2 为 215 个，Issue #3 为 259 个。
- Issue #3 CLI 216/216 通过；日常仿真 108/108 通过。
- 64 组历史 API 基线通过；第八批 A 的 184 份 CLI 正例摘要及原生报告完全一致。
- 三项原始验收要求已映射，LLaMA 10 个块、DeepSeek 13 个块均有代码与实际测试证据，`full_issue_complete=true`。
- 最终审查另做了 88 组跨形状案例，共 176 次 count/timing 仿真，工作量公式、动作一致性及容量检查均通过。

证据见 `validation/issue3/run/acceptance.json`、`commands.json`、`batch8a_comparison.json` 和 `validation/issue3/baselines/report_format.json`。第八批 B 首次验收生成于 `8533f2c` HEAD 下的工作区，当时已核对其源码快照与实现提交 `59c8331` 一致。2026-11-23 同步使用说明、接口契约和历史状态措辞后，重新运行既有验收以更新文档快照；当前 `run/` 的生成时间、运行时 HEAD 与文件快照以报告本身为准。仿真实现、测试及配置未修改，详见[文档同步记录](documentation_update.md)。

`verify_issue3.py` 预期返回 0。`run_all --scope all` 当前返回 2，仅来自 Issue #2 的 A1/A5 历史本地审阅快照未对齐；其 215 个自动测试通过。PR #7 已完成审阅并合入，此状态不表示再次等待其审阅。旧快照未被重写为通过。

## 模型边界与审查入口

当前为 INT8 矩阵/INT32 累加及显式 FP32 非线性和存储边界代理，按单核/TensorCore/半片执行。全权重驻留、串行物化、符号布局和路由分布均是明确的建模条件；MoE 计入路由服务，但实际专家 assignments 由调用方提供，不执行数值路由。完整 Decoder/模型拼装、整片分片/通信/卸载、真实数值精度和硬件/能耗标定另行推进。

建议先审阅 `src/operators/` 的契约、`src/runtime/` 的组合和容量计划、`src/core/` 新增 Vector/复合内核，再看 `scripts/verify_issue3.py` 与 `validation/issue3/model_coverage.json`。详细阅读顺序、复核命令和审查记录见 `validation/issue3/delivery/review.md`。大量 `validation/**/run/` 差异为保存的验收报告及日志，可与实现分开阅读。
