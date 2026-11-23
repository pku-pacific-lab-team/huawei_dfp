# Issue #3：已实现算子的独立验证

在工程根目录运行：

```powershell
.\.venv\Scripts\python.exe scripts/verify_issue3.py --require-complete
```

默认输出到 `validation/issue3/run/`，其中 `acceptance.json` 是验收汇总。该目录下的报告及日志不纳入 Git 跟踪，按需运行脚本生成。临时验证可使用 `--output-dir tmp/issue3-experiment`。脚本拒绝输出到 `validation/issue2/` 及其子目录；Issue #2 脚本也不能写入本目录。

普通回归（不带 `--require-complete`）退出 0 表示[覆盖清单](scope.json)的已实现范围通过。正式交付模式还要求 `full_issue_complete=true`，否则退出 1 并输出 `delivery_failures`。报告用 `acceptance_mode=regression|delivery`、`require_complete` 和 `exit_code` 标识本次判定。

`full_issue_complete` 由原始三项验收映射、全部条件、LLaMA3-70B/DeepSeek-R1 共 23 个模型块的代码与实际测试证据、无 Issue #3 内部待办以及源码快照一致性共同决定。预期 ID 集合独立定义在 `scripts/verification.py`：LLaMA 10 项、DeepSeek 13 项；`model_coverage.json` 只提供待检查的证据。缺失、替换、额外或重复 ID，以及模型缺失/重复均失败，具体差异记录在 `delivery_audit.inventory`。该字段仅表示本地粗粒度算子交付，不代表 GitHub 已关闭或 Issue #4 完整模型运行已完成。

日常单算子仿真报告与 Issue #2 一样放在工程根目录 `results/`：全部非 GEMM 算子的默认名称为 `<架构>_<算子>.json`，例如 `ascend_fp32_add.json`。同一架构与算子的 count/timing 共用文件名，成功重跑时覆盖；显式 `--output` 可另存，任务失败时保留旧文件。该目录仍是 Git 忽略的可再生输出，与本目录的验收证据分开。

也可使用 `scripts/run_all.ps1 -Scope issue3` 一次生成四架构 × 26 类操作（GEMM 加 25 个非 GEMM 样例）的 104 份精简日常 timing 报告、每算子一行的 CSV 汇总并执行本验证；`-Scope all` 再包含 Issue #2 的 4 份原生 GEMM 报告，共 108 份，并执行两套独立验证。汇总报告在 `results/run_all_<范围>.json`；GEMM 适配报告另存为 `<架构>_gemm_operator.json`，不覆盖原生报告。

正式批量交付使用 `python scripts/run_all.py --scope issue3 --require-complete --details` 或 `scripts/run_all.ps1 -Scope issue3 -RequireComplete -Details`。汇总记录 `issue3_acceptance_mode`；即使子脚本退出 0，正式模式也会复核完整交付字段。`--scope all` 的严格要求只作用于 Issue #3，保留 Issue #2 原判定；`--scope issue2 --require-complete` 拒绝执行，避免无效选项被忽略。

## 检查与证据

| 范围 | 测试与证据 |
| --- | --- |
| Tensor、OperatorSpec、OpResult、ActionLedger | `tests/operators/test_contracts.py` |
| 四架构 GEMM 适配与原生兼容 | `test_gemm.py`、`test_cli.py`；8 项 CLI 正例 |
| INT32 ADD 与 FP32 ADD/MUL/EXP/RSQRT | `test_vector*.py`；40 项 CLI 正例 |
| FP32 RMSNorm / Softmax | `test_composite*.py`；16 项 CLI 正例，手算周期/字节和符号生命周期 |
| Residual / SwiGLU / FFN | `test_blocks.py`、`test_blocks_cli.py`；24 项 CLI 正例，原生 GEMM 对账、手算工作量/时间/字节、分支生命周期、全块与 scratch 容量、条带化和转换成本敏感性 |
| Decode Attention/GQA 与 KV Cache | `test_attention.py`、`test_attention_cli.py`；16 项 CLI 正例，手算 MAC/op/字节/峰值、缓存追加与 GQA 共享、状态别名及四架构边界 |
| INT8 RMSNorm 与 Attention 投影 | `test_projections.py`、`test_projections_cli.py`；24 项 CLI 正例，手算工作量/访问/峰值、多输出保留、视图共享、类型边界与归一化临时空间 |
| RoPE、Embedding 与词表输出 | `test_position_vocabulary.py`、`test_position_vocabulary_cli.py`；24 项 CLI 正例，通道对、位置与表、整表容量/行访问、FP32 logits 边界及手算周期 |
| MLA 前处理与压缩缓存 | `test_mla.py`、`test_mla_cli.py`；16 项 CLI 正例，独立维度、分支手算、共享缓存状态与容量、原生服务复用 |
| Absorbed MLA Attention | `test_mla_attention.py`、`test_mla_attention_cli.py`；8 项 CLI 正例，五次 head GEMM、FP32 分支合并、共享缓存/权重、输出投影及全块容量 |
| 完整粗粒度 MoE | `test_moe_block.py`、`test_moe_block_cli.py`；24项 CLI 正例，路由、合并及全局容量 |
| 给定路由的 MoE 专家 | `test_moe.py`、`test_moe_cli.py`；8 项 CLI 正例，实际专家负载、空专家容量、FP32 输出、分发/回填、原生 FFN 和全局生命周期 |
| 目标模型块清单 | `test_model_coverage.py`；检查代码/测试证据、缺口归属和覆盖状态；普通缓存不冒充 MLA |
| 名称、axis/epsilon、连接边界、严格校验 | `test_interface_finalization.py`、`test_output.py`；默认路径及显式覆盖检查，8 项 CLI 负例，拒绝空 composite 与未支持操作并保护已有输出 |
| 执行分层与行为兼容 | `test_layering.py`；64 组 API 的原生报告、算子身份和动作语义对账、core 依赖方向、原生结果适配隔离、显式内存路径及 Ascend Vector 入口 |
| 精简报告与独立诊断 | `test_reporting.py`；schema 2、直接访问时间结果、按需保留诊断、容量口径、动作角色和文件绑定 |
| 验证工具隔离与失败传播 | `tests/validation/test_verification.py`、`test_run_all.py`；范围选择、失败后继续、待审状态、旧报告拒绝与全仓库覆盖检查 |

正例矩阵为 4 架构 × 26 操作 × count/timing，共 208 项；加 8 项负例共 216 项。CLI 结构检查与 API 手算/容量/依赖测试共同提供证据。模块缺失、跳过、失败、超时或有测试未登记到覆盖清单时，都不能通过。

生成文件包括 `acceptance.json`（本 Issue 的条件和待办）、`tests.json` / `unittest.txt`（测试结果）、`commands.json`（命令与逐项检查），以及精简 CLI 报告、`details/` 下的独立详细报告、输出日志和负例输入。正例验证显式请求详细输出以检查原生结果与摘要的关联，日常 CLI 默认不导出详细数据。读取结果时同时核对：

- `implemented_scope_passed=true`；
- 正式交付还要求 `acceptance_mode=delivery`、`full_issue_complete=true`、`exit_code=0`；
- `conditions` 全部 passed，测试全部 passed，CLI 全部通过；
- `snapshot_unchanged_during_run=true`，且快照仍匹配当前文件。

快照分为 `issue3` 自有文件和 `shared_dependencies`。后者包含四入口、原生内核、硬件配置、共享文档和验证工具；它们是算子运行与兼容性检查的依赖，不计作 Issue #3 新实现。验证工具测试会检查两个脚本的范围与目录隔离，不运行 Issue #2 的 A1—A5 判据，也不读取其 review/run 作为通过依据。

## 文件归属

Issue #3 的契约与执行编排位于 `src/operators/`、`src/runtime/`；内核包括各架构的 `vector.py`/`composite.py`，以及 `src/core/` 下的 `vector*`、`composite*`、`block_*`、`kv_cache.py`、`moe.py` 和 `operator_result.py`。执行 profile 类型位于 `src/hardware/vector.py`、`blocks.py`、`attention.py`，任务解析和输出位于 `src/utils/operator_*.py`。具体实现与测试映射见 [scope.json](scope.json) 和 [model_coverage.json](model_coverage.json)；执行配置在 `configs/execution/`，样例在 `configs/workload/operators/`，功能测试在 `tests/operators/`。这些执行层和资源装配文件也单列为 Issue #3 自有文件，不加入 Issue #2 的人工审阅快照。

新增执行假设及哈希使用独立[执行参数台账](../../calibration/operator_execution_audit.json)。[硬件参数台账](../../calibration/parameter_source_audit.json)继续维护原生规格，不再追加算子 profile 项。两套脚本复用 `scripts/verification.py` 中的测试记录、文本指纹和目录保护工具；该文件另定义仅供 Issue #3 使用的预期 block ID 集合，Issue #2 不读取它作为验收条件。

Issue #2 仅运行原生架构与 common 测试，仍输出 A1—A5；其 `snapshot` 不包含 Issue #3 专属文件。四入口目前会导入算子分派代码，因此 Issue #2 将这些源码单列到 `dependency_snapshot`，用于运行可复现性，不据此续签或作废 A1/A5 的人工审阅。修改共享入口、原生源码或共享文档，仍可能使 Issue #2 旧审阅失效；脚本不会自动重写其 review.json。

全仓库回归另行运行 `python -m unittest discover -s tests -t . -v`。全仓库通过、Issue #2 验收和 Issue #3 当前范围通过是三种独立结果。

技术模型继续维护在 [ARCHITECTURE.md](../../ARCHITECTURE.md#operator-contract)。计数和解析时间通过测试不等于数值精度验证或硬件标定。

## 重构与报告迁移基线

下文提到的 `run/*_comparison.json` 是历史对账记录，原文件位于清理前的 PR #8 提交 `82971de28ec2cb5b78a42af6d6641a550e138b72`，不属于当前验收脚本的输出。`baselines/` 中的固定回归基线继续纳入 Git 跟踪。

`baselines/layering.json` 与 `run/layering_comparison.json` 保留前一轮分层重构的历史证据：当时的 64 组 API 结果和 100 份文件完整一致。它们不代表新报告格式的验证结论。

当前 [baselines/report_format.json](baselines/report_format.json) 在精简报告改动前采集，包含源码快照、64 组 API 的原生报告/算子身份/完整动作语义哈希，以及 100 份报告的原生结果哈希和旧大小。API 矩阵为四架构 × 八类操作 × count/timing；非 GEMM 使用 `[2,7]` 和 4 元素 tile，覆盖跨块与尾块。普通验证不重写基线。

2026-11-20 同步 main 的 `7c84187`（包含 PR #7 修复 `d49d258`）后，人工核对并更新了 3 组 API 预期：Ascend GEMM timing 的 L1 释放条件说明，以及 CIM-TSP GEMM count/timing 的 `useful_mac` 逻辑分类和内部动作引用。用同一份 Issue #3 实现分别加载修复前后的原生模块，确认修复前 64 组结果全部匹配原始基线，修复后只有上述差异；这些用例的数值计数、周期和算子身份均未变化。旧/新哈希和逐字段差异保存在基线的 `reviewed_updates` 中。原始采集时间、`source_snapshot`、100 份 `artifacts` 及 `run/report_format_comparison.json` 保留为报告迁移的历史证据，不代表同步 main 后的当前验证；当前结果以重新生成的 `run/acceptance.json` 和 `results/run_all_all.json` 为准。

既有回归继续验证 64 组 API 的原生结果与接口语义。报告迁移阶段曾完整重跑 `scripts/run_all.ps1 -Scope all -Details` ，对当时的 36 份日常报告和 64 份 CLI 正例分别取原生结果对账，记录在 `run/report_format_comparison.json`。Issue #3 通过独立详细文件取得原生结果，Issue #2 直接使用原始报告。外层格式有意从 schema 1 改为 schema 2，因此不再要求整份输出文件哈希相同。

精简结果保留一次动作计数及必要的作用域、单位、dtype、basis 和 role；完整原生快照、来源和 io_contract 在可选详细文件中。默认 API 不保留原生快照，诊断需要 `include_details=True`。详细文件的 `result_sha256` 与摘要绑定，验证会检查绑定、原生周期和动作计数。

Issue #2 输出是否需要调整，另见[报告评估](../report_output_review.md)；该评估不改变其报告格式或验收判据。

## 分批证据的阅读约定

以下第四批至第八批 A 记录对应批次完成时的范围、数量和待办，保留历史状态。当前范围以本文开头、[第八批 B 交付核对](#issue3-delivery)及实际验收快照为准。

## 第四批证据范围

第四批将三种块接入四架构的 count/timing，当时独立 CLI 矩阵为 88 项正例及 8 项负例，共 96 项。该批 `run_all -Scope all` 生成 48 份日常报告（Issue #2 的 4 份 + Issue #3 的 44 份），再运行两套验证。新增测试不改写 64 组历史 API 基线；既有算子输出应继续匹配。

`blocks.json` 的转换与 RCP 成本明确为假设，符号布局和单核/单半片范围记录在报告中。验收仅覆盖这组明确策略下的工作量、依赖、容量及解析时间，不证明量化数值、真实布局转换、整片性能或整个 Issue #3 已完成。

## 第五批证据范围

第五批加入单 token Decode Attention/GQA 和预分配 KV Cache 访问，当时 CLI 矩阵共112项；`run_all --scope all` 生成56份日常报告，再运行两套验证。重点验证只追加新 token、读取有效前缀、GQA 存储共享与读取重复的区别、符号 head 视图绑定、串行 head 工作区复用及精度边界成本。

新增 [model_coverage.json](model_coverage.json) 区分两个目标模型的 implemented/partial/pending 块，纳入验证和源码快照。第五批当时 `full_target_model_coverage=false`，普通 GQA/Cache 不表示 DeepSeek-R1 MLA 已覆盖；该批之后的投影边界、RoPE、embedding、MLA、MoE 等按清单继续推进；当前进展见第八批 B。完整模型执行仍属于 Issue #4。

本批还排除了资源摘要中误列的 `lifetime_validation` 项：它是容量校验证据，不是物理内存。此前第四批的真实峰值、原生计数和时间不变；所有既有 64 组历史 API 基线继续回归。

## 第六批 A 证据范围

新增 `rmsnorm_int8`、`qkv_projection`、`attention_output_projection`，复用已有 GEMM、RMSNorm 和存储边界代理。该批独立 CLI 矩阵为136项，`run_all --scope all --details` 生成68份日常报告，并分别执行两套验证。该批完成时，目标覆盖矩阵将常规 INT8 归一化和 LLaMA QKV/输出投影更新为 implemented；DeepSeek MLA 组合当时尚未完成。

新增测试检查同时存活的三个最终输出、head/matrix 视图不重复分配、全块容量超限和 TSP 归一化条带化行状态/临时空间。既有64组 API 基线保持不变，刷新后的第五批原生 CLI 结果与本地提交 `b50e82c` 对账记录在 `run/batch5_comparison.json`。RoPE、Embedding 和词表输出边界随后由第六批 B 实现，完整模型与整片执行仍不在本次完成范围。

## 第六批 B 证据范围

新增 `rope`、`embedding`、`vocabulary_projection`，四架构 CLI 正例共152项，加8项负例共160项；`run_all --scope all --details` 生成80份日常报告并运行两套验证。RoPE 使用调用方提供的 FP32 表，Embedding 以完整词表校验容量并按选中行计流量，词表投影以独立 FP32 输出保存 logits。

第六批 A 已保存为本地提交 `df3501e`。既有128份 CLI 正例的摘要与原生报告对账记录在 `run/batch6a_comparison.json`；历史64组 API 基线不改写。本批来源、假设和 scratch 使用方式另列于执行参数台账；共享执行 profile 的参数与哈希保持不变。

第六批 B 完成时，LLaMA 清单内独立粗粒度块已覆盖，DeepSeek 当时仍有 MLA/MoE 缺口；该批的 `full_target_model_coverage` 与 `full_issue_complete` 均为 false。implemented 不代表完整模型权重能在单核/单半片驻留，也不表示完成真实张量计算、量化精度或端到端模型拼装。

## 第七批 A 证据范围

第六批 B 已保存为本地提交 `cbcf281`。本批新增 `mla_preprocess`、`mla_cache`；CLI 为168项正例和8项负例，共176项；`run_all --scope all --details` 生成88份日常报告并运行两套验证。既有152份正例摘要与原生报告的对账记录保存在 `run/batch6b_comparison.json`，64组历史 API 基线保持不变。

新增测试核对三次原生 GEMM、两次 FP32 归一化、两个位置旋转分支及选中通道的物化读写/转换。共享缓存区分 capacity、length、append_tokens；latent 与位置 K 均不按 query head 数复制，缓存状态输出复用原 backing。两个不同 latent GEMM 角色按最大补齐视图预留一次；读取只覆盖有效范围，不含未来 Attention 自己的 per-head 读取。

来源和执行假设见 `calibration/operator_execution_audit.json` 的 `mla_preprocessing_and_cache`。矩阵、Vector、转换及 cache profile 参数不变。INT8、串行物化、符号布局和复用策略是项目假设；没有官方 FP8/BF16 数值复现或新硬件测量。第七批 A 完成时，DeepSeek 清单中的 MLA RoPE/压缩缓存为 implemented，MLA projections 为 partial，Attention 核心为 pending；这些剩余投影与核心工作随后由第七批 B 显式计入。

## 第七批 B 证据范围

第七批 A 已保存为本地提交 `b086b36`。本批新增 `mla_decode_attention`，四架构 CLI 共176项正例、8项负例，总184项；`run_all --scope all --details` 生成92份日常报告并执行两套验证。与第七批 A 的168份正例摘要/原生结果对账记录保存在 `run/batch7a_comparison.json`；64组历史 API 基线不改写。

MLA 核心采用每 head 五次原生 GEMM，分别计入 Q→latent、两路打分、加权 latent 及 latent→V。FP32 合并/缩放/Softmax 和所有 INT8 边界明确计费；共享缓存仅追加一次，没有额外独立扫描。可选输出投影复用已有节点并纳入完整块的容量校验；不提供 Wo 时输出 context。新测试核对1314 MAC、120个非矩阵 op、TPU读写2474/1386 B、941 B峰值，以及共享权重/缓存、有效长度、显式 scale、位置绑定、生命周期、容量拒绝、四架构入口和输出保护。

执行假设见台账 `mla_absorbed_attention`。原生硬件与三个共享执行 profile 的参数/哈希保持不变。MLA 相关清单在所述单 token、INT8、absorbed 和符号布局条件下更新为 implemented；第七批 B 完成时，目标模型仍有 MoE 缺口，该批 `full_issue_complete=false`。

## 第八批 A 证据范围

第七批 B 已保存为本地提交 `ff44d8c`。本批新增 `moe_experts` 和可选 FP32 FFN 输出；CLI 为184项正例、8项负例，共192项，日常 `all` 生成96份报告。已有176份正例的摘要与原生结果对账保存在 `run/batch7b_comparison.json`；64组历史 API 基线不改写。

给定路由 `[B,top_k]` 中每行索引必须互异且有效。按实际 token 数执行各专家，空专家跳过；全部专家权重仍驻留。输入行分发与 FP32 结果回填明确计流量；输出保留原 token/slot 顺序，另返回共享分支结果。第八批 A 完成时，路由器与加权合并仍为 pending，该批 `full_issue_complete=false`；第八批 B 随后补齐。

样例 B=3、H=5、I=7、E=5、top_k=2、shared=2，专家负载 `[3,1,2,0,0]`：1260 MAC、504个非矩阵 op、FP32/INT8 边界213/84元素。TPU读写1725/1266 B，峰值1272 B；容量1271 B必须拒绝。新增测试覆盖不均匀分配、token/slot 守恒、空专家权重容量、batch/top_k 缩放、原生 action/role、条带化与 scratch 回退、未知服务/时钟、四架构 CLI 和错误输出保护。

执行假设见 `calibration/operator_execution_audit.json` 的 `moe_fixed_routing_experts`；`src/core/moe.py` 归属 Issue #3。原生硬件、三个执行 profile 的参数/哈希不变；INT8、串行、全权重驻留及符号分发属于本项目假设。


<a id="issue3-delivery"></a>

## 第八批 B 与 Issue #3 交付核对

第八批 A 已保存为本地提交 `8533f2c`。本批增加 `moe_router`、`moe_combine`、`moe`；日常 all 为108份报告，独立 CLI 为208项正例+8项负例。前184份正例的摘要/原生结果与该提交比较，证据保存于 `run/batch8a_comparison.json`；64组历史 API 基线继续原样使用。

对照 [Issue #3 原文](https://github.com/pku-pacific-lab-team/huawei_dfp/issues/3)（2026-11-20 核对），三项验收分别映射为：独立 Vector执行与分层测试；GEMM操作数/存储/边界及 Decode形状测试；全部目标块的实现、测试及归属清单。动作类型、资源/延迟报告、存储边界与粗粒度假设仍由其余登记条件共同约束。LLMCompass参考版本继续固定为 `98e79ee5b34991f09c65efe395e1e84d58589fab`，仅用于接口/形状/结果组织参考。

`evaluate_delivery` 逐块检查实现文件存在、实际测试记录全部通过和 owner/status；缺模块、未执行、失败、缺验收映射、未登记证据或仍有内部待办都会阻止完成判断。`full_target_model_coverage=true` 表示10个 LLaMA块与13个 DeepSeek块在所述解析假设下覆盖；`full_issue_complete` 还取决于本次运行的全部验证与快照。此结论不代替新的人工代码审阅，也不改变 GitHub 状态。

保留边界：INT8性能假设、单核/单半片、全权重驻留、符号布局和路由分布场景。完整 Decoder/模型拼装、真实规模的整片分片与供给、跨资源通信以及端到端评估另行推进；数值精度、ISA与硬件/能耗标定不纳入这次解析验收。它们登记在 `follow_up_outside_issue3`，不以已实现算子覆盖来冒充已完成。

## PR #8 修订后的性能口径

中央 VXM 整片理论峰值为 5120 op/cycle，默认单半片固定分配 2560 op/cycle；这是均分建模假设，不代表实测持续吞吐。TSP/CIM 的向量与跨 K 累加共用预算，访存及依赖另行计时。默认 `tile_elements=1024` 是可覆盖的软件策略，不是硬件上限或已验证最优值。受容量约束的自动分块/调度搜索留在 Issue #4，不属于本轮交付条件。

本轮基线逐项比较保留在 `baselines/report_format.json` 的追加更新记录：64 组小形状 API 的周期、动作及容量均不变，60 份原生报告仅因来源哈希、预算元数据或策略说明更新而变化。原始快照、历史产物字段和 `layering.json` 保留原义。较大形状下的性能变化由原生 VXM 和 Vector 测试分别核验计算占用、访存服务及动作不变性。
