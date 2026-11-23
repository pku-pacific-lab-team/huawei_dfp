# 2026-11-23 文档同步记录

本次仅修订配套文档，并用现有脚本重新生成验收证据。仿真实现、测试、硬件/执行配置、来源台账和 Issue #2 旧人工审阅快照保持原样。

## 修订范围

- 验证总说明区分 `implemented_scope_passed` 与 `full_issue_complete`，明确 23 个目标模型块的本地交付边界。
- Issue #3 运行矩阵统一为 26 类操作、104 份日常报告；含 Issue #2 时为 108 份。独立 CLI 验收为 208 个正例和 8 个负例。
- 统一结果按 schema 2 描述，详情按 `operator_details` schema 1 描述；明确生命周期和完整连接信息的读取位置。
- 输出命名按单输出、多输出及缓存状态契约区分；补充 FFN 可选 FP32 输出及其最后转换路径。
- 同步 MLA、MoE、TSP 复合块的实现状态；A5 负例准确限定为未支持的完整 `transformer` workload。
- 补全章节目录、文件职责和依赖说明；旧批次的 false/pending、旧 schema、历史报告大小与审查提交均明确作为历史记录保留。
- PR 正文与审查说明区分最初的实现验收、后续文档变更及当前 `run/` 证据。

## 验证与证据更新

文档变更前，已在 `ca3d20f` 工作区重新完成全部 474 项测试、233 项 CLI 检查（Issue #2 为 17 项、Issue #3 为 216 项）和 108 项日常仿真；60 条 CLI 示例及 8 段 Python 示例通过。该轮检查支持文档修订的事实依据，但其快照不代表修订后的文件。

README、ARCHITECTURE 等说明文件参与验收快照。修订后使用下列既有命令刷新正式 `run/` 证据，不手工改写旧报告哈希或续签人工审阅：

```powershell
python -B -X utf8 scripts/run_all.py --scope all --details
```

结果入口为 [Issue #2 验收](../../issue2/run/acceptance.json)及运行后生成的 `validation/issue3/run/acceptance.json`；本地汇总在 `results/run_all_all.json`。两套测试集合的完整覆盖由汇总的 `combined_test_coverage` 核对。此次文档同步对应的历史 Issue #3 结果位于清理前的提交 `82971de`；当前 Issue #3 运行结果不再提交。

本次刷新结果：

- 474/474 项单元测试通过，两组测试并集覆盖全部发现的测试，无遗漏或重复。
- Issue #2 CLI 17/17、Issue #3 CLI 216/216 通过；Issue #3 全部 18 项条件通过，`implemented_scope_passed=true`、`full_issue_complete=true`。
- 108/108 项日常仿真通过，JSON 内容与文档修订前的检查结果一致。
- 两套验收快照、Issue #2 运行依赖快照及总运行快照均匹配当前文件，运行期间没有快照变化。
- 346 个本地文件/锚点链接有效，8 段 Python 示例通过，`git diff --check` 通过；代码、测试、配置和基线无差异。

统一运行退出码为 2，仅来自 Issue #2 A1/A5 的旧本地审阅证据；没有执行失败项。链接/示例核验、报告对账与运行日志保存在本地忽略目录 `results/documentation-update-20261123/`，该目录是本次检查的补充记录，不是运行依赖。

生成记录的 `repository_head` 是运行时提交；如果文档尚未提交，应结合 `snapshot` 判断具体文件版本，不只看 HEAD。Issue #2 A1/A5 的旧审阅状态不属于本次文字修订的通过条件，也不代表 PR #7 尚待审阅。

本次验证结论只覆盖声明的粗粒度解析模型。完整模型、整片执行、真实数值精度和硬件/能耗标定仍在后续范围。
