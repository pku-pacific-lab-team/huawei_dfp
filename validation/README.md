# 验证记录与历史归档

Issue #2 与 Issue #3 分别维护入口、覆盖范围、判据和生成结果。全仓库回归使用 `python -m unittest discover -s tests -t . -v`，不能替代任一 Issue 的逐项验收。

| 范围 | 入口 | 结果及通过字段 |
| --- | --- | --- |
| Issue #2 原生模型 | `python scripts/verify_issue2.py` | [issue2/run/acceptance.json](issue2/run/acceptance.json)：`local_acceptance_passed` 与 A1—A5 |
| Issue #3 已实现算子 | `python scripts/verify_issue3.py` | 运行后查看 `issue3/run/acceptance.json`：`implemented_scope_passed` 表示登记范围通过；`full_issue_complete` 另核对原始验收映射、23 个模型块、内部待办及快照，仅表示本地粗粒度算子交付 |

统一运行入口为 `scripts/run_all.ps1 -Scope issue2|issue3|all`，默认 `all`；各范围先运行对应日常 timing 仿真，再调用表中验证脚本。总结写入 `results/run_all_<范围>.json`，原始日志写入 `results/run_all/<范围>/`。`all` 会核对两组测试并集覆盖全仓库，不重复执行整套测试。退出码 0 表示所选范围通过，1 表示实际失败，2 表示仅 Issue #2 本地审阅证据需对齐；该状态不等同于 GitHub PR 尚待审阅，任一组失败或待审都不阻止其余选中组执行。

Issue #3 的具体范围、来源和剩余缺口见[验证说明](issue3/README.md)及[覆盖清单](issue3/scope.json)。当前清单的 23 个模型块均为 implemented，Issue #3 内部待办为空；实际完成结论仍须由对应版本的验收生成，不代表 GitHub 已关闭、完整模型拼装或整片执行完成。两个脚本都保存文本规范化的文件快照；判断结果时核对其范围和 `snapshot_unchanged_during_run`，历史通过状态不能直接沿用。

[issue2/review.json](issue2/review.json) 保存原生模型来源和迁移说明的语义审阅依据。自动测试不会更新该审阅快照，原生或共享文件变化后，A1/A5 可能显示 `needs_review`；Issue #3 专属文件不进入该判据。Issue #3 不读取这份审阅记录或将其状态作为自己的通过条件。

PR #7 已由师兄（Bwoah-Kimi）审查并合入 main，修复提交为 `d49d258`，合并提交为 `7c84187`。[审阅状态标注](issue2/review_context.json)记录 PR、提交及适用范围；Issue #2 验证和统一运行报告会附带 `review_context`，它只解释状态，不参与通过判定。后续不要因旧快照的 `needs_review` 重复提示“还在等待师兄审阅 PR #7”；本地证据维护、新问题和后续 Issue #3 改动分别处理。

| 位置 | 用途与保留规则 |
|---|---|
| `issue2/run/` | Issue #2 默认验收输出；只由其脚本生成和覆盖 |
| `issue2/review.json` | 审阅依据，持续保留；完成实际审阅后才能更新 |
| `issue2/review_context.json` | 已合入 PR 的审阅事实与范围说明；不自动续签 A1/A5 |
| `issue3/run/` | Issue #3 默认验证输出；由验收脚本生成和覆盖，不纳入 Git 跟踪 |
| `issue3/baselines/` | 固定回归基线；保留在 Git 中，验收脚本不覆盖 |
| `issue3/scope.json` / `issue3/README.md` | 已实现范围、测试/CLI 映射和剩余缺口；作为源码维护 |
| [archive/2026-11-18-cleanup/](archive/2026-11-18-cleanup/README.md) | 本次清理前的验证材料、迁移基线、旧源码备份和一次性脚本；原路径保存在压缩包内 |
| `../results/` | 可再生仿真报告，不是验收依据；使用 `scripts/clear_results.ps1` 清理 |
| `../tmp/` | 临时工作目录；需要保留的基线必须随验证材料一起归档，不能只记录一个被 Git 忽略的路径 |

以后需要保留历史验收时，将原始输出及依赖的基线一并归档，记录原路径、文件哈希、对应版本和当时的验收状态，再更新对应 Issue 的输出。同一 Issue 避免并列维护多个声称代表当前状态的 `run/` 目录。临时实验可使用 `--output-dir results/<实验名>`；两个脚本均拒绝输出到对方 Issue 的证据树。

硬件规格归档另见 [configs/archive/](../configs/archive/README.md)。原生参数来源使用[硬件台账](../calibration/parameter_source_audit.json)，Issue #3 新增执行策略使用[执行参数台账](../calibration/operator_execution_audit.json)；这些都不属于生成文件的清理范围。
