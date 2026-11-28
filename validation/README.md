# 验证记录与历史归档

Issue #2、#3、#4 分别维护专项验收。全仓库单元测试使用 `python -m unittest discover -s tests -t . -v`。

| 范围 | 入口 | 结果及通过字段 |
| --- | --- | --- |
| Issue #2 原生模型 | `python scripts/verify_issue2.py` | [issue2/run/acceptance.json](issue2/run/acceptance.json)：`local_acceptance_passed` 与 A1—A5 |
| Issue #3 算子交付 | `python scripts/verify_issue3.py --require-complete` | 运行后查看 `issue3/run/acceptance.json`：`implemented_scope_passed` 表示登记范围通过；`full_issue_complete` 另核对原始验收映射、23 个模型块、内部待办及快照，仅表示本地粗粒度算子交付 |
| [Issue #4 整模 Decode](issue4/README.md) | `python scripts/verify_issue4.py` | `results/issue4/acceptance/acceptance.json` 核对模型单测、八组整模用例、四组放置对齐与源码快照，全部通过时 `status=passed` |

批量入口 `scripts/run_all.ps1 -Scope issue2|issue3|all` 选择 `all`（默认）时运行 Issue #2 和 #3 的日常仿真与专项验收，再补跑 Issue #4 模型单测，核对全仓库单测覆盖。八组整模用例通过 `python scripts/verify_issue4.py` 单独运行。批量总结写入 `results/run_all_<范围>.json`，原始日志位于 `results/run_all/<范围>/`。批量入口退出码 0 表示所选范围通过，1 表示失败，2 表示 Issue #2 本地审阅证据需对齐。

Issue #3 的具体范围、来源和剩余缺口见[验证说明](issue3/README.md)及[覆盖清单](issue3/scope.json)。当前清单的 23 个模型块均为 implemented，Issue #3 内部待办为空；实际完成结论仍须由对应版本的验收生成，不代表 GitHub 已关闭、完整模型拼装或整片执行完成。Issue #2、#3 的验收脚本都保存文本规范化的文件快照；判断结果时核对其范围和 `snapshot_unchanged_during_run`，历史通过状态不能直接沿用。

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
| [issue4/README.md](issue4/README.md) | 整模实现、四个里程碑、默认放置、验收口径及源码版本字段 |
| `../results/issue4/acceptance/` | Issue #4 专项脚本生成的计算、通信和验收报告，可从对应工程版本重新生成 |
| [archive/2026-11-18-cleanup/](archive/2026-11-18-cleanup/README.md) | 本次清理前的验证材料、迁移基线、旧源码备份和一次性脚本；原路径保存在压缩包内 |
| `../results/` | 可再生输出。顶层仿真报告可用 `scripts/clear_results.ps1` 清理 |
| `../tmp/` | 临时工作目录；需要保留的基线必须随验证材料一起归档，不能只记录一个被 Git 忽略的路径 |

以后需要保留历史验收时，将原始输出及依赖的基线一并归档，记录原路径、文件哈希、对应版本和当时的验收状态，再更新对应 Issue 的输出。同一 Issue 避免并列维护多个声称代表当前状态的 `run/` 目录。临时实验可使用 `--output-dir results/<实验名>`。Issue #2 和 #3 的验收脚本各自保护对方的证据树，Issue #4 同时保护这两棵证据树。

硬件规格归档另见 [configs/archive/](../configs/archive/README.md)。原生参数来源使用[硬件台账](../calibration/parameter_source_audit.json)，Issue #3 新增执行策略使用[执行参数台账](../calibration/operator_execution_audit.json)；这些都不属于生成文件的清理范围。

Issue #3 普通回归不带 `--require-complete`；正式交付必须带该选项并满足 `full_issue_complete=true`，否则退出 1。批量入口使用 `--require-complete` / `-RequireComplete`，仅约束 Issue #3，汇总中记录验收模式。生成目录 `validation/issue3/run/` 保持 Git 忽略。
