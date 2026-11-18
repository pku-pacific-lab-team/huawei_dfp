# 验证记录与历史归档

当前完整验收只使用 [issue2/run/acceptance.json](issue2/run/acceptance.json)。在工程根目录运行 `python scripts/verify_issue2.py` 可重新生成该目录中的测试、命令和报告证据。判断结果时同时检查 `snapshot` 是否对应当前文件、`local_acceptance_passed` 及 A1—A5 状态；历史记录中的通过状态不能直接沿用。

[issue2/review.json](issue2/review.json) 保存来源和迁移说明的语义审阅依据。自动测试不会更新该审阅快照，代码、配置或文档变化后，A1/A5 可能显示 `needs_review`。刷新生成结果不等于完成语义审阅。

| 位置 | 用途与保留规则 |
|---|---|
| `issue2/run/` | 唯一的当前完整验收输出；由验收脚本生成和覆盖 |
| `issue2/review.json` | 审阅依据，持续保留；完成实际审阅后才能更新 |
| [archive/2026-11-18-cleanup/](archive/2026-11-18-cleanup/README.md) | 本次清理前的验证材料、迁移基线、旧源码备份和一次性脚本；原路径保存在压缩包内 |
| `../results/` | 可再生仿真报告，不是验收依据；使用 `scripts/clear_results.ps1` 清理 |
| `../tmp/` | 临时工作目录；需要保留的基线必须随验证材料一起归档，不能只记录一个被 Git 忽略的路径 |

以后需要保留历史验收时，将原始输出及依赖的基线一并归档，记录原路径、文件哈希、对应版本和当时的验收状态，再更新当前输出。避免并列维护多个声称代表当前状态的 `run/` 目录。临时实验可使用 `--output-dir results/<实验名>`，完成后只保留必要结论或归档证据。

硬件规格归档另见 [configs/archive/](../configs/archive/README.md)，参数来源台账另见 [calibration/parameter_source_audit.json](../calibration/parameter_source_audit.json)，两者不属于生成文件的清理范围。
