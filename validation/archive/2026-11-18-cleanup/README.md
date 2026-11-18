# 2026-11-18 过程材料归档

[evidence.zip](evidence.zip) 原样保存清理前的 596 个文件；[manifest.json](manifest.json) 记录每个文件的原路径、字节数和 SHA-256，以及压缩包的整体哈希。压缩前共 33,691,184 字节，压缩后为 5,977,414 字节。创建后已逐文件核对原文件与包内内容。

[cleanup.json](cleanup.json) 记录本次整理动作。自动审批拒绝永久删除历史原件，因此这些文件已移入本目录下的 `originals/`，作为被 Git 忽略的本地恢复副本；18 个旧缓存目录也移入其中。该目录不是当前源码或验收入口，不计入压缩包清单中的新增材料。可移交、可纳入版本控制的历史证据是 `evidence.zip` 和 `manifest.json`。

| 包内原路径 | 内容 |
|---|---|
| `validation/issue2/` | 清理前的完整验收输出和审阅记录副本；其中的通过结论仅针对当时快照 |
| `validation/timing_breakdowns/` | 服务周期与总时间分解的迁移对比及验收输出 |
| `validation/config_documentation/` | 四份 JSON 注释整理的对比及验收输出 |
| `validation/hardware_archival/` | 11 项硬件参数归档的对比及验收输出 |
| `validation/tsp_mem_concurrency/`、`validation/tsp_weight_prefetch/` | MEM 并发和权重预取的历史验证 |
| `tmp/` | 上述迁移使用的基线、前后报告、源码备份、一次性迁移脚本及过程日志 |

`tmp/config_annotation_baseline.json` 和 `tmp/stages_cycles_refactor_baseline.json` 等被历史验证记录引用的文件均包含在包内。历史 JSON 中的路径、版本、哈希和状态保持原样；路径按包内的原工程根目录解释，不是对当前工作区的引用。

需要查看原始材料时，解压到独立目录，例如在工程根目录运行：

```powershell
Expand-Archive -LiteralPath validation/archive/2026-11-18-cleanup/evidence.zip -DestinationPath results/history-2026-11-18
```

不要覆盖当前工程目录。包内脚本和源码对应历史输入，保留用于追溯，不作为当前维护入口。当前验收、重新生成方式与保留规则见 [验证记录说明](../../README.md)。原 `validation/issue2/review.json` 仍保留在活动目录，自动清理不会续签其审阅状态。
