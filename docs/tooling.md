# 工具地图

这里不解释怎么用，只回答一个问题：**我想问的那句话，对应哪个命令。**

18 个入口。这个数字是砍出来的：2026-11-01 之前是 26 个，删掉了 8 个——宏子系统（`expression_macros` / `ast_macros` / `macro_parity`，被 `alpha_unit` 的模板+槽位取代）、`sugar_rewrite`（量过一次，结论已在文档里）、`family_tree`（被 `family_graph` 取代）、以及三个跑完的台账迁移（`repair_ledger` / `backfill_registry` / `backfill_reason_codes`，东西已经改了，工具留着只会让人以为还要跑）。

删除标准只有一条：**被取代，或者跑完就没下次。** 不是"文件不够干净"。

## 我要回答……

### 台账里有什么、对不对

| 问题 | 命令 |
| --- | --- |
| 台账的 status 和平台对不对得上 | `python3 tools/reconcile_status.py` |
| 有没有两个 alpha 的 PnL 其实是一个 | `python3 tools/dedupe_ledger.py` |
| 哪些行缺 `family` 标签 | `python3 tools/derive_family.py` |
| 同一批里有没有「不同表达式同一个结果」 | `python3 tools/degenerate_batch.py` |
| 按 PnL 相关聚类（不是按结构） | `python3 tools/cluster_ledger.py` |

### 构造：族、零件、单元

| 问题 | 命令 |
| --- | --- |
| 这些 alpha 是不是同一个构造（按结构，不按 PnL） | `python3 tools/family_graph.py --all` |
| 画出来看 | `python3 tools/family_graph_png.py` |
| 有哪些可复用的零件，哪些只服务过一个族 | `python3 tools/parts_catalogue.py --seeds` |
| 我们到底有几个构造、每个还差什么 | `python3 tools/unit_report.py`（`--gaps` / `--cross` / `--derive`） |
| 书里现在拿的是什么，按构造分组 | `python3 tools/book_inventory.py --mechanisms` |
| 台账的 verdict 和真实的书是否一致 | `python3 tools/book_audit.py` |

### 写表达式、跑模拟

| 问题 | 命令 |
| --- | --- |
| 我要开一个新实验 | 复制 `tools/research_template.py` |
| 我想跑一批，但怕名字写错 | `tools/guarded_sim.py`（库，不是 CLI） |
| 把脚本里的 `ml.simulate_alphas` 改成走护栏 | `python3 tools/wire_guard.py` |
| 换手把 Fitness 压住了怎么办 | `python3 tools/turnover_repair.py` |
| 这个池子够不够厚 | `python3 tools/pool_stats.py` |
| 这个字段到底什么量纲 | `python3 tools/probe_units.py` |

### 备份与重放

| 问题 | 命令 |
| --- | --- |
| 存一份、验一份 | `python3 tools/backup_replay.py backup` / `verify` |
| 2026-10-25 那天的台账是什么样 | `python3 tools/backup_replay.py asof --at ...` |
| 平台数字漂了没有 | `python3 tools/backup_replay.py replay --run` |

## 加工具之前，先回答三个问题

写在 `AGENTS.md` 的规则里，这里重复一遍，因为这是唯一能阻止数量涨回去的地方。

1. **最接近的现成工具是哪个？** 说不出名字，说明没找。
2. **为什么它不能被扩一个参数/子命令？** 「单独一个文件更干净」不算理由。
3. **它回答的问题，今天没有任何工具能回答吗？** 如果答案是「能，但要用两个工具」，那是组合的问题，不是缺工具。

三个都过了再写；过不了就把功能塞进最接近的那个。

**更常发生的其实是第四种情况：这个问题只问一次。** 那就别做工具——写个一次性脚本放在研究目录，把结论写进报告和 `mechanism-atlas`。工具是给会跑第二次的东西用的。

## 边界

* 仓库（`/data/src/magix/WorldQuantUser`）放**工具**：任何能把函数跑两次的东西。
* 研究目录（`/data/worldquant-research`）放**结果**：台账、缓存、一次性脚本、报告。一次性脚本不进 git。