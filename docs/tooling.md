# 工具地图

这里不解释怎么用，只回答一个问题：**我想问的那句话，对应哪个命令。**

26 个入口如果只靠记忆维护，迟早会变成 40 个。所以每个工具在这里有一个状态：

| 状态 | 意思 |
| --- | --- |
| **active** | 现在就该用它 |
| **maintenance** | 平时不动，台账/账号和平台不一致时才跑 |
| **migration-done** | 一次性的历史修复，已经执行完；留着是为了可审计，不要再跑 |
| **superseded** | 被更好的实现取代，不要用 |

## 我要回答……

### 台账里有什么、对不对

| 问题 | 命令 | 状态 |
| --- | --- | --- |
| 台账的 status 和平台对不对得上 | `python3 tools/reconcile_status.py` | maintenance |
| 有没有两个 alpha 的 PnL 其实是一个 | `python3 tools/dedupe_ledger.py` | maintenance |
| 哪些行缺 `family` 标签 | `python3 tools/derive_family.py` | maintenance |
| 同一批里有没有「不同表达式同一个结果」 | `python3 tools/degenerate_batch.py` | maintenance |
| 按 PnL 相关聚类（不是按结构） | `python3 tools/cluster_ledger.py` | active |

### 构造：族、零件、单元

| 问题 | 命令 | 状态 |
| --- | --- | --- |
| 这些 alpha 是不是同一个构造（按结构，不按 PnL） | `python3 tools/family_graph.py --all` | active |
| 画出来看 | `python3 tools/family_graph_png.py` | active |
| 有哪些可复用的零件，哪些零件只服务过一个族 | `python3 tools/parts_catalogue.py --seeds` | active |
| 我们到底有几个构造、每个构造还差什么 | `python3 tools/unit_report.py`（`--gaps` / `--cross` / `--derive`） | active |
| 书里现在拿的是什么，按构造分组 | `python3 tools/book_inventory.py --mechanisms` | active |
| 台账的 verdict 和真实的书是否一致 | `python3 tools/book_audit.py` | active |
| 旧的族树 | `python3 tools/family_tree.py` | **superseded**（用 family_graph） |

### 写表达式

| 问题 | 命令 | 状态 |
| --- | --- | --- |
| 糖能不能把已有表达式收束 | `python3 tools/sugar_rewrite.py` | active |
| 文本后端和树后端是否等价 | `python3 tools/macro_parity.py` | active |
| 我要开一个新实验 | 复制 `tools/research_template.py` | active |
| 把脚本里的 `ml.simulate_alphas` 改成走护栏 | `python3 tools/wire_guard.py` | active |

### 模拟

| 问题 | 命令 | 状态 |
| --- | --- | --- |
| 我想跑一批，但怕名字写错 | `tools/guarded_sim.py`（库，不是 CLI） | active |
| 换手把 Fitness 压住了怎么办 | `python3 tools/turnover_repair.py` | active |
| 这个池子够不够厚 | `python3 tools/pool_stats.py` | active |

### 字段

| 问题 | 命令 | 状态 |
| --- | --- | --- |
| 这个字段到底什么量纲 | `python3 tools/probe_units.py` | active |
| 目录里有哪些字段 | `python research_cli.py catalog fields` | active |

### 备份与重放

| 问题 | 命令 | 状态 |
| --- | --- | --- |
| 存一份、验一份 | `python3 tools/backup_replay.py backup` / `verify` | active |
| 2026-10-25 那天的台账是什么样 | `python3 tools/backup_replay.py asof --at ...` | active |
| 平台数字漂了没有 | `python3 tools/backup_replay.py replay --run` | active |

### 已经执行完的一次性迁移

这些不是工具，是历史。留着是为了让台账里那些改动可追溯；**不要再跑**。

| 做过什么 | 文件 | 状态 |
| --- | --- | --- |
| 修好 9 条尾部粘行号的表达式、把 191 条空壳标 abandoned | `tools/repair_ledger.py` | migration-done |
| 从平台和旧结果文件第一次灌满台账 | `tools/backfill_registry.py` | migration-done |
| 把拒绝原因码回填到 1515 行历史 | `tools/backfill_reason_codes.py` | migration-done |

## 加工具之前，先回答三个问题

写在 `AGENTS.md` 的规则里，这里重复一遍，因为这是唯一能阻止数量继续涨的地方。

1. **最接近的现成工具是哪个？** 说不出名字，说明你没找。
2. **为什么它不能被扩一个参数/子命令？** 「单独一个文件更干净」不算理由——26 个文件每个都很干净，合起来是这座山。
3. **它回答的问题，今天没有任何工具能回答吗？** 如果答案是「能，但是要用两个工具」，那是组合的问题，不是缺工具。

三个问题都过了，再写。**过不了就把功能塞进最接近的那个。**

实测（2026-11-01）：26 个工具、24.5k 行、20 个根模块。死代码几乎没有（只有一个 0 引用的 `family_tree`），所以成本不在逻辑重复，在**入口数量**——26 套 argparse、26 套报告格式、26 个名字要记。

## 边界

* 仓库（`/data/src/magix/WorldQuantUser`）放**工具**，任何能把函数跑两次的东西。
* 研究目录（`/data/worldquant-research`）放**结果**：台账、缓存、一次性脚本、报告。一次性脚本不进 git，也不该被当成工具维护。