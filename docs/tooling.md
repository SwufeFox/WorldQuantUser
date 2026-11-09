# 工具地图

这里不解释怎么用，只回答一个问题：**我想问的那句话，对应哪个命令。**

20 个入口。这个数字是砍出来的：2026-11-01 之前是 26 个，删掉了 8 个——宏子系统（`expression_macros` / `ast_macros` / `macro_parity`，被 `alpha_unit` 的模板+槽位取代）、`sugar_rewrite`（量过一次，结论已在文档里）、`family_tree`（被 `family_graph` 取代）、以及三个跑完的台账迁移（`repair_ledger` / `backfill_registry` / `backfill_reason_codes`，东西已经改了，工具留着只会让人以为还要跑）。2026-11-04 加了 `ask`：一个只读查询入口，替掉每次都要重写一遍的内联脚本。2026-11-09 又砍了 9 个：`book_audit` / `reconcile_status` / `pool_stats` / `book_inventory` / `turnover_repair` / `wire_guard` / `research_template` / `family_graph` / `family_graph_png`——前六个被 `ledger sync`、`brief --drift`、`health --pool`、`ledger mechanisms`、`probe` 一一取代，`family_graph` 的历史结论留在 `docs/ast-pipeline.md`，`research_template` 里唯一还有用的整词替换搬进了 `research_app/substitution.py`。记忆索引另由 `memory_index` 从条目生成，用 `--check` 检测漂移。

删除标准只有一条：**被取代，或者跑完就没下次。** 不是"文件不够干净"。

## 先从统一 CLI 进入

```bash
python3 research_cli.py                       # 简短入口指南
python3 research_cli.py help catalog fields   # 具体任务的参数和例子
python3 research_cli.py schema alpha get      # 单个命令的机器契约
```

常用路径是 `catalog` 找数据、`lint` 检查表达式、`simulate` / `probe`
先 `--dry-run` 再运行、`compare` 看结果。平台 alpha 读写在 `alpha`；本地台账
统一在 `ledger`：`list`、`record`、`label`、`coverage`、`impact`、`lineage`。
`ledger list` 同时支持标签、字段、机制和指标筛选。

所有 CLI 命令共用 `--save PATH`、`--compact`、`--quiet`、`--research-dir PATH`，
参数放在哪一级都有效。`--save` 原子保存与 stdout 完全相同的 JSON envelope，
也保存结构化错误；写入路径在执行命令前检查。下表的辅助工具继续使用自己的
`--json` 报告接口。

`lint` / `simulate` 共用可重复的 `--file`、`--expr`；后文件覆盖前文件设置，
CLI 参数最后覆盖，每条表达式的 decay 单独保留。需要文件的命令用 `-`
读取 stdin，每次调用只能读一次。`probe` 同样接受模拟设置和会话参数。

实现位置也按用途固定：参数、帮助和 schema 在 `research_app/parser.py`；
执行边界在 `app.py`，公共 CLI 约定在 `runtime.py`。目录查询、表达式检查、
实验、平台 alpha、本地台账分别在 `catalog.py`、`validation.py`、
`experiments.py`、`alphas.py`、`ledger.py`。账户与缓存、研究摘要与查看记录、
文献检索分别在 `account.py`、`campaign.py`、`literature.py`。
这些是内部模块，入口仍是 `research_cli.py`；改命令时直接修改、测试所属模块。

字段枚举统一走 `field_index.fetch_all_fields`。`validation.py` 的在线 lint 和
模拟预检从一次查询同时生成字段名、类型及市场/披露分类，避免两条路径给出
不同诊断；`tools.guarded_sim` 复用同一枚举。基础安装只需 `requests`，
`pandas` 在调用 `fields_to_df()` 时按需加载，DataFrame 用户再单独安装。

## 我要回答……

### 台账里有什么、对不对

| 问题 | 命令 |
| --- | --- |
| 台账的 status 和平台对不对得上 | `python3 research_cli.py ledger sync --apply` |
| 按轴分组看一眼：`n` / 中位数 / 地板税占比 | `python3 research_cli.py ask --by neutralization --stat n,median:sharpe,floor_share` |
| 某个旋钮在面板内的斜率（不是语料趋势） | `python3 research_cli.py ask --by decay --panel turnover,fitness` |
| 同一构造换设置之后还相关多少 | `python3 research_cli.py ask --matrix corr --matrix-within expr` |
| 哪些行「零 FAIL」但检查其实没算 | `python3 research_cli.py ask --where check=SELF_CORRELATION:PENDING --list` |
| 有没有两个 alpha 的 PnL 其实是一个 | `python3 tools/dedupe_ledger.py` |
| 哪些行缺 `family` 标签 | `python3 tools/derive_family.py` |
| 同一批里有没有「不同表达式同一个结果」 | `python3 tools/degenerate_batch.py` |
| 按 PnL 相关聚类（不是按结构） | `python3 tools/cluster_ledger.py` |
| 记忆索引与条目是否一致 | `python3 tools/memory_index.py --workspace <workspace> --check`；生成链接统一使用 `/`，兼容 Windows 和 POSIX |

### 构造：族、零件、单元

| 问题 | 命令 |
| --- | --- |
| 这些 alpha 是不是同一个构造（按结构，不按 PnL） | `python3 research_cli.py ledger mechanisms --status ACTIVE` |
| 书在哪些机制上堆得太多、哪些几乎没用 | 同上，看 `median_marginal` 与 `members` 两列 |
| 有哪些可复用的零件，哪些只服务过一个族 | `python3 tools/parts_catalogue.py --seeds` |
| 我们到底有几个构造、每个还差什么 | `python3 tools/unit_report.py`（`--gaps` / `--cross` / `--derive`） |
| 这个数据族以前试过什么形状、最好多少分 | `python3 tools/unit_report.py --dataset model51`（数据集 id 自动映射到族） |
| 台账的 verdict 和真实的书是否一致 | `python3 research_cli.py brief --drift` |

### 写表达式、跑模拟

| 问题 | 命令 |
| --- | --- |
| 我要跑一批臂（常态） | 写一个 spec.json，然后 `python3 research_cli.py probe spec.json`——lint、模拟、train-only 表格、JSON、台账一次做完 |
| 不登录、不花额度，先查看整批请求 | `python3 research_cli.py probe spec.json --dry-run`；逐组验证配置并输出完整 payload，实时目录检查在正式模拟前执行 |
| 我要开一个单次实验 | 直接写一个两三条臂的 spec，`python3 research_cli.py probe spec.json`（没有模板文件了） |
| 换字段 / 换窗口 / 换权重的批量实验 | `research_app/substitution.py` 的 `expand()` 生成 `(label, expression)` 对，再落成 spec |
| 我想跑一批，但怕名字写错 | `tools/guarded_sim.py`（库，不是 CLI） |
| 这个池子够不够厚 | `python3 research_cli.py health --pool` |
| 这个字段到底什么量纲 | `python3 tools/probe_units.py` |

日常实验只维护 spec 文件：先离线预览，再正式运行。所有配置组在登录和模拟前完成本地校验；等效设置合并，同一批复用缓存会话。`simulate` 的研究文件和 `probe` 的 spec 文件都按 UTF-8 读取，兼容 BOM。

### 备份与重放

| 问题 | 命令 |
| --- | --- |
| 存一份、验一份 | `python3 tools/backup_replay.py backup` / `verify` |
| 2026-10-25 那天的台账是什么样 | `python3 tools/backup_replay.py asof --at ...` |
| 平台数字漂了没有 | `python3 tools/backup_replay.py replay --run` |

## 每个工具的输出约定

`research_cli.py` 有一条契约：stdout 恰好一个 JSON 文档。工具现在跟它对齐，用同一个开关：

```bash
python3 tools/unit_report.py --json            # 文档去 stdout，人读的报告去 stderr
python3 tools/unit_report.py --json out.json   # 同一个文档写进文件
python3 tools/unit_report.py --json -           # 显式写 stdout
python3 tools/unit_report.py                    # 不传就是人读的报告
```

好处只有一个但够了：`tool --json | jq` 永远成立。以前 12 个工具用三种名字（`--json` / `--out` / `--save`），而且写完文件还要自己去打开——多一步就多一次出错的机会。

两条实现上的坑，都踩过：

* **`--json` 一旦出现，stdout 就只属于文档。** 先打印报告再写文档会让 `json.loads(stdout)` 失败，所以报告整体改道 stderr。这不是"消失"，是搬家。
* **`stream=sys.stdout` 不能当默认参数。** 默认值在函数定义时求值，重定向 `sys.stdout` 对它无效——先是报告照旧出现在 stdout，文档跟在后面，看起来像"契约没生效"。所有 `report(..., stream=None)` 现在在函数体里解析。

写文件失败也不再是 traceback：路径不可写 → 一行 stderr + exit 2（用法错误，因为路径是调用方给的）。共用的实现在 `tools/_output.py`，它没有 `main()`，不是工具。

## 看一条 alpha 在算什么

```bash
python3 research_cli.py alpha explain akVznkO2
```

三层，故意分开：

| 层 | 内容 |
| --- | --- |
| **构造** | 每条腿一句话（"ranked (close minus low) over (high minus low) minus 0.5"），带字段、数据族、权重 |
| **证据** | 指标、Fitness 的算术复算（分母用的是真实换手还是 0.125 地板）、跑过的消融、本地对书相关 |
| **风险** | 族是否已关闭，以及一句提醒：**权重是乘数，不是贡献** |

腿是从表达式树读出来的，所以不会和公式漂移。`0.25*(a+b)` 这种会被摊开成两条腿——否则一个六腿复合体会显示成"一条腿"。

## 提交是唯一会写平台的动作

```bash
python3 research_cli.py alpha submit ID            # 预检：跑 live checks，什么都不发
python3 research_cli.py alpha submit ID --yes --record   # 真提交 + 回读 + 记账
```

三条硬规矩：

* **没有 `--yes` 就不发**——平台没有"试提交"，预检就是 check 流程；
* **`name` / `category` 必须先设**，否则拒绝（`alpha describe ID --name ... --category ...`）。没有标签的 alpha 事后无法归属；
* **`--record` 用 `regular_expression()` 和 `is` 段**取数，不直接摸平台原始 payload——`regular` 有时是字符串、有时是对象，摸错一次就记了一行没有 metrics 的账（2026-11-01 真实发生过）。
* **`--record` 顺带把这条 alpha 的日频 PnL 缓存到 `<research>/pnl/`**。本地相关扫描读的就是那个目录，不缓存的话新提交在下次扫描里是隐形的——而扫描正是看清"这次提交挤掉了什么"的便宜办法（2026-11-01：提交 `leMnKWol` 一条，同族 10 条从 0.35 跳到 0.95）。

台账那边现在也会拦：`record_entry` 拒绝用没有 metrics 的行覆盖有 metrics 的行，除非显式 `allow_metrics_loss=True`。

## 工具会读的封闭清单

`<research>/closed-families.json` 是机制地图第二节的机器可读版：20 条已关闭的方向，带 id、类型（`blocked-by-check` / `falsified` / `structurally-impossible`）、理由、证据。

只有带 `match` 的条目能机械匹配（今天是 `family: option`），其余是技术方向（"把腿放慢"、"跨族移植"），表达式里看不出来，留给人和 agent 读。

读它的地方：

* `parts_catalogue --seeds` —— 来自封闭族的种子会标 `[CLOSED: 理由]` 并排到最后，**不删**（藏起来正是它上次被推荐的原因）。
* `unit_report` —— kind 表里标封闭族，另外单独列一节"封闭族里分数很高的"，因为那种行最容易看成机会（`fundamental,option` 那个 kind best F **3.89**）。

文件缺失或损坏时两个工具都照常运行，只是什么都不标——研究文件不见了不该让工具停摆。

## 加工具之前，先回答三个问题

写在 `AGENTS.md` 的规则里，这里重复一遍，因为这是唯一能阻止数量涨回去的地方。

1. **最接近的现成工具是哪个？** 说不出名字，说明没找。
2. **为什么它不能被扩一个参数/子命令？** 「单独一个文件更干净」不算理由。
3. **它回答的问题，今天没有任何工具能回答吗？** 如果答案是「能，但要用两个工具」，那是组合的问题，不是缺工具。

三个都过了再写；过不了就把功能塞进最接近的那个。

**更常发生的其实是第四种情况：这个问题只问一次。** 那就别做工具——写个一次性脚本放在研究目录，把结论写进报告和 `mechanism-atlas`。工具是给会跑第二次的东西用的。

## 实验设计：一个批次要么有对照，要么声明自己在扫描

`probe` 的 spec 里每个臂都必须带 `design`，取值只有五个：

| design | 是什么 | 约束 |
| --- | --- | --- |
| `control` | 空白对照：父表达式原样、设置不变 | 每批**恰好一个** |
| `add` | 加法：加一条腿或一个字段 | settings 必须与对照完全相同 |
| `drop` | 减法：拿掉一条腿 | 同上 |
| `parallel` | 相互对照：只动一个旋钮做梯度 | 与对照的 settings 差**恰好一个键** |
| `rescue` | 互补：换一个*不同实现*把效果补回来 | 必须写 `proxy_of` |

不带对照的批次必须把 `"design": "explore"` 写在 spec 顶层。这不是逃生门，是**记录**：扫描和实验在台账里长得一样，唯一区别是扫描自己承认了。

理由不是洁癖。2026-11-03 那批四个臂，参照是一个本来就不过门的 base（S 1.23 / F 0.68）；两个臂"更差"，但差在哪里无法归因——**没有合格的对照，减法实验就没有解释力**。

## 边界

* 仓库（`/data/src/magix/WorldQuantUser`）放**工具**：任何能把函数跑两次的东西。
* 研究目录（`/data/worldquant-research`）放**结果**：台账、缓存、一次性脚本、报告。一次性脚本不进 git。
