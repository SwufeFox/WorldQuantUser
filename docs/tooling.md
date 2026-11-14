# 工具地图

这里不解释怎么用，只回答一个问题：**我想问的那句话，对应哪个命令。**

20 个入口。这个数字是砍出来的：2026-11-01 之前是 26 个，删掉了 8 个——宏子系统（`expression_macros` / `ast_macros` / `macro_parity`，被 `alpha_unit` 的模板+槽位取代）、`sugar_rewrite`（量过一次，结论已在文档里）、`family_tree`（被 `family_graph` 取代）、以及三个跑完的台账迁移（`repair_ledger` / `backfill_registry` / `backfill_reason_codes`，东西已经改了，工具留着只会让人以为还要跑）。2026-11-04 加了 `ask`：一个只读查询入口，替掉每次都要重写一遍的内联脚本。2026-11-09 又砍了 9 个：`book_audit` / `reconcile_status` / `pool_stats` / `book_inventory` / `turnover_repair` / `wire_guard` / `research_template` / `family_graph` / `family_graph_png`——前六个被 `ledger sync`、`brief --drift`、`health --pool`、`ledger mechanisms`、`probe` 一一取代，`family_graph` 的历史结论留在 `docs/ast-pipeline.md`，`research_template` 里唯一还有用的整词替换搬进了 `research_app/substitution.py`。记忆索引另由 `memory_index` 从条目生成，用 `--check` 检测漂移。

删除标准只有一条：**被取代，或者跑完就没下次。** 不是"文件不够干净"。

## 先从统一 CLI 进入

```bash
python3 research_cli.py brief                 # 当前状态、证据摘要和来源路径
python3 research_cli.py help                  # 不熟悉命令时才读导航
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


## 给 AI 的信息按阶段进入

常驻入口是 `AGENTS.md`；它只保留启动方法、研究原则和执行约束。
`brief` 给出当前状态；本页负责按问题导航；台账和报告保存完整证据。
全量 `schema` 留给接口集成，不作为每次启动的必读项。

| 阶段 | 现在需要的信息 | 留下什么 |
| --- | --- | --- |
| 恢复 | 当前问题、`brief` 的来源路径和记录是否存在、相关未决事项 | 这一轮要消除的一个不确定性 |
| 发散 | 与问题有关的成功、受挫和未知；必要时查机制文献 | 少量不同经济解释及各自可观察的预测 |
| 收敛 | 各解释的支持证据、反例、数据可得性和最便宜的区分办法 | 要检验的假设、对照、停止或改向条件 |
| 执行 | 所选命令的 `help` / `schema`、字段与设置、dry-run 请求 | spec、结果和台账记录 |
| 复盘 | 实际结果与原预测的差异；未解决的替代解释 | 更新对应问题及证据链接，保留下一步 |

这张表是信息路由，不是每轮必须填写的表单。定位明确的修复直接进入对应阶段。
研究范围开放时，可以先保留两到四个机制不同的解释，再查实现所需字段。
同一表达式换窗口或权重属于局部变体，不能单凭数量算作发散。
比较方向时同时考虑能学到什么、可行性和已有证据，不仅看历史最高分。
没有记录的方向只能叫“本地证据未知”；“已测过的形状”也不等于经济假设被否定。

### 摘要怎样保持小而可追溯

`brief` 的 `research_dir` 是当前研究目录，`ledger.path` 和 `threads.path`
指向来源，`exists` 区分文件缺失与文件存在但没有记录。换机器后先核对这些字段，
避免把一个新目录当成全部研究历史。

`recent.best` 保留分数线索，带 `family` / `mechanism`；指标限训练期，缺乏 train 的
记录列入 `missing_train_metrics`，不用全窗口数字补齐。`book.sharpe_median` 同样按训练期。
`recent.setbacks` 补充近期 `rejected`、`falsified`、`no_go` 的原始判断与 note，也包含模拟之后的
人工重标。按记录时间倒序，对相同 family、mechanism、verdict 只展示最新代表；
缺少机制标签时按记录 key 区分。这个去重只是减少重复阅读，不证明机制独立。
`setbacks.count` 是窗口内相关记录数，展示项只是样本，不能用它代替失败率分析。

`--top` 至少为 1，分别限制高分项、受挫项、报告、open 问题和 closed 问题的数量。
`book.self_duplicates.window` 为 `train`；测试期共同波动不会改变启动诊断。
`threads.counts` 给出完整问题数，问题按文件顺序展示，优先级通过移动原文件的行维护。
问题和证据 note 等长文本只预览至 240 字符，以 `…` 标记裁剪；需要使用结论时，
从路径和记录 key 回读完整条件。摘要只呈现已记录事实和判断，不生成机会或自动关闭方向。

`book.evidence` 给出本地 ACTIVE/SUBMITTED 成员摘要、平台快照时间与作用域，以及 PnL
完整/未知/缺失计数。成员数相同不代表成员相同；digest 用于识别换人。平台快照只证明其
`as_of` 时刻的所选范围，本地台账和旧 PnL 文件不能被提升为实时完整的组合证据。

### 发散需要的边界

| 记录 | 能支持的结论 | 下一步可能是什么 |
| --- | --- | --- |
| `rejected` | 这个表达式在这些设置下表现不合格 | 检查实现或替代代理，机制仍可能成立 |
| `falsified` | 有对照的检验否定了具体假设 | 保留决定性对照；仅在适用条件或新证据变化时重开 |
| `no_go` | 数据或执行条件阻止了公平检验 | 寻找可用代理，或等阻碍条件变化 |
| 缺记录 / 缺标签 | 当前本地信息不足 | 查工作目录与台账，不推断“没人试过” |

复盘时分别写清观察、解释和待检验假设。文献是候选机制的来源，不能替代本地检验。
`triage` 默认只用训练期指标和收益；先 `--freeze PATH` 保存选择，再用
`--show-test --selection PATH` 查看测试期。验证保持冻结的候选集合与顺序，日志绑定选择 ID。
`--internal` 依赖测试期校准，必须搭配该验证流程；`--train-end` 不能越过记录中的测试起点。
训练证据缺失时保持 unknown，不用全窗口数字补齐。

### 用现有问题文件交接

`<research>/open-threads.md` 是跨会话的当前问题清单，不另建一份状态摘要。
保留现有 `- [open]` / `- [closed]` 格式；问题标题加粗也便于 `feed --from-threads`
抽取查询。下面只是写法示例，不是研究结论：

```markdown
- [open] **修订效应还是价格动量？** — 证据：批次文件 / alpha key；下一步：保持设置，加入区分两者的对照；query: analyst revisions price momentum
- [closed] **某代理能否检验该机制？** — 结论及适用范围；证据：报告路径；重开条件：覆盖率或代理变化。
```

只更新本轮受证据影响的问题。完整推导、负例和参数放在所链接的记录里；
交接保留决策理由、未解决分歧和下一步，不复制工具输出或整段对话。
`feed --action` 在原始结果上补本地书暴露和既往试验证据；缺失台账或未知领域保持 unknown，
零命中不等于未被探索。检索失败、空结果和已读重复分列在 `searches`，论文保留 DOI/URL。
记忆索引仍由 `memory_index` 生成，按当前问题取相关条目；路径与账户限制从当前 CLI
和平台读取，索引不写死旧机器路径、日期报告或费用假设。

### 有界并发、缓存与主动下一步

```bash
python3 research_cli.py feed --from-threads --workers 4 --timeout 30 --dry-run
python3 research_cli.py feed --topic "analyst forecast dispersion" --topic "earnings quality" --action
```

`--workers` 为 1–8，默认 4；它限制同时请求的主题数，`--timeout` 是单次网络请求的
超时秒数。检索完成后仍按原主题顺序均衡选取，完成先后不会改变排序。网络调用在锁外；
多个 feed 提交结果前会锁住状态并重新读取已读集合，避免按旧快照重复选中论文。

OpenAlex 使用 `<research>/cache/papers/`；网页搜索使用 `<research>/search-cache/`。
缓存键包含实际请求选项，全文开关、原问题和结果数量不同会分别保存。相同请求并发未命中时
只由持锁者抓取；有效缓存直接复用，过期或损坏才重新读取。缓存写入失败仍保留成功获取的结果。
论文输出包括可用的出版页、DOI 与 PDF 链接；来源未提供的全文保持未知。

`next_actions` 是有上限的建议：失败查询附 CLI 参数数组 `argv`，未搜主题给下一批参数，
无法推导查询的问题给原文件位置，已获得的论文给来源阅读和反例对照任务。
`argv` 是传给 `research_cli.py` 的参数列表，不是可以直接拼接进 shell 的字符串。
中文问题可以保留中文标题，并显式写 `; query: ...`；系统不自动编造翻译。

在用户授权的研究范围内，AI 应主动完成可获取的前置事实、比较不同解释、准备下一项实验并
先 dry-run。存在反证时先解释分歧，避免反复微调同一个高分表达式。工具的下一步建议提供证据
入口；模拟、验证和提交仍遵循各自边界。

seen 与草稿各自原子提交，草稿先于 seen：草稿失败不会把论文标成已读。它们不是跨文件事务，
两次提交之间中断可能留下重复草稿；seen 已提交但 stdout 未送达时，也可能已消费本次选中的论文。
`--dry-run` 不写 seen/草稿，检索缓存仍可写入。Windows 使用系统字节锁，POSIX 使用 flock；
进程退出后系统释放锁，旧 `.pid` 文件不会阻塞新任务。


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
| 我要跑一批臂（常态） | 写一个 sample-batch.spec.json，然后 `python3 research_cli.py probe sample-batch.spec.json`——lint、模拟、train-only 表格、对照差值、JSON、台账一次做完 |
| 不登录、不花额度，先查看整批请求 | `python3 research_cli.py probe sample-batch.spec.json --dry-run`；输出完整 payload、设计上下文和精确重复证据，正式模拟前再做实时目录检查 |
| 按当前账户核验这份 spec，但不模拟 | `python3 research_cli.py probe sample-batch.spec.json --check`；使用与正式运行相同的臂和生效设置 |
| 我要开一个单次实验 | 直接写一个两三条臂的 spec，`python3 research_cli.py probe sample-batch.spec.json`（没有模板文件了） |
| 这批臂相对对照变了多少 | 看 `comparisons.rows`：arm 减 control 的训练期差值；缺失项列在 `unavailable`，explore 批次不计算归因 |
| 换字段 / 换窗口 / 换权重的批量实验 | `research_app/substitution.py` 的 `expand()` 生成 `(label, expression)` 对，再落成 spec |
| 我想跑一批，但怕名字写错 | `tools/guarded_sim.py`（库，不是 CLI） |
| 这个池子够不够厚 | `python3 research_cli.py health --pool` |
| 这个字段到底什么量纲 | `python3 tools/probe_units.py` |

日常实验维护一份 spec：先 `--dry-run` 离线预览，再用 `--check` 在线核验，最后正式运行。
`idea` / `mechanism` / `note` 会随输出和批次文件保存；`--strict-gate` 只拒绝表达式与完整设置
都相同的已完成结果。所有组先通过本地校验，等效设置合并，整批复用会话。输入支持 UTF-8 BOM。

```bash
python3 research_cli.py probe sample-batch.spec.json --batch-path sample-batch.json --save sample-response.json
python3 research_cli.py probe --recover sample-batch.json --max-wait-seconds 60
python3 research_cli.py probe --resume sample-batch.json --dry-run
python3 research_cli.py probe --resume sample-batch.json --max-wait-seconds 60
python3 research_cli.py resume checkpoint.json --max-wait-seconds 60
```

示例文件放在研究目录；从该目录运行或传绝对路径。`--batch-path` 是持久批次，`--save` 是
输出回执，两者使用不同路径。回执给出 `path`、`run_id`、`started_at`、`request_fingerprint`、
原始 `arm_order`、逐臂状态、`execution`、`recording` 和 `next_actions`。
`--save` 与批次、checkpoint、恢复输入或冻结选择同路径时会在执行前拒绝；错误回执也不会
写入受保护文件。读取保存回执恢复时，同样保护其指向的实际批次。

每个 POST 之前保存提交意图，拿到任务身份后立即保存。没有身份的未知提交不会自动重发。
`--max-wait-seconds` 默认 300；前组未完成时后组保持排队。`--recover` 只读平台并幂等补账，
接受批次或保存的回执；`--resume` 才显式继续已知排队/已接受任务，支持离线预览。
执行失败、未完成和证据缺失列为 deferred，不自动判成研究失败；旧批次的记账策略不明时
不会推测为允许记账。同批次运行/恢复互斥，`--force` 不能覆盖正在运行的批次。

目录检查携带账户、region/universe/delay/instrumentType、完整性和时效证据。缺失、过期、
不完整目录不能冒充预检通过；显式覆盖仍保留错误。字段类型按 AST 检查，经济口径不明则提示
unknown。`feed` 的关键词只产生阅读线索；明确字段需求才与当前作用域目录核对可用性。


### 合并已有结果与读取实时检查

```bash
python3 research_cli.py compare sample-batch-a.json sample-batch-b.json
python3 research_cli.py compare --match "**/sample-batch*.json" --latest --sort none
python3 research_cli.py compare --match "sample-batch-a.json" --match "**/sample-batch-b*.json" --latest 2
python3 research_cli.py compare sample-batch.json --freeze choice.json --cost-bps 5 10 20 --summary
python3 research_cli.py compare --frozen --selection choice.json --summary
python3 research_cli.py alpha check ALPHA_A ALPHA_B --workers 2 --max-wait 300
```

`compare` 可以省略 FILE，改用可重复的 `--match NAME_OR_GLOB`。模式相对研究目录解析：
裸文件名只匹配根目录，`**/sample-batch*.json` 才包含递归子目录。`--latest [N]` 先按
文件 mtime 选取再解析 JSON，不传 N 时取最新一个；不能与 stdin 一起使用。
`selection.selected_paths` 列出文件，`selection.provenance` 保留路径、匹配来源和 mtime。
显式文件与匹配结果按路径去重；匹配或 latest 选择按文件从新到旧排列，只有显式文件时保持参数顺序。

默认按训练期 fitness 排序；`--sort none` 保留输入顺序。`--freeze PATH` 保存不可覆盖的
训练选择：候选身份及顺序、指标、阈值、来源与成本场景，并校验内容 hash。
`--frozen --selection PATH` 才显示全窗口、测试期和检查证据，复用保存的文件与顺序，拒绝
新的排序/筛选/成本假设。多出的候选忽略，缺失的候选明确列出，测试期分数不能改变训练选择。
`--summary` 只缩短展示，不改变冻结内容或总数。原始模拟结果与 probe 回执都可比较。

成本口径为每美元单边成交额：`margin_bps = raw_margin * 10000`，
`net_margin_bps = margin_bps - cost_bps`，默认情景为 5/10/20 bps。报告保留期间与换手单位；
缺失训练指标保持未知，汇总换手不能推导精确逐日费用或净 Sharpe。
`checks_state` 的枚举为 `hidden`（训练视图隐藏）、`available`（冻结视图有保存的检查证据）、
`unavailable`（没有检查证据）。`failed_checks: null` 代表未知；即使保存的失败列表为空，
也不能据此认定所有平台检查已完成或通过。

`alpha check` 接受一个或多个 ID，按首次出现顺序去重，`--workers` 为 1–3（默认 1），
并发完成先后不改变结果顺序。单个和多个 ID 都返回 `count`、`results`、`complete`、`summary`。
每项的 `status` 为 `complete` / `pending` / `error`，`outcome` 为
`passed` / `failed` / `warning` / `pending` / `error`：已判 FAIL 是完整的失败证据，
但该项若仍有未决检查则 `complete` 仍为 false，可能同时是 `pending` 状态和 `failed` 结果。
全部项证据完整时顶层 `complete` 才为 true；缺失检查不能称为通过。
`summary` 统计三种状态及 passed/failed/warning 结果，单项错误保留在该项而不抹掉其他结果。
`--max-wait` 是每个 alpha 异步检查的等待预算，默认 180 秒，不是整批共享预算。


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
python3 research_cli.py alpha explain ALPHA_ID
```

三层，故意分开：

| 层 | 内容 |
| --- | --- |
| **构造** | 每条腿一句话（"ranked (close minus low) over (high minus low) minus 0.5"），带字段、数据族、权重 |
| **证据** | 指标、Fitness 的算术复算（分母用的是真实换手还是 0.125 地板）、跑过的消融、本地对书相关 |
| **风险** | 族是否已关闭，以及一句提醒：**权重是乘数，不是贡献** |

腿是从表达式树读出来的，所以不会和公式漂移。`0.25*(a+b)` 这种会被摊开成两条腿——否则一个六腿复合体会显示成"一条腿"。

## 提交与本地回写

```bash
python3 research_cli.py alpha submit ALPHA_ID
python3 research_cli.py alpha submit ALPHA_ID --yes --selection choice.json
python3 research_cli.py buffer submit ALPHA_ID --yes --selection choice.json
python3 research_cli.py ledger sync --pnl
```

新提交需要环境提交锁开启、显式 `--yes`、有效 name/category，以及包含该候选的冻结训练选择。
不带 `--yes` 只预检。`alpha submit` 与 `buffer submit` 共用持久提交状态；未知结果再次运行时
只回读平台，不隐式重复 POST。

确认远端成功后自动同步台账、消费候选架、刷新 PnL，并使旧平台组合快照失效，不再需要
`--record`。回执分别报告 `submission` 与 `reconciliation.{ledger,shelf,book_snapshot,pnl}`；
某项本地修复失败不抹掉远端成功，`repair_actions` 列出未完成项。对已确认 alpha 再运行
同一提交命令会修复本地状态，不再次提交。

`ledger sync` 的有效分页大小不超过 100，并保存状态/hidden/type 过滤范围、时间、成员 digest
和完整性。反向漂移只在完整列表后逐条回读确认。membership、metadata、PnL 各自报告缺口；
到达页数或记账上限不算完整同步。PnL 只有完整抓取才覆盖缓存，旁车元数据校验 alpha ID 与
数据 hash；旧缓存完整性为 unknown。台账仍拒绝用缺指标的行覆盖已有指标。

## 工具会读的封闭清单

`<research>/closed-families.json` 保存已关闭方向的 id、类型（`blocked-by-check` / `falsified` / `structurally-impossible`）、理由和证据。

只有带 `match` 的条目能机械匹配；其余技术方向需要结合研究上下文判断，留给人和 agent 读。

读它的地方：

* `parts_catalogue --seeds` —— 来自封闭族的种子会标 `[CLOSED: 理由]` 并排到最后，**不删**（藏起来正是它上次被推荐的原因）。
* `unit_report` —— kind 表里标封闭族，另外单独列出封闭族中的高分项，防止把局部得分误当成重新开放方向的证据。

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

对照必须与要回答的问题相符。若基准本身不满足设计条件，实验臂变差也无法说明是哪项改动造成；对照、设置和变化轴需要随批次一起保存，才能解释差值。

## 边界

* 仓库（`/data/src/magix/WorldQuantUser`）放**工具**：任何能把函数跑两次的东西。
* 研究目录（`/data/worldquant-research`）放**结果**：台账、缓存、一次性脚本、报告。一次性脚本不进 git。
