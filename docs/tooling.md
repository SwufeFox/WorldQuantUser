# 工具地图

按研究问题选择命令。日常使用 `research_cli.py`，后台跑批见文末的 runner。
具体参数以当前 CLI 的 `help` / `schema` 为准。

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

JSON `settings` 也接受 `test_period`、`nan_handling`、`instrument_type` 等已知键的
CLI 拼法；预览、缓存指纹和发出的载荷统一使用 `testPeriod`、`nanHandling`、
`instrumentType`。先逐层归一，再按 CLI、arm、文件、默认值的顺序覆盖；同一对象
里两种拼法给出不同的明确值才报错。未知的平台扩展键保持原样透传。

实现位置也按用途固定：参数、帮助和 schema 在 `research_app/parser.py`；
执行边界在 `app.py`，公共 CLI 约定在 `runtime.py`。目录查询、表达式检查、
实验、平台 alpha、本地台账分别在 `catalog.py`、`validation.py`、
`experiments.py`、`alphas.py`、`ledger.py`。账户与缓存、研究摘要与查看记录、
文献检索分别在 `account.py`、`campaign.py`、`literature.py`。
这些是内部模块，入口仍是 `research_cli.py`；改命令时直接修改、测试所属模块。

在线 lint 和模拟预检通过字段目录核对账户数据范围。FASTEXPR 还检查字段类型与算子，
Python 检查源码的字段声明。`probe` 将预检报告交给 `tools.guarded_sim`，
仅在完整设置与源码匹配时复用。用 `pip install -r requirements.txt` 安装运行依赖。
`pandas` 在调用 `fields_to_df()` 时按需加载，DataFrame 用户再单独安装。


## 给 AI 的信息按阶段进入

常驻入口是 `AGENTS.md`；它只保留启动方法、研究原则和执行约束。
`brief` 给出当前状态；本页负责按问题导航；台账和报告保存完整证据。
全量 `schema` 留给接口集成，不作为每次启动的必读项。

`auth check` 先实时读取当前账户的 `OPTIONS /simulations`，再按每个公布的股票地区选取
合法股票池和 delay 查询目录，不再给所有地区套美股默认股票池。`data_scope_evidence`
保留实际查询范围、目录时效及 available / empty / unknown；它只证明抽查的组合。
错误、旧目录回退和未查询组合不能作为“该地区无数据”的证据。

账户设置证据由核心模块 `simulation_capabilities.py` 统一管理，保留原生 schema、账户指纹、
来源、时效和 capability ID；每次在线模拟/核验刷新一次，批次内复用。`schema` 的
`settings_hints` 只是常见值，地区、delay、字段类型等不再受旧列表限制。字段目录保留完整性，
完整的空目录可以替换旧字段；缺失、过期和未公布的能力仍是 unknown。

平台增加设置时可以直接写入文件 `settings`，或重复使用 `--setting NAME=VALUE`。
值支持 JSON 或普通文本，例如 `--setting visualization=true`；对象、布尔值和扩展键的
显式 null 都保留。通用参数与明确的命名参数冲突时拒绝。离线预览只检查结构；
当前平台证据明确不支持的设置不能用 `--allow-lint-errors` 绕过。


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

平台文档建议先缩小初始搜索空间：结合 Coverage、Date Coverage、字段描述与样本可用性，
从相似字段、算子族、时间尺度和中性化中各选一个有经济依据的代表；得到训练期信号后再
逐项扩展。日期、标识符等技术字段先进入语义核对，不把“数值类型”当成可交易信号。
Alpha Count / User Count 只能描述使用情况，低计数不证明新机制，高计数也不证明没有价值。
BRAIN Labs 的覆盖、典型值、换手和收益相关分析可作补充证据，当前 CLI 不冒充已执行 Labs 分析。

“先做 10–50 次”是探索建议，不是必须花完的预算；应按能否区分解释来安排实验。持续无信号
时先检查模板、数据定义与条件，不自动穷举整个参数笛卡尔积，也不由一次低分否定机制。
文档将 SLOW_AND_FAST 对应到 risk-handled、MaxTrade ON 对应到 investability-constrained；
这提供设置语义，不能单独证明 ON 的结果总与 OFF 的某个返回分块完全一致。合法设置仍查
当前账户的 OPTIONS 元数据，STATISTICAL 等不同风险处理需要单独提出解释与对照。

### Pyramid theme multipliers：平台激励维度

Pyramid 按 **region × delay × 平台数据类别** 定位；theme multiplier 在研究决策中作为
平台激励信息，与经济机制、训练期质量和组合贡献分别考虑。它不等同于本地 `family`，
也不作为 Sharpe、Fitness、PnL 或表达式权重的乘数。

AST 用于识别表达式及所用字段；字段所属的数据集、平台类别仍需从目录元数据核对，不能
由算子、字段名前缀或机制标签猜测。未知类别不自动归入 Other。使用多类别字段时保留候选
归属，最终适用资格、跨类别倍率合成和其他主题的叠加方式以平台规则为准，不自行取最大值、
平均值或乘积。

选题时，在经济解释、数据可用性与执行条件明确的候选之间，将倍率作为分配探索预算的
参考；保留跨主题探索，低倍率不是否决机制的理由。改变 region 或 delay 会改变实验条件，
不能仅为更高倍率把它当成原实验的等价替换。候选保留和验证仍遵循训练期证据、成本与相关性规则。

倍率应带来源、观察时间和适用账户/活动范围，视为该时点的信息；具体数值不固化为程序
常量，截图不能冒充持续有效的平台快照。表格留白只表示未提供数值，不默认填 0 或 1，
也不据此判断数据访问权限。缺少规则时，计分基数和最终奖励金额保持未知。

读取入口已经接入现有 CLI：

```bash
python research_cli.py catalog datasets --region USA --theme true --summary
python research_cli.py catalog fields --dataset fundamental6 --summary --expressions
python research_cli.py alpha get ALPHA_ID
python research_cli.py alpha check ALPHA_ID
```

目录的 `--summary` 最多展示 20 行，成员列表最多展示 5 项，并报告省略数量；不加此选项
保留原生目录记录。`--theme true|false` 仅用于数据集目录，省略时不发送该筛选项，三种查询
分别缓存。结合响应中的请求范围和缓存信息判断来源与新鲜度，需要重新获取时使用
`--refresh-meta` 并检查缓存/错误报告。

Alpha 和 probe 的 `incentives` 摘要保留 `MATCHES_PYRAMID`、`MATCHES_THEMES` 的原生状态及
有限的成员预览，完整 checks 保留平台扩展字段。`alpha get` 的依据为 Alpha 快照，
`alpha check` 的依据为实时检查端点。`effective` 与 `multiplier` 分别保留，不互相推算；
缺失倍率保持未知，主题命中不能覆盖质量检查失败。该能力提供平台报告的信息，不实现奖励计分公式。

### 平台组合前后表现：显式验证视图

```bash
python research_cli.py alpha get ALPHA_ID --performance
python research_cli.py alpha get ALPHA_ID --performance full --team TEAM_ID
python research_cli.py alpha get ALPHA_ID --performance --competition COMPETITION_ID --max-wait 0
```

默认读取个人范围，团队和比赛范围互斥；指定范围必须同时使用 `--performance`。
摘要保留平台 before/after 统计和 score，只对两侧均有有限数值的统计量计算 after-minus-before，
缺失值不补零；默认省略 `pnl`、`yearlyStats`，`full` 保留完整原生响应。
结果带范围、来源 URL 和取得时间。`Retry-After` 等待默认最多 180 秒，`--max-wait 0`
只请求一次；尚未就绪会保留 incomplete 状态和下次读取时间信息。

这份统计的训练期边界未经核验，输出明确标记 `train_only_verified: false`，成功读取记入
validation 视图日志；按冻结后的验证证据使用。命令只读，不触发提交。

### 候选冻结、验证、检查与入库一次完成

```bash
python3 research_cli.py promote ALPHA_ID --note "冻结后验证通过" --cost-bps 5 10 20
python3 research_cli.py promote ALPHA_ID --retention-floor 0.2 --dry-run
python3 research_cli.py triage --ids ALPHA_ID --against-shelf --max-shelf-correlation 0.7 --min-delta 0 --freeze choice.json
python3 research_cli.py promote ALPHA_ID --selection choice.json --max-self-correlation 0.7
```

最接近的现有入口是 `triage`，但它负责选择与验证，不负责平台检查或写入候选架。
`promote` 只串联现有的四步实现，解决每个候选都要等待并手工衔接的问题：
先 `triage --ids --freeze`，再 `triage --selection --show-test`，然后 `alpha check`，
最后满足条件时调用 `buffer add`。冻结时就保存本次 retention floor 和 cost scenarios，
验证沿用同一标准；入库复用第三步的实时检查结果，避免重复等待检查接口。

仅当检查全部 resolved、没有 FAIL/ERROR（WARNING 允许）、`test_sharpe > 0` 且
`retention >= retention_floor` 才入库；floor 默认 0.2，缺失指标不算通过。
不满足条件返回 `ok: true`、`outcome: rejected`、`buffered: false` 和 `reasons`，退出 0。
默认冻结文件为研究目录下 `promote-freeze-<ID>-<UTC timestamp>.json`；
可用 `--freeze PATH` 和 `--buffer PATH` 指定位置，冻结文件保持原有不可变规则。
已有单候选冻结文件时，`promote ALPHA_ID --selection PATH` 检查候选 ID，并沿用里面的
留存门槛和成本情景。此模式和 `--freeze` 互斥。若冻结使用自定义候选架，`promote`
也应传入同一路径。`--max-self-correlation` 另行要求实时平台自相关有数值且严格低于上限。

`triage --against-shelf` 用候选、组合和候选架成员已记录的最早测试起点截断训练窗，
`--train-end` 可以把窗口缩短。`--buffer PATH` 可指定候选架。回执的 `evidence` 保留
时间窗、组合成员摘要、候选与组合的 PnL 覆盖和交集。它还给出全书相关性、最相关的成员和
训练前后半段的边际。候选架相关性另见 `shelf_correlation`，按相关绝对值比较并保留符号。

加上 `--max-shelf-correlation` 后才启用训练筛选。所有待冻结候选都须有可核验的 PnL 与
足够的交集，边际须为正且达到 `--min-delta`，也须通过组合与候选架去重。
候选架最高绝对相关达到上限时不通过。缺训练边界、覆盖或交集时为 `unknown`。
未通过时回执仍保留证据，但不会写冻结文件。若书、候选架成员或候选源码与设置在冻结后变化，
按候选架筛选生成的冻结选择会在查看测试期之前被拒绝。

Campaign 用同一份训练筛选生成的冻结选择调用 `promote`，沿用原来的边际、去重与覆盖证据。
单独运行 `promote ALPHA_ID` 仍走原有冻结流程，适合已经人工审过训练证据的候选。

回执包含 `alpha_id`、`freeze_path`、`validate` 的训练/测试 Sharpe、retention、
rho_book、max_member_rho，以及完整 `checks`、`buffered`、`outcome`、`reasons`。
`steps` 保存各步状态和已完成的冻结、验证及入库证据。任何执行异常立即停止，
返回 `ok: false`、现有类型的 `error` 和退出码，同时在 `data` 保留已完成证据及 `failed_step`；
`--save PATH` 保存完全相同的回执，且不能覆盖冻结文件或候选架。

`promote --dry-run` 仍然读取测试期并执行实时检查。未提供 `--selection` 时也会写冻结文件。
通过时返回 `outcome: would_buffer`，候选架不写入。平台提交另见下文。

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

`--workers` 为 1–8，默认 4；`--max-topics` 限制关键词、种子扩展、固定订阅源与行情任务总数，
超过预算的任务会列为 deferred。`--timeout` 是单次网络请求的超时秒数；检索后按输入顺序
轮流选取，完成先后不会改变结果。
网络调用在锁外；提交结果前会锁住状态并重读 seen，避免并发重复选中。
关键词保留相关性顺序；引用图的顺序、来源和覆盖范围另记在 `retrievals`。
`--min-citations` 默认 0，让新论文保留资格；引用量不替代相关性或经济解释。
`provider_ranks` 记录各查询中的原始位置；兼容字段 `rank` 仍是旧分数，不代表选取顺序。

OpenAlex 使用 `<research>/cache/papers/`。缓存键包含实际请求选项。相同请求并发未命中时
只由持锁者抓取；有效缓存直接复用，过期或损坏才重新读取。缓存写入失败仍保留成功获取的结果。
论文输出包括可用的出版页、DOI 与 PDF 链接；来源未提供的全文保持未知。

`next_actions` 是有上限的建议：失败查询附 CLI 参数数组 `argv`，未搜主题给下一批参数，
无法推导查询的问题给原文件位置，已获得的论文给来源阅读和反例对照任务。
`argv` 是传给 `research_cli.py` 的参数列表，不是可以直接拼接进 shell 的字符串。
中文问题可以保留中文标题，并显式写 `; query: ...`；系统不自动编造翻译。
摘要用 `reference_count` 显示参考文献数量，完整 ID 列表留在单篇记录；同一论文的 DOI 和
OpenAlex ID 会共同参与去重，旧的无 DOI 标题记录继续有效。

### 多来源线索与本地论坛材料

```bash
python3 research_cli.py feed --source all --topic "earnings guidance" --max-topics 12 --summary --dry-run
python3 research_cli.py feed --market coinbase:BTC-USD --book-depth 10 --summary --dry-run
python3 research_cli.py feed --market alpaca:AAPL --market-feed iex --summary --dry-run
```

没有选择来源时沿用论文检索。`--source` 可重复，`all` 选择内置来源及配置文件中的来源；
行情须由 `--market` 或配置文件明确指定市场与品种。`--limit` 限制每项任务的返回条数，
`--total` 限制最终选取条数。各来源共享 `items`，旧的 `papers` 数组已移除。

| 入口 | 实际来源 | 覆盖与访问条件 |
| --- | --- | --- |
| `--source paper` | OpenAlex 元数据、摘要、参考文献与后续引用 | 正文仍由 `paper --id --full-text` 显式读取；引用边不表示赞同 |
| `--source news` | Google News RSS 搜索 | 聚合标题与摘要，不代表已读原始报道或完整新闻覆盖 |
| `--source social` | Bluesky 公共帖子搜索 | 公共端点可能拒绝访问；403 是来源失败，不是空结果 |
| `--source forum` | Hacker News 索引搜索 | 特定社区的故事与讨论线索，不代表全部金融论坛 |
| `--source trends` | Hacker News front_page 索引 | 当前社区注意力，不能推断全网热度或收益方向 |
| `--market coinbase:BTC-USD` | Coinbase Exchange 聚合 L2 盘口 | 单一加密资产交易所；不是股票盘口，也不是逐笔委托队列 |
| `--market alpaca:AAPL` | Alpaca 股票最佳买卖报价 | 需要 `APCA_API_KEY_ID`、`APCA_API_SECRET_KEY`；默认 IEX 仅一个交易所，其他 feed 取决于订阅权限 |

行情每次重新请求，不复用旧快照。`--book-depth` 为本地保留的档数（1–50），不扩展提供方
覆盖，也不会把 Alpaca 的单档报价变成多档盘口。结果保留来源时间、采集时间、市场、feed、
原生数量单位、覆盖范围及价差/失衡等派生观察；缺失或非法档位不会被补成可用的统计量。
这些是当前观察，不能直接冒充历史回测数据。

可以在研究目录的 JSON 配置中接入已有论坛摘录。

```json
{
  "sources": [
    {"id": "community-export", "kind": "forum", "adapter": "file", "path": "forum-excerpts.jsonl"}
  ]
}
```

```bash
python3 research_cli.py feed --sources-file radar-sources.json --source community-export --summary --dry-run
```

每个来源都要有唯一 `id`、`kind`、`adapter`；自定义 ID 不能覆盖内置 ID。
`rss` 接受 `url`，支持 RSS/Atom 和可选的 `{query}` 占位符。`file` 接受 `path`。
`market` 接受 `venue`、`symbol` 和可选 `depth`、`feed`。
另可显式配置 `bluesky`、`hn`、`hn_frontpage`。固定适配器的 kind 分别为 `social`、`forum`、
`trend`、`market`，RSS 和文件的 kind 可选 `news/social/forum/trend/web`。
未知字段、重复 ID 或含凭据的来源 URL 会在任何请求前被拒绝。密钥只放环境或私有 `.env`。

本地 `.jsonl` 每行一个对象，`text` 或 `content` 必填，其余字段可选：

```json
{"id":"comment-1","title":"参数变化的观察","text":"已保存的评论摘录","author":"source-author","parent_id":"thread-1","published_at":null,"published_at_raw":"2个月前","content_scope":"selected_excerpts"}
```

也接受 UTF-8（可带 BOM）的 `.txt`、`.md`，每文件上限 4 MiB。相对路径以来源配置文件目录
为基准。缺少 ID 时生成确定性标识；缺少标题时使用首个非空文本行。导入保留原始记录、作者、
主帖/评论关系、文件路径和行号；`review` 等附加内容留在原始证据里。坏行与可用行分别计数。
`collected_at` 是导入时刻；“6个月前”、年份、无时区日期等不会被捏造成精确发布时间。
文件中的代码始终作为文本保留，不执行。导入摘录不意味着已经抓取完整帖文或访问登录论坛。

先看 `--summary` 的短摘录、来源回执和下一步，再按 `evidence_path` 读取完整记录。
正常运行先将选中证据写入 `<research>/inspiration/evidence/`，再提交草稿与 seen；
`--dry-run` 不写这三者，但缓存和显式 `--save` 仍可写入。来源失败、真正空结果、重复已读
分别记录；检索预算、提供方截断和未取回内容通过 coverage/deferred 信息保留。
相同来源的转帖不能算独立佐证，热度和他人回测指标只产生待核验线索。

`next_actions` 按材料类型提供原始来源、出版时间与历史可观察性核对，论文阅读/引文扩展，
以及失败来源的重试参数。形成研究假设前，分别写清来源声称了什么、已有证据支持什么、
什么结果能够否定自己的解释；真实模拟与提交仍需遵守各自执行边界。

### 从发现走到可核对的阅读证据

```bash
python3 research_cli.py feed --topic "analyst revisions information diffusion" --dry-run
python3 research_cli.py paper --id W2741809807
python3 research_cli.py paper --id W2741809807 --full-text --max-chars 12000
python3 research_cli.py feed --seed W2741809807 --follow references --from-year 0 --dry-run
python3 research_cli.py feed --seed W2741809807 --follow citations --dry-run
```

示例 ID 用于演示接口，实际使用 feed 返回的 ID 和 `next_actions`。先用短摘要判断是否值得读，
再读取单篇完整记录、按需读取正文。正文超过窗口时执行返回的下一段命令，不重新抓取全文。
参考文献扩展用于追溯概念和识别设计；后续引用用于寻找延伸、复核与争议。引用边本身不表示
支持或反对。`--from-year 0` 可在追溯基础文献时解除默认年份门槛。

feed 保留 `openalex_id`、来源、阅读状态、摘要原长度与截断标记；正文始终由显式命令读取。
`paper --id` 复用检索时保存的完整记录。参考文献一次最多核对 100 个已公布 ID，后续引用只取
有界的一页，`coverage` 显示尚未取回、无法解析及被筛选的范围，不把局部阅读冒充穷尽调查。
DOI 的常见前缀形式会统一，已有 seen 历史继续用于去重。

正文使用 OpenAlex 已公布的 GROBID TEI，解压和解析均在本地完成，完整提取文本保存在研究目录。
默认每次返回 12000 字符，上限 100000；下一段带明确 offset。没有摘要、只有 PDF 链接、
没有 TEI、服务访问失败是不同状态。TEI 仍可能漏掉表格、公式或原 PDF 内容，阅读后要核对
样本、识别策略、经济解释和反例，再写研究假设；草稿不会自动填上这些判断。

| 开源工具 | 适用能力 | 本仓库的选择 |
| --- | --- | --- |
| [PyAlex](https://github.com/J535D165/pyalex) | OpenAlex 单篇、筛选、引用及 PDF/TEI 获取 | 沿用现有客户端，接入对应稳定接口与记录缓存 |
| [paperscraper](https://github.com/jannisborn/paperscraper) | 多预印本来源、DOI 到文件的获取流程 | 借鉴“元数据发现与正文获取分开”；当前不引入生物医学数据转储依赖 |
| [semanticscholar](https://github.com/danielnsilva/semanticscholar) | 学术图、推荐与异步分页 | 可作另一个来源；当前改进沿用已有 OpenAlex 证据与缓存 |
| [GROBID](https://github.com/grobidOrg/grobid) | PDF 转结构化 TEI、引文解析 | 使用 OpenAlex 提供的解析结果；无需本机部署 Java/模型服务，官方不保证原生 Windows 支持 |

访问政策以 [OpenAlex 当前认证文档](https://github.com/ourresearch/openalex-help/blob/main/content/api/authentication.md)
和 [全文文档](https://github.com/ourresearch/openalex-help/blob/main/content/access/fulltext.md) 为准。
目前基础查询可不带 key，正文下载需要 key。`OPENALEX_API_KEY` 可从环境或私有 `.env` 读取，
不会进入缓存和回执；不自动注册账户、不自动下载整库，也不把网页或摘要当成已经读过的正文。


论文的输入需求和台账块提示按完整词或短语匹配，保留简单复数。`technology adoption`
不会因含有 `option` 子串而接上期权实验，`steps` 也不会被当作 EPS；这些提示仍须人工复核。

在用户授权的研究范围内，AI 应主动完成可获取的前置事实、比较不同解释、准备下一项实验并
先 dry-run。存在反证时先解释分歧，避免反复微调同一个高分表达式。工具的下一步建议提供证据
入口；模拟、验证和提交仍遵循各自边界。

证据、seen 与草稿各自原子提交，顺序为证据 → 草稿 → seen；前面的写入失败不会消耗 seen。
它们不是跨文件事务：中断可能留下未引用的证据或重复草稿，seen 提交后 stdout 未送达也可能
消耗本次选中条目。`--dry-run` 不写证据/seen/草稿，缓存和显式 `--save` 仍可写入。
Windows 使用系统字节锁，POSIX 使用 flock；进程退出后系统释放锁，旧 `.pid` 文件不会阻塞新任务。


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

### 字段挖掘：先看缺什么定义

```bash
python3 research_cli.py mine --dataset fundamental2 --limit 5
python3 research_cli.py mine --dataset news18 --expr 'vec_sum(reviewed_event_field)' --emit draft.json
```

`mine` 是离线复核，不会登录。`catalog` 展示默认模拟范围、缓存新鲜度及范围不匹配的排除数量；每个字段的 `source` 保留选中快照、年龄和范围状态。旧缓存未声明范围时保持 unknown，不冒充当前账户可用性。

每个数据集的 `top` 是数值 MATRIX 待审线索；`needs_definition` 单列分类编码、日期、缺说明、VECTOR/EVENT 聚合，以及与已用字段同描述的项目。`related_history` 给出对应台账记录；同描述只提示需要复核，不证明 alias 等价。`unused` 只指最新台账可解析表达式没引用过该 id，不含旧报告/未入账实验，更不是新机制数量。字段提取走 AST，不把注释、字符串、局部变量当数据。

换手中位数只用指定 `--registry` 的训练指标，同时给出样本数与缺失训练数量；不拿全窗口补齐，也不把混合表达式的中位数解释成字段更新频率或下一条信号的换手。

自动 `--emit` 不替 VECTOR 选择 `vec_avg`，只生成数值待审项目；经济聚合明确后用 `--expr`。输出仅为探索草稿，原子保存且不覆盖已有 spec；正式实验仍须经济含义/时间/PIT 审计和预注册，再 dry-run、在线 check。不要把草稿的裸 rank 当已确定的经济方向。

`MATRIX` 也可能承载城市、公司注册国家、邮编或技术编号。它们进入分类定义待审列表，不自动生成代码数值的方向性 rank；州税金额和城市数量等数值指标仍正常筛选。定义、编码与时间口径审清后，可用 `--expr` 显式构造分类分组，默认筛选不会替研究者选择这种用法。

### 滚动极值：先查原生算子，再组合

目录没有 `ts_max` 时，若当前账户的 `max` 明确支持多个输入，可把窗口内的延迟值传给它。
下面计算含当日在内、连续 21 个观察值的最大收益；先用默认 `filter=false` 的 `add` 检查完整窗口，
缺一个观察值就保留 NaN，不把剩余值的最大值当成完整窗口结果。

```text
r1 = ts_delay(returns, 1); r2 = ts_delay(returns, 2);
r3 = ts_delay(returns, 3); r4 = ts_delay(returns, 4);
r5 = ts_delay(returns, 5); r6 = ts_delay(returns, 6);
r7 = ts_delay(returns, 7); r8 = ts_delay(returns, 8);
r9 = ts_delay(returns, 9); r10 = ts_delay(returns, 10);
r11 = ts_delay(returns, 11); r12 = ts_delay(returns, 12);
r13 = ts_delay(returns, 13); r14 = ts_delay(returns, 14);
r15 = ts_delay(returns, 15); r16 = ts_delay(returns, 16);
r17 = ts_delay(returns, 17); r18 = ts_delay(returns, 18);
r19 = ts_delay(returns, 19); r20 = ts_delay(returns, 20);
s = add(returns, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10,
        r11, r12, r13, r14, r15, r16, r17, r18, r19, r20);
m = max(returns, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10,
        r11, r12, r13, r14, r15, r16, r17, r18, r19, r20);
if_else(is_nan(s), s, m)
```

长窗口可以分块：将上述 21 日极值按 0、21、42、……、231 日延迟，再取 `max`，
恰好覆盖 0 至 251 日；完整窗口的 NaN 检查也须覆盖相同分块。用 `min` 可构造对应最小值，
仍须核验当前算子签名。观察值窗口不等于自然月或自然年。

多项求和可优先用原生 `add(a,b,c,...)`，避免长串二元 `+` 消耗预检的运算数量预算；
不要打开 `filter=true` 改写缺失语义。各臂只保留最终结果实际读取的变量，随后仍走 lint 和 dry-run。
这一配方只补表达能力，不证明任何极值信号有预测能力。

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

#### Python Alpha

把源码保存为 `reversal.py`。下面的例子读取最近一行收益，返回每只标的一维 `float32` 信号。

```python
from brain.alphas import alpha


@alpha(data=["returns"], store=[])
def reversal(data, store):
    return (-data.returns[-1]).astype("float32")
```

用 `simulate` 预览单个文件，或者把它和其他实验臂放在同一份 `probe` spec 中。

```bash
python3 research_cli.py lint --file reversal.py --setting lookback=0 --offline
python3 research_cli.py simulate reversal.py --setting lookback=0 --dry-run
python3 research_cli.py probe python-reversal.json --dry-run
python3 research_cli.py probe python-reversal.json --check
```

将下面的 JSON 存为与源码同目录的 `python-reversal.json`。

```json
{
  "name": "python-reversal",
  "design": "explore",
  "settings": {"language": "PYTHON", "lookback": 0, "testPeriod": "P1Y"},
  "arms": [{"label": "reversal", "source_file": "reversal.py"}]
}
```

直接读取 `.py` 时，`lint` 和 `simulate` 把整份文件作为一个 Python Alpha。
`probe` 则按每臂的有效设置选择语言。Python 臂可使用相对于 spec 的 `source_file`，
也可用 `expression` 写入完整源码，二者只能选一个。stdin spec 使用内联源码。
探索批次可以混合语言并分组运行。受控实验的对照和干预臂须使用同一语言。

静态检查会解析单个顶层 `@alpha(data=[...])` 函数，核对两个位置参数与字面量字段列表。
`data.universe` 由平台隐式提供。回测运行在平台上，离线 lint 与预览仅解析源码。
在线预检还要求当前账户的 `OPTIONS /simulations` 声明支持 `PYTHON`，并核对同一账户、
region/universe/delay/instrumentType 的字段目录。语言证据不足时会报告
`python_language_unverified`。字段目录无法核验时会报告 `catalog_unverified`。
`--allow-lint-errors` 对 Python 预检无效。
FASTEXPR 算子目录、结构匹配、换手估计和自动消融均不适用于 Python 源码。

设置按平台字典透传。例如 `lookback` 是额外历史行数，完整窗口长度为 `lookback + 1`。
热身阶段行数可能更少。源码负责处理 NaN、原生 dtype、热身与返回形状。
REGULAR 请求统一省略 `selectionHandling` 和 `selectionLimit`，Python 还省略
`unitHandling` 和 `nanHandling`。核对实际请求看预览的 `payload.settings`，
在线证据的 `settings_capabilities.omitted_from_payload` 会列出 Python 请求中涉及的省略项。

持久批次与 checkpoint 保存当时的源码，恢复沿用该版本。源文件路径只保留在本地批次中，
平台请求的 `regular` 字段包含源码。`--save` 和 `--batch-path` 不能覆盖 Probe 引用的源码文件。

#### 批次流程与复用

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

精确身份由设置签名和对应语言的源码标识组成。FASTEXPR 经 AST 规范化，消除注释和
冗余括号等差异。Python 只统一换行符，保留缩进、注释和字符串内的空白，设置以实际发送的
REGULAR 字段为准。
`gate.completed_refs` 给出精确匹配的台账标识、批次和现有训练指标。`gate.structural_refs`
给出 FASTEXPR 同经济块、同骨架的台账线索。`--strict-gate` 继续按 `exact_repeat_keys`
拒绝台账中的已完成精确重复。历史设置缺失时不补上今天的默认值。

预览也会查看当前指定的批次路径。精确匹配的已完成记录见
`gate.checkpoint_completed_refs`，能回读的任务见 `gate.checkpoint_recovery_refs`
以及 `next_actions` 里的 `probe --recover`。这只检查当前默认路径或显式的
`--batch-path`，可以用该选项指向以往的批次。预览不联系平台，给出的引用不会自动替换结果或重提任务。
恢复时优先沿用已记录的任务地址或 Alpha ID。任务身份缺失时才核对平台最近的候选列表，
须有唯一匹配的源码和完整设置。Python 源码仍按保留缩进的规则匹配。台账中旧的部分
`settings_fingerprint` 用于占位记录，不能拿来替代完整请求判重。

```bash
python3 research_cli.py probe sample-batch.spec.json --batch-size 6 --summary --batch-path sample-batch.json --save sample-response.json
python3 research_cli.py probe --recover sample-batch.json --max-wait-seconds 60 --summary
python3 research_cli.py probe --resume sample-batch.json --dry-run --summary
python3 research_cli.py probe --resume sample-batch.json --batch-size 6 --max-wait-seconds 60 --summary
python3 research_cli.py resume checkpoint.json --batch-size 6 --max-wait-seconds 60
```

示例文件放在研究目录；从该目录运行或传绝对路径。`--batch-path` 是持久批次，`--save` 是
输出回执，两者使用不同路径。回执给出 `path`、`run_id`、`started_at`、`request_fingerprint`、
原始 `arm_order`、逐臂状态、`execution`、`recording` 和 `next_actions`。
`--save` 与批次、checkpoint、恢复输入或冻结选择同路径时会在执行前拒绝；错误回执也不会
写入受保护文件。读取保存回执恢复时，同样保护其指向的实际批次。

`probe --summary` 返回训练期表格：label、role、执行状态、S/F、换手、margin_bps、
两段训练 Sharpe 和 alpha_id。展示和预检明细最多 20 条，并给出总数、略去数量与完整批次路径。
缺失显示 `-`，真实的零保留为零；完整批次不截断。读取旧结果可用
`compare sample-batch.json --summary --format markdown`，表格仍装在单个 JSON 回执里。

臂可声明 `role: "candidate"` 或 `role: "diagnostic"`，与实验 `design` 分开。
diagnostic 保留用于归因，但不能进入候选冻结、候选架、晋级或新提交；恢复和刷新不抹掉角色。
旧记录没有声明时保持 unspecified，不从 C1 之类的标签或备注猜测。

每个 POST 之前保存提交意图，拿到任务身份后立即保存。没有身份的未知提交不会自动重发。
`--max-wait-seconds` 默认 300；前组未完成时后组保持排队。`--recover` 只读平台并幂等补账，
接受批次或保存的回执；`--resume` 才显式继续已知排队/已接受任务，支持离线预览。
执行失败、未完成和证据缺失列为 deferred，不自动判成研究失败；旧批次的记账策略不明时
不会推测为允许记账。同批次运行/恢复互斥，`--force` 不能覆盖正在运行的批次。

模拟并发统一使用 `--batch-size N`，不再硬编码上限 3。已验证的账户上限会约束请求；
平台没有公布上限时，默认 3 只是有来源标记的保守本地策略，可显式提高请求预算，
但工具不会把配置当成已获授权的并发。`scheduling` 保留 requested/effective/platform_limit。
`simulation_slots.py` 在同账户、同研究目录内共享持久占位；接受或结果不明的提交不按时间过期，
只有确认完成或拒绝才释放。不同研究目录及工具外启动的任务不共享这份占位。
恢复可调整并发而不改变研究请求指纹；权限收缩时先取回已接受任务，再核验新启动的请求。

`simulate/resume` 使用 `scheduling.quota` 展示配额观察；完整 probe 批次保留
`scheduling[].quota`，`probe --summary` 的 `summary.quota` 仅保留最近一次观察，不复制调度历史。

每日配额、并发占位与请求限流是不同信息。成功的重复模拟及 multi 的每个成功子模拟都会
计数；Alpha 列表只保留首次创建日，不能靠“今天新增多少 Alpha”重建剩余额度。
本工具目前发送单个 REGULAR 请求；API 文档中的 2–10 个对象 multi 数组另需
`MULTI_SIMULATION` 权限，不能把该数组长度当成 `--batch-size` 的含义或账户日限额。

模拟响应头的 `X-Ratelimit-Limit/Remaining/Reset` 作为配额观察保留；Reset 是秒数间隔。
`limit/remaining` 缺失或非法保持 null，`observed_at/reset_at` 使用 Unix 秒，
`reset_seconds` 保留原间隔，`reset_after_seconds` 给出据此计算的剩余间隔。
确认 remaining 为零时停止当前运行的新提交，已接受任务继续取回结果，未提交工作保留
QUEUED 与恢复点；429 不会导致跳到后面的批次继续提交。短期限流仍按 Retry-After 退避。
同账户同研究目录共享观察，工具外任务不受此状态协调，也没有据此预留整日配额。

到达已知重置时刻只把旧证据标为 `reset_due`，不自动将 remaining 加满。没有 Reset 时，
零余额仅停止当前调用；下一次显式调用可获取新响应，避免旧观察永久阻塞。离线 dry-run
保持配额 unknown，不为了估算余额发起模拟，也不硬编码论坛额度或按 EST 自行推算重置时间。

记录集的 HTTP 200 和非空正文不代表就绪，正的 Retry-After 优先。每日 PnL 读取有总等待
预算（默认 180 秒）；超出等待或尝试预算时返回 RATE_LIMITED、保留不完整状态与等待秒数，
不覆盖已有缓存。TIMEOUT、FAIL、ERROR、取消释放模拟占位并计入失败；WARNING 是完成态。
传输错误造成的 UNKNOWN_SUBMISSION 仍需要核对任务身份，不能按终态释放后重新 POST。

认证继续沿用会话缓存和结构化错误。文档的 persona 示例需要人工完成认证，不能把
`input()` 或自动重登循环移入 CLI；此处没有增加另一套凭据文件或自动生物识别登录流程。

目录检查携带账户、region/universe/delay/instrumentType、完整性和时效证据。
`feed` 的关键词只产生阅读线索。提供了明确字段后，才与当前作用域目录核对可用性。

FASTEXPR 从 AST 提取用到的字段与类型，Python 则使用 `@alpha` 的字段声明。
两者先用同账户、同作用域的私有快照定位数据集，再实时刷新所需目录。
过期快照只作定位提示。所用字段的证据缺失、过期或不完整时，预检无法确认可用性。
回执分别保留 `required_fields_only` 核验与全目录完整性。局部核验不覆盖离线用的完整快照。
字段未知或数据集迁移时回到全目录发现。


### 合并已有结果与读取实时检查

```bash
python3 research_cli.py compare sample-batch-a.json sample-batch-b.json
python3 research_cli.py compare --match "**/sample-batch*.json" --latest --sort none
python3 research_cli.py compare --match "sample-batch-a.json" --match "**/sample-batch-b*.json" --latest 2
python3 research_cli.py compare sample-batch.json --freeze choice.json --cost-bps 5 10 20 --summary
python3 research_cli.py compare --frozen --selection choice.json --summary
python3 research_cli.py compare sample-batch.json --summary --format markdown
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

#### 请求超时或没有结果文件时

`--save` 在命令完成后原子写入最终 JSON，不是实时检查点。默认串行检查两个 Alpha，
仅异步等待预算就可能达到 2 × 180 秒，另有认证、登录锁及网络等待；外层 `timeout 240`
直接终止进程时，可能还没有最终文件。退出码 124 只说明外层时间预算耗尽，不能证明平台过载。
认证 POST 与通用目录请求使用现有的 60 秒连接/读取超时；这不是整个命令的墙钟上限。

先用 `cache info` 读取本地会话状态，不打印缓存里的凭据。定位无输出时，Python 使用 `-u`，
阶段信息与堆栈写 stderr；在一次诊断中区分导入、认证、请求及异步等待，不反复执行相同的长调用。
下面是 Linux 示例，先把 `CHECK_DIR` 改为 `brief` 返回的研究目录，每次只查一个 Alpha：

```bash
CHECK_DIR=/data/worldquant-research
check_run="$CHECK_DIR/check-$(date -u +%Y%m%dT%H%M%SZ)"
timeout --signal=INT --kill-after=5s 120s \
  python -u -c 'import faulthandler, runpy; faulthandler.dump_traceback_later(20, repeat=True); runpy.run_path("research_cli.py", run_name="__main__")' \
  alpha check ALPHA_ID --workers 1 --max-wait 30 --save "$check_run.json" \
  2>"$check_run.stderr.log"
```

SIGINT 给 CLI 保存 `INTERRUPTED` 响应的机会，5 秒后仍不退出才强制终止；强制终止仍可能没有
最终 JSON，此时保留 stderr 堆栈。另一个候选单独执行并保存到另一文件，已经获得的证据不会
因后一个调用被终止而丢失。文件存在、顶层 `ok` 为 true 都不能代替每项检查的完成状态。

查看外部 runner 时用 `pgrep -af '[a]tom_runner\.py'` 避免匹配查询它的 shell 自身。
进程存在不证明剩余模拟仍在运行，`done` 行数也不包含失败与未知状态；核对已保存的任务
Location/Alpha ID 后只读恢复，不重跑整份 spec。直接 runner 的日志需要逐项查看终态，
不能只等进程消失。

`GET /alphas/{id}` 的 checks 是快照；生产相关、SELF_CORRELATION 与 OLD_SIMULATION 要保留
实时检查的状态、值与阈值，未取得或 PENDING 都记为未知。`dateCreated` 是创建时间，不能
单独替代最近成功模拟的证据或平台有效期检查。字段听起来冷门、self correlation 较低，
都不能推出 production correlation 更低；这些硬条件未核实前不作提交排序。


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

Python Alpha 的 `alpha explain` 仍可输出台账指标、Fitness 复算与本地相关性。
自动拆腿的 `construction.available` 为 false。

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

首次带 `--yes` 的新提交通过预检后返回 `CONFIRM_REQUIRED`。十分钟内用同一入口、冻结选择
及强制选项重发，才会尝试提交。

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

不带对照的批次把 `"design": "explore"` 写在 spec 顶层。混合 Python 与 FASTEXPR 的批次
也使用 `explore`。需要归因时，选择同一语言的对照，并在各臂显式记录变化轴。

对照必须与要回答的问题相符。若基准本身不满足设计条件，实验臂变差也无法说明是哪项改动造成；对照、设置和变化轴需要随批次一起保存，才能解释差值。

## 边界

* 仓库（`/data/src/magix/WorldQuantUser`）放**工具**：任何能把函数跑两次的东西。
* 研究目录（`/data/worldquant-research`）放**结果**：台账、缓存、一次性脚本、报告。一次性脚本不进 git。

## 后台研究工具

| 工具 | 职责 |
| --- | --- |
| `tools/sim_queue.py` | 消费 `queue/` 中的 Probe spec，保存批次与回执 |
| `tools/campaign_runner.py` | 消费 `backlog/`，冻结训练期去重与组合边际证据后调用 `promote` |
| `tools/atom_runner.py` | 把 `label / expr / settings` 列表交给 `probe`，用可恢复批次保存进度，把全 IS 指标和 checks 写成 `atom-results.jsonl` |
| `tools/atom_screen.py` | 从 ATOM 账本按平台子宇宙检查初筛，同一 Alpha 分数相同时采用最新记录 |
| `tools/spc_client.py` | 独立预测竞赛接口，提供 `recon` / `validate` / `submit` |

队列与 ATOM 共用 `probe` 的预检与调度。研究目录的优先级依次为 `--research-dir`、
`BRAIN_RESEARCH_DIR`、兼容的 `WQ_RESEARCH_DIR`、CLI 默认目录。
两个队列都接受 `--root` 作为 `--research-dir` 的别名。

```bash
python3 tools/sim_queue.py --research-dir RESEARCH_DIR --once
python3 tools/campaign_runner.py --research-dir RESEARCH_DIR --once
python3 tools/atom_runner.py arms.json --research-dir RESEARCH_DIR --dry-run
python3 tools/atom_runner.py --recover BATCH --research-dir RESEARCH_DIR
python3 tools/atom_runner.py --resume BATCH --research-dir RESEARCH_DIR
python3 tools/atom_screen.py --research-dir RESEARCH_DIR
```

把 `.json` Probe spec 放到所选队列的入口目录。下面的路径都相对于研究目录，`NAME` 是
spec 去掉扩展名后的文件名。

| 队列 | 入口 | 批次 | CLI 回执 | 日志 |
| --- | --- | --- | --- | --- |
| Simulation queue | `queue/NAME.json` | `queue/batches/NAME-batch.json` | `NAME-result.json` | `queue/queue.log` |
| Campaign | `backlog/NAME.json` | `backlog-results/NAME-batch.json` | `backlog-results/NAME-result.json` | `campaign-log.jsonl` |

两个队列的 `--once` 只处理排序后的首份 spec。省略该选项时，worker 会持续轮询。
已知终态才归档到对应的 `done/` 或 `failed/`。需要补证据、重新轮询或只读恢复时，原 spec
保留在入口目录。恢复前核对源码与设置指纹以及实验角色，旧批次指向多个可能结果时要求显式恢复。
归档会移动 spec。回读已发出的请求时，使用保存了完整源码的批次文件。

| `--once` 退出码 | 含义 |
| --- | --- |
| 0 | 队列为空或本次完成 |
| 1 | 本次终态失败 |
| 3 | 仍待恢复、训练证据待补，或已有另一个 worker |

Campaign 先过滤诊断臂，再按下表筛选候选。`triage` 在同一训练窗核对组合边际与候选架去重。
通过后把冻结选择交给 `promote --selection`。测试期读取、实时自相关和候选入架都发生在此之后。

| 步骤 | 当前条件 |
| --- | --- |
| 训练指标初筛 | Sharpe ≥ 1.25，Fitness ≥ 1.0，训练期前后两半的 Sharpe 均大于 0 |
| `triage` 冻结 | 训练边际为正，未被组合判为重复，有完整测量证据，候选架最大绝对相关低于 0.70 |
| `promote` 验证 | test Sharpe 为正，达到冻结的留存门槛，实时平台 checks 通过，实时自相关低于 0.70 |

Campaign 的冻结文件与 `promote-ALPHA_ID.json` 回执也放在 `backlog-results/`。

ATOM 的默认批次路径在 `checkpoints/`，以臂清单内容生成名称。已有同名批次时使用
`--recover` 或 `--resume`。`--recover` 回读已知任务，`--resume` 也可继续已知未提交的臂。
`--dry-run` 可用于新扫描和续跑预览，跳过结果账本写入。
`atom-results.jsonl` 保存带 `metrics_scope: full_is` 的指标与平台检查快照，供子宇宙初筛。
常规训练筛选使用训练期回执，提交资格仍需实时平台 checks。

`spc_client recon` 从 `OPTIONS /competitions/spc/submissions` 读取实时 schema。
`validate` 按 `modelFamily` 和 `sampleOutput` 等字段校验 spec。
`submit SPEC.json --yes` 是单独的比赛提交入口，需要明确授权。
