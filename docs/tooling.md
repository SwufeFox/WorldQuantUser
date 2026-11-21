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

CLI 统一按需加载处理模块。执行具体命令时才导入它的处理模块，`help`、`--help` 和
`schema` 直接从参数定义生成输出。这些命令与离线 lint、Probe 预览可以在未安装
HTTP 客户端 `requests` 的环境下运行。联网命令需要安装运行依赖。

`catalog datasets`、`fields`、`find` 和 `categories` 优先使用匹配当前账户与查询范围的
磁盘缓存，未命中时在同一次命令中复用会话。`--relogin` 和 `--no-cache` 强制刷新会话。
`--refresh-meta` 刷新目录内容，`--meta-max-age` 可缩短允许的缓存年龄，取 `0` 时跳过
目录缓存。刷新失败时，同账户旧快照可以降级返回，回执标明 `stale`、`degraded` 和错误类型。
字段类型、关键词和条数过滤可从同账户、同范围的新鲜完整目录派生，回执标明
`derived_from_complete`。显式刷新仍请求平台。

在线 lint 和模拟预检通过字段目录核对账户数据范围。FASTEXPR 还检查字段类型与算子，
Python 检查源码的字段声明。同次 `probe` 可复用已经在线核验的同账户算子和同范围字段，
每组仍独立核对设置。`probe` 将预检报告交给 `tools.guarded_sim`，
仅在完整设置与源码匹配时复用。用 `pip install -r requirements.txt` 安装运行依赖。
`pandas` 在调用 `fields_to_df()` 时按需加载，DataFrame 用户再单独安装。

## 读平台文档和论坛

```bash
python research_cli.py docs get --query "power pool" --text
python research_cli.py docs list --query submission --summary
python research_cli.py docs get consultant-submission-tests --text
python research_cli.py docs articles 41681062887191 --limit 5
python research_cli.py docs article 41681217386903
python research_cli.py forum topics --limit 5
python research_cli.py forum search "Systematic Predictions Challenge" --limit 5
python research_cli.py forum get 41113878259479
python research_cli.py forum comments 41996433774359 --limit 5
python research_cli.py forum get 41113878259479 --offline
python research_cli.py docs get how-translate-fast-expression-alpha-python-alpha --refresh-meta
```

知道主题时用 `docs get --query KEYWORD --text` 直接读。标题或 ID 的完整匹配优先，
匹配到多个页面时从 `error.details.pages` 选 ID。`docs list --summary` 返回扁平的标题、ID 和链接列表。
`--text` 将正文放在 `data.text`，并在 `data.unsupported_blocks` 标明无法转换的内容块类型。
省略 `--text` 时，`docs get` 返回原始 `content` 块。两种视图都保留来源与更新时间。
`docs articles` 按分类列英语 Help Center 文章，`docs article` 读取 `body`。
`forum get` 的 `details` 和 `forum comments` 的 `body` 保留原始 HTML。

论坛搜索和 Help Center 分类用 `--page` 翻页。论坛主题和评论用 `--cursor` 接收
上页的 `pagination.after_cursor`。列表每次返回一页，`--limit` 控制当前页大小。

`docs` 和 `forum` 按账户、读取对象和分页参数缓存。文档列表的 `--query` 在本地过滤，
`docs get` 解析页面链接时也复用文档目录。默认缓存有效期为 24 小时，命中时直接从磁盘读取，
只有缺失、过期或主动刷新才建立平台会话和发请求。
响应的 `data.cache` 给出命中状态、保存时间和缓存年龄。

`--offline` 只读当前账户的有效缓存，缺失或过期返回 `NOT_FOUND`。
`--meta-max-age SECONDS` 控制允许的缓存年龄，`0` 停用内容缓存。
`--refresh-meta` 刷新内容，`--relogin` 同时刷新登录，`--no-cache` 跳过内容缓存并重新登录。
后三项与 `--offline` 互斥。

这些是账户内的只读入口。平台文档的 `/tutorials`、`/tutorial-pages/{slug}` 和
论坛登录的 `/authentication/support` 来自 BRAIN 网页自身的调用。当前未核实到
这三条路由的公开开发者契约，网页改版可能影响它们。

SSO 后的论坛和 Help Center 读取使用公开文档化的 Zendesk API，运行在独立会话里。
需要把本次读取留作证据时，用 `--save` 写到研究目录。
[Zendesk 帖子 API](https://developer.zendesk.com/api-reference/help_center/help-center-api/posts/) ·
[Zendesk 搜索 API](https://developer.zendesk.com/api-reference/help_center/help-center-api/help_center_search/)


## 生成 AQP 战略地图

```bash
python3 tools/aqp_strategy_maps.py --cycle 202611 --output-dir /path/to/research/aqp/202611
python3 tools/aqp_strategy_maps.py --cycle 202611 --output-dir /path/to/another-view \
  --roster /path/to/research/aqp/202611/aqp-roster.json
```

第一次读取账户可见的已提交 ACTIVE USA Fast Expression Regular Alpha，保留 D0 和 D1。
两张图共用这份名单，按 Alpha ID 连接本地台账和已有 PnL 缓存。带旧周期标签的条目仍可在地图上
比较。名单快照与图输出到指定的研究目录。使用 `--roster` 时，按快照及本地缓存离线重算。

主题图按台账家族和机制标签分组，并能查看平台原生 Theme 回执。台账分类、表达式结构和
目录字段是不同证据。只有平台明确命中才归入原生 Theme。没有明确归类的条目保留其证据状态。

相关图逐对取各 Alpha 的训练窗内日频 PnL，按共同交易日对齐，计算带符号的 Pearson 相关。
默认至少重叠 200 日。聚类采用完全链接，按两簇之间最弱的一对绝对相关决定能否合并。
筛选阈值默认 0.70，可在图上调整。灰格表示未测量，相关簇及单条高相关边分别计数。
图中结果是训练期收益的关系，平台生产相关、持仓相似度和 AQP 评分需要各自的证据。

## 给 AI 的信息按阶段进入

常驻入口是 `AGENTS.md`；它只保留启动方法、研究原则和执行约束。
`brief` 给出当前状态；本页负责按问题导航；台账和报告保存完整证据。
全量 `schema` 留给接口集成，不作为每次启动的必读项。

`auth check` 先实时读取当前账户的 `OPTIONS /simulations`，再按每个公布的股票地区选取
合法股票池和 delay 查询目录，不再给所有地区套美股默认股票池。`data_scope_evidence`
保留实际查询范围、目录时效及 available / empty / unknown；它只证明抽查的组合。
错误、旧目录回退和未查询组合不能作为“该地区无数据”的证据。
`auth check` 和 `api probe` 从读取会话开始使用同一个 25 秒探测预算。单次探测的 HTTP
超时至多 5 秒。预算用完后保留已取得的证据，其余项目标记为未完成。

BRAIN 会话缓存跟随当前配置的账户。只有用户名一致且仍有效的会话才会复用。
旧缓存缺少用户名绑定时会跳过，下次正常登录后自动写入绑定当前账户的会话。

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

默认读取个人范围。团队和比赛范围互斥，且需要 `--performance`。摘要保留平台原生
before/after 统计和 score，对两侧均为有限数值的统计量计算差值。缺失值保持未知。
需要 `pnl`、`yearlyStats` 等完整原生响应时，使用 `--performance full`。

### 读取 Alpha 的诊断记录集

```bash
python research_cli.py alpha get ALPHA_ID --recordsets
python research_cli.py alpha get ALPHA_ID --recordsets yearly-stats coverage turnover
python research_cli.py alpha get ALPHA_ID --recordsets pnl-by-sector --max-wait 0
```

`--recordsets` 单独使用时列出平台公布的记录集。传入名称时，还会读取相应的原生
`schema` 和 `records`。目录缺失时仍可按名称显式读取，例如某些 Alpha 的年度统计。
每条结果携带来源、读取时间和可用状态，保留单次响应。无法根据声明行数或分页标记判断
完整性时，`complete` 为 `null`。

日覆盖、换手和分组图表取决于该 Alpha 是否有可读取的 Visualization 数据。平台文档说明
图表数据在模拟后保留一周。Alpha 覆盖衡量按日变化的股票数，字段缺失原因仍需另行核查。

`--performance` 和 `--recordsets` 都是只读诊断，训练边界未经核验，回执标明
`train_only_verified: false`。前后表现已就绪或记录集返回非空序列时，写入验证视图日志。
每类视图单独使用一份 `--max-wait` 预算，默认 180 秒，包含 `Retry-After` 等待。取
`--max-wait 0` 时每个端点只请求一次。用 `--save` 将回执保存到研究目录供后续复用。
[官方 API](https://platform.worldquantbrain.com/learn/documentation/brain-api/brain-api) ·
[Visualization 说明](https://platform.worldquantbrain.com/learn/documentation/consultant-information/visualization-tool)

### 候选冻结、验证、检查与入库一次完成

```bash
python3 research_cli.py promote ALPHA_ID --note "冻结后验证通过" --cost-bps 5 10 20
python3 research_cli.py promote ALPHA_ID --retention-floor 0.2 --dry-run
python3 research_cli.py triage --ids ALPHA_ID --against-shelf --max-shelf-correlation 0.7 --min-delta 0 --freeze choice.json
python3 research_cli.py promote ALPHA_ID --selection choice.json --max-self-correlation 0.7
```

最接近的现有入口是 `triage`，但它负责选择与验证，不负责平台检查或写入候选架。
`promote` 只串联现有的四步实现，解决每个候选都要等待并手工衔接的问题：
先 `triage --ids --freeze`，再 `triage --selection --show-test`。测试 Sharpe 与留存通过后
运行 `alpha check`，最后满足条件时调用 `buffer add`。冻结时就保存本次 retention floor 和 cost scenarios，
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
rho_book、max_member_rho，以及 `checks`、`buffered`、`outcome`、`reasons`。
测试期门未通过时，`steps.checks` 为 `skipped`。
`steps` 保存各步状态和已完成的冻结、验证及入库证据。任何执行异常立即停止，
返回 `ok: false`、现有类型的 `error` 和退出码，同时在 `data` 保留已完成证据及 `failed_step`；
`--save PATH` 保存完全相同的回执，且不能覆盖冻结文件或候选架。

`promote --dry-run` 同样读取冻结后的测试期，测试期门通过后执行实时检查。未提供 `--selection` 时也会写冻结文件。
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

### 导入外部定向检索

外部检索工具可以按经济机制分别搜集交易员、投资者和量化研究者的观察。保留实际检索式，
打开候选链接核对作者原文，再把检索回执和读到的摘录交给 `feed`。配置和导出文件放在研究目录。

在 `radar-sources.json` 中声明一个 `format` 为 `harvest` 的本地来源。

```json
{
  "sources": [
    {"id": "web-harvest", "kind": "web", "adapter": "file", "path": "harvest.jsonl", "format": "harvest"}
  ]
}
```

`harvest.jsonl` 每行一个对象。下面的域名和内容仅用于演示记录格式。

```jsonl
{"record_type":"search_receipt","query":"relative strength pullback","audience":"trader","searched_at":"2026-11-17T08:00:00Z","status":"complete","results":[{"url":"https://example.org/trader-notes","status":"opened"},{"url":"https://example.org/search-only","status":"snippet_only"},{"url":"https://example.org/restricted","status":"blocked"}]}
{"record_type":"observation","query":"relative strength pullback","audience":"trader","read_status":"opened_original","read_at":"2026-11-17T08:05:00Z","url":"https://example.org/trader-notes","title":"Relative strength pullbacks","text":"作者观察到，部分走势强于同行的股票在低量回撤后恢复上涨。","review":{"source_claim":"作者描述了低量回撤后的走势恢复。","mechanism":"短期卖压可能在回撤期间消退。","variables":["行业内相对强弱","回撤深度","回撤期间成交量"],"falsification":"控制回撤深度和波动率后，低量组后续收益没有改善。","control":"匹配回撤深度和波动率的普通成交量回撤","classification":"classic_baseline","closest_family":"相对强弱与短期反转","overlap_note":"需要核对已有趋势回撤实验的成交量对照。"}}
```

| 记录 | 必填信息 | 进入 feed 的位置 |
| --- | --- | --- |
| `search_receipt` | `query`，`audience`，带时区的 ISO 时间 `searched_at`，`status` 为 `complete` 或 `failed`，`results` 为包含 `url` 与 `status` 的数组。结果状态为 `opened`、`snippet_only` 或 `blocked`。 | 完整输出的 `search_receipts`。来源覆盖记录回执数量 `search_receipt_count`、检索状态 `search_status_counts`、结果状态 `search_result_counts` 和各视角的次数 `search_queries_by_audience`。 |
| `observation` | `query`，`audience`，`read_status: opened_original`，带时区的 ISO 时间 `read_at`，公开 `url`，以及 `text` 或 `content`。可沿用普通导出的标题、作者、发布日期和 `content_scope`。 | `items`、证据与草稿。完整输出、摘要和草稿保留 `harvest` 及可选 `review`。 |

`audience` 可取 `trader`、`investor` 或 `quant`。只看到搜索摘要或遇到访问限制的链接留在回执里。
`search_queries_by_audience` 统计各类受众的检索尝试次数，包含失败与重试。`search_result_counts`
统计回执报告的所有结果行。`--limit`、`--total` 和 seen 去重针对可入选的观察。

检索结果与已读状态由采集者报告。`read_at` 是实际阅读时间，`collected_at` 表示本地导入时间。
来源的陈述进入 feed 后仍处于 `unverified_claim` 状态。

`review` 各字段可选。用 `source_claim` 写作者的原意，用 `mechanism`、`variables`、`falsification`
和 `control` 写自己的研究设计。`variables` 的数组元素须是非空字符串。`classification` 可取
`classic_baseline` 或 `new_lead`。`closest_family` 和 `overlap_note` 记录需要核对的既有研究。
同一原文尽量使用同一个链接，不同网址仍可分别出现。判断转载和机制重复需要对照原文与已有试验。

`--summary` 中 `review` 的文本字段和变量名称各最多显示 200 个字符，变量最多显示前 8 个。
截断时标记 `review_truncated`。完整输出及草稿保留完整备注和原有换行。

```bash
python3 research_cli.py feed --sources-file radar-sources.json --source web-harvest --summary --action --dry-run
python3 research_cli.py feed --sources-file radar-sources.json --source paper --source web-harvest --topic "relative strength pullback" --summary --dry-run
python3 research_cli.py feed --sources-file radar-sources.json --source web-harvest --action --draft-cards research-drafts.jsonl
```

检索和打开网页都由采集者完成。单独运行本地来源只读取导出文件。组合命令还会向 OpenAlex 检索论文。
去掉 `--summary` 可查看完整回执。

`--action` 的完整输出为结构化观察列出 `block_guess`、`book_exposure` 和 `prior_trials`。
摘要的 `ledger_hint` 汇总这些线索，供复核已有试验。

确认摘录、回执和研究假设后，用最后一条命令写入证据、草稿和 seen。相对草稿路径以当前目录为基准。

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

`catalog find` 的每条命中分别保留平台目录的 `coverage` 和 `dateCoverage`，后者在
输出中为 `date_coverage`，缺失时为 `null`。先对照截面覆盖和日期覆盖，再决定是否
需要查具体日期的数据。排序和 `--min-coverage` 仍使用 `coverage`。

可以省略关键词，按报告的字段使用量、覆盖或类型筛选多个数据集。

```bash
python3 research_cli.py catalog find --region EUR --universe TOP2500 --delay 1 \
  --dataset analyst69,analyst15,model109 --type MATRIX --max-alpha-count 0 \
  --min-coverage 0.8 --sort pyramid-multiplier --limit 20 --stream
```

`--max-alpha-count 0` 只接收平台目录明确报告 `alphaCount=0` 的字段。符合其他筛选条件但计数未知的字段计入
`alpha_count_unknown`。`--sort pyramid-multiplier` 对全部匹配排序，缺失倍率排在最后，`--limit` 只限制输出。
字段目录完整分页后按账户、数据集和范围缓存，换本地筛选条件时可重用。`--type MATRIX` 优先复用匹配缓存，
冷读时直接向平台请求 MATRIX 分页。使用目录倍率来找线索时，仍需核对字段定义。

`--stream` 将扫描进度写到 stderr。`discovery.complete` 表示请求范围的目录是否完整读取或命中新鲜缓存。
每个数据集报告扫描行数、匹配数和错误。`matches: null` 表示未取得可筛选的记录，`matches: 0` 表示在读到的记录中没有匹配。
遇到最终 `RATE_LIMITED` 时保留已有结果，后续数据集标为 `not_scanned`。

```bash
python3 research_cli.py mine --dataset fundamental2 --limit 5
python3 research_cli.py mine --dataset news18 --expr 'vec_sum(reviewed_event_field)' --emit draft.json
```

`mine` 和 `catalog unused` 都从本地缓存读取字段，并用 AST 识别台账里的字段引用。
`mine` 使用默认模拟范围，`catalog` 展示缓存新鲜度和范围不匹配的排除数量，每个字段的 `source` 保留选中快照及账户信息。
`catalog unused` 遍历所有缓存范围，每个字段 ID 只计一次。优先选择匹配当前账户的快照，再按新鲜度和保存时间选择。
两条命令都会跳过其他账户的快照，未绑定账户的旧目录只作为账户归属未知的研究线索。

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

### 反复查询训练期 PnL

跨批次查看训练期两半 Sharpe 时，可以给 `ledger stability` 加 `--polars`。
`--where` 和 `--status` 可继续用来筛选同一份分析缓存。

```bash
pip install "polars>=2.1,<3"
python3 research_cli.py ledger stability --polars --list
python3 research_cli.py ledger stability --polars --where 'family=quality'
```

命令从本地日度 PnL 建立 `cache/pnl-train.parquet`。只缓存各 Alpha 已记录的测试起始日之前的数据。
缺少训练边界时，两半的 Sharpe 保持未知。日期排序和 Sharpe 口径与普通 `ledger stability` 一致。
返回的 `pnl_snapshot.state` 表示复用或重建。命中时直接读取已算好的两半 Sharpe，
筛选和翻转判定仍按本次参数执行。有可复用的快照时，PnL 文件或训练边界更新只重读变更的 Alpha，
再按 `alpha_id` 和日期把新增训练行与保留行流式合并到 Parquet。旧版或损坏的快照会全量重建。
本轮刷新遇到源文件读取错误时，`source_unavailable` 表示已回退到普通读法，后续调用会重试刷新。

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

Python 的 `lookback` 是额外历史行数，最多可用 `lookback + 1` 行，热身阶段可能更少。
静态检查用每臂的生效设置比较直接字段访问。`data.close[-N]` 所需行数超过上限时
提示越界风险，`data.close[-N:]` 提示切片可能截短。两项都是警告，用来复核源码的
保护逻辑，检查范围限于上述字面量访问。源码负责处理 NaN、原生 dtype、热身与返回形状。
REGULAR 请求统一省略 `selectionHandling` 和 `selectionLimit`，Python 还省略
`unitHandling` 和 `nanHandling`。核对实际请求看预览的 `payload.settings`，
在线预检与执行器都按实际发送的设置验证。`settings_capabilities.omitted_from_payload`
会列出输入中被省略的设置。

持久批次与 checkpoint 保存当时的源码，恢复沿用该版本。源文件路径只保留在本地批次中，
平台请求的 `regular` 字段包含源码。`--save` 和 `--batch-path` 不能覆盖 Probe 引用的源码文件。

#### 批次流程与复用

| 问题 | 命令 |
| --- | --- |
| 我要跑一批臂（常态） | 写一个 sample-batch.spec.json，然后 `python3 research_cli.py probe sample-batch.spec.json`——lint、模拟、train-only 表格、对照差值、JSON、台账一次做完 |
| 不登录、不花额度，先查看整批请求 | `python3 research_cli.py probe sample-batch.spec.json --dry-run`；输出完整 payload、设计上下文和精确重复证据，正式模拟前再做实时目录检查 |
| 按当前账户核验这份 spec，但不模拟 | `python3 research_cli.py probe sample-batch.spec.json --check`；使用与正式运行相同的臂和生效设置 |
| 已经在网页或其他客户端跑过模拟 | `python3 research_cli.py probe --from-simulation SIMULATION_ID`，读取任务状态和已有 Alpha ID |
| 我要开一个单次实验 | 直接写一个两三条臂的 spec，`python3 research_cli.py probe sample-batch.spec.json`（没有模板文件了） |
| 这批臂相对对照变了多少 | 看 `comparisons.rows`：arm 减 control 的训练期差值；缺失项列在 `unavailable`，explore 批次不计算归因 |
| 换字段 / 换窗口 / 换权重的批量实验 | `research_app/substitution.py` 的 `expand()` 生成 `(label, expression)` 对，再落成 spec |
| 我想跑一批，但怕名字写错 | `tools/guarded_sim.py`（库，不是 CLI） |
| 这个池子够不够厚 | `python3 research_cli.py health --pool` |
| 这个字段到底什么量纲 | `python3 tools/probe_units.py` |

`health` 用共同交易日评估账本。默认训练窗截止于 `--train-end` 和可测成员已记录测试起点中的最早日期，
池内每条候选还受自身测试起点约束。最低重叠天数与 `triage` 相同，可用 `--min-overlap` 提高。
`--pool` 对有缓存的成员计算日频相关性，报告与通过计数使用同一次测量，返回每条候选的截止日期和重叠天数。

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
任务进入终态且没有 Alpha ID 时，回执计入 `execution.terminal`，结束该任务的回读。

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

`probe --from-simulation` 接受已有任务 ID 或
`https://api.worldquantbrain.com/simulations/ID`。多模拟按父任务返回顺序读取子任务，
单模拟沿用同一回执结构。`rows` 保留任务状态，`alpha_ids` 收录已取得的 Alpha ID。
任务返回 `Retry-After` 时记录建议等待时间并结束该任务的读取。子任务列表暂不可取时，
`children_state` 为 `unavailable`。

需要进一步分析时，用 `--save` 保存回执，对未入账的 Alpha 显式执行
`ledger record ALPHA_ID`，再用 `triage --batch PATH` 读取对应训练证据。
`triage --batch` 也接受模拟和检查回执的 `results`，以及模拟 checkpoint 的 `records`，默认仍读取训练视图。
`--from-simulation` 本身只读现有任务，回执以任务身份和状态为主。
[BRAIN API 的父子任务说明](https://platform.worldquantbrain.com/learn/documentation/brain-api/brain-api)

`probe --summary` 返回训练期表格：label、role、执行状态、S/F、换手、margin_bps、
两段训练 Sharpe 和 alpha_id。展示和预检明细最多 20 条，并给出总数、略去数量与完整批次路径。
缺失显示 `-`，真实的零保留为零；完整批次不截断。读取旧结果可用
`compare sample-batch.json --summary --format markdown`，表格仍装在单个 JSON 回执里。

臂可声明 `role: "candidate"` 或 `role: "diagnostic"`，与实验 `design` 分开。
diagnostic 保留用于归因，但不能进入候选冻结、候选架、晋级或新提交；恢复和刷新不抹掉角色。
旧记录没有声明时保持 unspecified，不从 C1 之类的标签或备注猜测。

每个 POST 之前保存提交意图，拿到任务身份后立即保存。准备阶段中断时，下次调度会回收
可确认尚未发出 POST 的占位，保存的批次可恢复为排队。若槽位账本比批次先保存了任务地址，
中断后 `resume`、`probe --resume` 和 `probe --recover` 会用同一预约补回地址，
由对应账户继续查询。取得 Alpha ID 后，指标读取中断可直接沿用该 ID。没有确切任务身份的
发单尝试保持结果未知，需先核对身份。

`simulate` 和 `resume` 对同一份 checkpoint 互斥。结果写入台账后才清理已完成的自动
checkpoint。收到取消或记账写入失败时，错误回执给出已保留的 `error.details.checkpoint`。
修复写入问题后，可用 `resume PATH` 向原台账补记已完成的结果。

`--max-wait-seconds` 默认 300 秒。登录、预检、模拟、恢复查询和 PnL 补读共用一次等待预算。
预算耗尽时，未发出的工作保持排队。任务进入终态后立即释放槽位，空位可在同一组接下一条任务。
前组若只剩指标待取，`probe` 会先启动后续设置组，再在剩余预算内补读。指标尚未生成时，
回执保留 Alpha ID、等待状态和建议重试时间。`--recover` 接受批次或保存的回执，只读现有任务并幂等补账。
连接中断和暂时的网关错误允许在预算内重试读取。重试后仍无法读取时，已接受模拟保留原有任务
或 Alpha ID，供 `resume` 使用。网关错误或超时造成的未知提交要先回读平台状态。
`--resume` 继续已排队或已接受的任务，支持离线预览。同批次的运行和恢复持有互斥锁。
执行失败、未完成和证据缺失列为 deferred。旧批次的记账策略不明确时，保持未补账。

模拟并发统一使用 `--batch-size N`，不再硬编码上限 3。已验证的账户上限会约束请求；
平台没有公布上限时，默认 3 只是有来源标记的保守本地策略，可显式提高请求预算，
但工具不会把配置当成已获授权的并发。`scheduling` 保留 requested/effective/platform_limit。
`simulation_slots.py` 在同账户、同研究目录内共享持久占位；接受或结果不明的提交不按时间过期，
只有确认完成或拒绝才释放。不同研究目录及工具外启动的任务不共享这份占位。
恢复可调整并发而不改变研究请求指纹；权限收缩时先取回已接受任务，再核验新启动的请求。

批量初筛可以用 `--setting simulationMode=QUICK --setting visualization=false`。
有训练期和留存期拆分要求时可同时显式设置 `--test-period P1Y`。候选晋级、入架和新提交
需要用 `--setting simulationMode=FULL` 复跑，通过原有门槛后使用 Full 结果的 Alpha ID。

QUICK 初筛可以显式加 `--defer-quick-pnl`，先保存 Alpha ID、训练期汇总指标和检查点，
把日收益及两段训练 Sharpe 留到补读阶段。回执中的 `quick_pnl.pending_labels` 列出待补臂，
`quick_pnl.recover_argv` 给出补读命令。FULL 照常获取训练证据并沿用原有门槛。

```bash
python3 research_cli.py probe quick.spec.json --dry-run
python3 research_cli.py probe quick.spec.json --defer-quick-pnl --batch-path quick-batch.json
python3 research_cli.py probe --recover quick-batch.json
```

补读时省略 `--defer-quick-pnl`。`probe --recover` 只查询已有任务和 Alpha，
把取到的日收益证据写回原批次。

`simulate/resume` 使用 `scheduling.quota` 展示配额观察；完整 probe 批次保留
`scheduling[].quota`，`probe --summary` 的 `summary.quota` 仅保留最近一次观察，不复制调度历史。

每日配额、并发占位与请求限流是不同信息。成功的重复模拟及 multi 的每个成功子模拟都会
计数；Alpha 列表只保留首次创建日，不能靠“今天新增多少 Alpha”重建剩余额度。

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
预检在当前账户和作用域下为每个所需字段搜索一页平台目录，单次搜索最多等待 8 秒，
同组字段合计最多等待 30 秒。只有唯一的精确 ID 命中可证实字段存在。
无精确命中、多个精确命中或搜索出错时，未核验字段保留为 `unknown`。
回执以 `required_fields_only` 记录这次核验，离线所用的完整快照由独立的目录检查维护。


### Quick 批量回测与续跑

```bash
python3 tools/pipeline/batch_quick.py arms.json --out ledgers/quick.jsonl --concurrency 4 --account-slots 5
```

输入可用 JSON 数组或 JSONL，每条包含 `label`、`expr` 和 `settings`。默认使用 `QUICK`，
显式 `settings.simulationMode` 优先。回测结束后由独立 worker 读取指标，回测 worker 可以接下一条。
兼容入口 `tools/pipeline/async_batch.py` 使用同一执行器，默认输出 `ledgers/async.jsonl`，默认 worker 数为 6。

| 参数 | 控制范围 |
| --- | --- |
| `--concurrency` | 本进程的回测 worker 数 |
| `--account-slots` | 同账号、同研究目录内共享的本地槽位预算，默认 5 |
| `--poll-timeout` | 启动回测重试、轮询进度、补读指标分别计时，每阶段默认 420 秒 |
| `--limit` | 本轮处理的候选数，默认 0，表示不限 |

阶段预算包含该阶段的会话获取和重试，用于决定能否继续等待或发起下一次请求。HTTP 超时限制连接和读取间隔。已发出的回测 POST
使用独立的 90 秒网络超时接收 `Location`，因此整个命令的耗时还取决于各阶段和正在执行的请求。
进度和指标轮询优先采用平台给出的 `Retry-After`。响应未给出有效的正值时，使用本地轮询间隔。

续跑时用同一份 arms 和 `--out` 原样重跑。JSONL 保存结果，旁边的 `.state/` 保存已受理任务的
`Location` 和待补取的结果。跨台账只复用可核实为同账号、同规范化表达式和同完整请求设置的结果。
原台账仍支持旧完成记录。旧 `.loc` 仅按 label 保存地址时，工具通过原始 Alpha 回执核验表达式和设置。

| 台账 `status` | 续跑处理 |
| --- | --- |
| `QUEUED` | 尚待发单，检查槽位、配额及已有的 `next_retry_at` 后再启动 |
| `PENDING` | 使用保存的任务地址轮询，或用已取得的 Alpha ID 补读指标，遵守已记录的 `Retry-After` |
| `UNKNOWN_SUBMISSION` | 保留待核对的提交，找到原始 POST 的 `Location` 后登记 |
| `UNVERIFIED_RESULT` | 保留待核实的结果，核对原始任务和表达式及设置 |

退出后，平台仍可能继续执行已受理的回测。查对应候选的最新台账行及 `.state/`，完成情况以逐项状态为准。
自动回收只查看可轮询且无人接管的任务，读到该任务的终态回执后才释放共享槽位。

#### 补登记原始 Location

```bash
python3 tools/pipeline/batch_quick.py arms.json --out ledgers/quick.jsonl \
    --recover-job JOB_KEY --recover-location 'ORIGINAL_BRAIN_LOCATION'
```

`JOB_KEY` 取结果行的 `simulation_job_key`，`ORIGINAL_BRAIN_LOCATION` 填那次 POST 响应的原始
`Location`，保留查询参数。地址必须指向一个具体的回测任务。命令用原账户读取该地址并登记，
随后用原回测命令续跑，核验 Alpha 的表达式和设置。缺少地址的旧 `TIMEOUT` 记录需要先找回原任务身份。

确认原任务已经失败且没有 Alpha ID 时，在同一条登记命令末尾加 `--confirm-original-failure`。
平台回执必须明确返回匹配的任务 ID 和失败终态，工具才会释放槽位。随后续跑会记录该失败，
再处理其余候选。

### Full 回测与生产相关检查

```bash
python3 tools/pipeline/pipeline_pc.py arms.json --out ledgers/full-pc.jsonl --concurrency 4 --pc-max-wait 300
```

使用同一槽位调度器，固定回测模式为 `FULL` 并关闭 `visualization`，回测各阶段预算为 600 秒。
先落账 Alpha ID 和指标，再由独立队列查询满足筛选条件的 Alpha 的生产相关性。`--pc-max-wait` 为逐条
检查的等待预算，默认 300 秒，设为 0 时只跑回测并取指标。

用相同 arms 和台账路径续跑。已核实的 Full 结果直接沿用，`check_pending` 从原 Alpha ID 继续查，
平台 `Retry-After` 会保存为 `next_pc_poll_at`。等待期超过本轮检查预算的候选让出 `--limit` 名额。
`REUSE_UNRESOLVED` 表示旧结果身份待核验，保留原 Alpha ID 供后续查询。

Full 遇到 `UNKNOWN_SUBMISSION` 时，用 Quick 的补登记命令配合原 Full 台账路径。传入同一候选的
label、表达式和完整设置，显式写入 `"simulationMode": "FULL"` 和 `"visualization": false`，再用 Full 命令续跑。

### 原生 Multi 回测

在 `simulate` 或 `probe` 上显式设置 `--multi-size`，每个父任务打包指定数量的 Alpha。
`--dry-run` 展开实际会发送的数组和单条请求。`simulate` 的预览在 `scheduling.multi_requests`，
`probe` 的预览在 `multi_requests`。完整批次或 checkpoint 保存打包设置，恢复时直接使用。

```bash
python3 research_cli.py simulate sample-batch.json --multi-size 4 --batch-size 2 --dry-run
python3 research_cli.py simulate sample-batch.json --multi-size 4 --batch-size 2 --checkpoint multi-run.json
python3 research_cli.py resume multi-run.json
python3 research_cli.py probe sample-batch.spec.json --multi-size 4 --batch-size 2 --dry-run
python3 research_cli.py probe sample-batch.spec.json --multi-size 4 --batch-size 2 --batch-path multi-probe.json
python3 research_cli.py probe --recover multi-probe.json
python3 research_cli.py probe --resume multi-probe.json
python3 research_cli.py probe --from-simulation PARENT_ID
```

BRAIN 要求 `MULTI_SIMULATION` 权限。同一父任务内的 Alpha 类型、标的类型、Region、
Delay 和 Language 必须一致。工具沿用 `probe` 的设置分组，末尾单独一条时发普通模拟。
父任务之间并发，组内子回测顺序运行。
[顾问 Multi 说明](https://platform.worldquantbrain.com/learn/documentation/consultant-information/multi-alpha-simulation)

| 参数和计数 | 原生 Multi 含义 |
| --- | --- |
| `--multi-size` | 每个父任务打包 2–10 条 |
| `--batch-size` | 同时占用的任务槽位，顾问文档的 Multi 父任务上限为 8 |
| 每日配额 | 每条成功子回测单独计 1 次 |

工具在发单前保存父任务和子配置，接受后记录父任务地址。恢复时从地址读取子任务，分别
记录成功、失败及待确认的结果。父任务报错并返回子列表时，已确认成功的子结果仍可入账。
任务身份不明时保留待核对状态。`probe --recover` 只读已登记的平台任务并补账。`probe --resume`
还可继续排队的回测。手里只有父任务 ID 时，用 `probe --from-simulation` 查看父任务和各子任务状态。
`--max-wait-seconds` 在原生 `simulate`、`resume` 和 `probe` 中按调用共享，包含父子任务查询。
[BRAIN 父子任务 API](https://platform.worldquantbrain.com/learn/documentation/brain-api/brain-api#simulations)

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
`--max-wait 0` 对每个 Alpha 发一次请求，使用常规连接和读取超时。

#### 请求超时或没有结果文件时

`--save` 在命令完成后原子写入最终 JSON，不是实时检查点。默认串行检查两个 Alpha，
仅异步等待预算就可能达到 2 × 180 秒，另有认证、登录锁及网络等待；外层 `timeout 240`
直接终止进程时，可能还没有最终文件。退出码 124 只说明外层时间预算耗尽，不能证明平台过载。
认证 POST 使用 60 秒连接和读取超时。通用 BRAIN 请求默认在 60 秒预算内重试，
会根据剩余预算决定是否等待 Retry-After。认证、分页和异步等待仍会增加整条命令的耗时。
登录锁等待到期时，如果其他进程尚未保存可复用会话，CLI 返回 `SESSION_BUSY`。
通用请求的 HTTP 401 触发一次会话刷新，HTTP 403 返回 `ACCESS_DENIED`，沿用当前会话缓存。

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
数据 hash。更换缓存时保留上个版本的校验信息，写入中断后仍可读取原有已验证的数据。旧格式的
缓存完整性为 unknown。台账仍拒绝用缺指标的行覆盖已有指标。

`ledger sync --max-wait-seconds` 默认 300 秒，登录、列表、指标和 PnL 共用预算。
预算耗尽后，已经取回的行仍会写入，回执保留未补齐的 ID 和已获知的建议重试时间。

台账逐行读取 JSONL，跳过损坏的行。追加时会补齐上个写入缺失的换行符，保留原始尾部。
模拟批量记账和 `ledger sync` 每个写入批次在锁内读取一次最新状态，复用写入句柄，
成功的行各自落盘。`ledger sync` 每收齐 10 条回读就保存一批，平台回读在锁外完成。
同步回读若与捕获期间新写入的本地状态冲突，保留该行并报告需要重新核对。

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

## 长程研究队列

把待研究的规格放入研究目录的 `backlog/`。先预览输入，再推进队列。已受理的任务、已确认未开始的臂、结果不明的请求都由 Probe 检查点记录。

```bash
python research_cli.py fmt DRAFT.json
python research_cli.py campaign preview SPEC.json --multi-size 4
python research_cli.py campaign prepare-full QUICK-BATCH.json --out FULL-SPEC.json --limit 8
python research_cli.py campaign status
python research_cli.py campaign run --recover-only
python research_cli.py campaign run --watch --batch-size 4 --multi-size 4 --stream
python research_cli.py campaign run --watch --defer-quick-pnl
```

所有新 FASTEXPR 回测请求默认先经过 `fmt` 的语法树等价检查。`simulate --dry-run` 和 `campaign preview` 的 `payload.regular` 显示实际发送的源码。单条、原生 Multi、兼容 QUICK/FULL 入口都走同一规则。含注释或无法确认改写安全的表达式保留原文。Python 源码和 SPC prompt 保持各自的校验规则。

单独运行 `fmt` 可以审阅排版。在入队前使用 `--write` 才会改写草稿。检查点保留原始规格身份，已受理的任务按原请求回收。历史生成脚本可以用 `alpha_sugar.fs()` 生成原生表达式，正式规格保留展开后的源码。

Campaign 开跑前复用 Probe 的离线扫描，确定的语法错误在联网前拦截，字段和设置仍由账号范围内的在线预检核对。`batch_quick.py` 和 `pipeline_pc.py` 仍是兼容入口。

`prepare-full` 仅读取已保存的 QUICK 训练证据。它按训练 Sharpe 排序，生成带原始 Alpha ID、批次和训练分数的 FULL 草稿。`--limit` 是 FULL 臂的总数上限，对照实验的原对照计入上限。预览并确认草稿后再放入 `backlog/`。生成草稿在本地完成。

| 规格 | 研究证据 |
| --- | --- |
| `track: REGULAR`，也可省略 | Probe 保存请求与训练对照。符合训练门的 FULL 候选冻结选择。 |
| `track: POWER_POOL` 或 `track: ATOM` | 共用 Probe 请求、回测和训练证据。平台分类与资格以实时检查为准。 |
| `settings.language: PYTHON` | 可与上述模拟研究路线组合。用 `lookback`、内联源码或相对规格的 `source_file`。预检只解析源码。 |
| `track: SPC` | 用 SPC 规格加 `runs` 文件列表和可选 `reference`。文件路径相对规格，比较已有输出，保留哈希与逐次对照。 |

研究路线记录在回执和批次中。原生请求类型、语言和 QUICK/FULL 设置仍按各自的平台协议处理。
平台原始分类随结果保存，可用 `alpha get` 或冻结后的验证视图核对。

`run` 默认推进一份当前可运行的规格，返回 JSON 的 `data.state`。需要长期运行时用 `--watch`。队列先处理可回收的检查点。剩余任务均已受理且尚未到轮询时间时，该批次让出本轮。原生 Multi 使用已登记槽位的父任务轮询时间，`status` 显示这些批次的 `next_poll_at`。同一队列只有一个写入者，Probe 按共享槽位和账号能力执行并发。同一账号和研究目录共用平台的模拟 POST 退避时间，退避期间仍可回收已受理任务。新的 Probe 会推迟相关字段预检，等退避期过后再发起。

批量 QUICK 初筛可用 `run --watch --defer-quick-pnl`，让队列在保存训练汇总后继续推进。
规格归档后，仍可按回执的 `quick_pnl.recover_argv` 对原批次执行只读补查。

先用 `--recover-only` 回收已有检查点。需要验证 FULL Regular 候选时启用 `--validate`，或对保存的训练冻结选择单独运行 `promote --selection`。已冻结的选择和终态验证回执可以复用。QUICK、Power Pool 与 ATOM 保留各自研究证据，SPC 结果在本地完成。Alpha 和 SPC 的平台提交仍走各自授权入口。

### 兼容入口

| 工具 | 职责 |
| --- | --- |
| `tools/sim_queue.py` | 消费 `queue/` 中的 Probe spec，保存批次与回执 |
| `tools/campaign_runner.py` | 使用同一 `backlog/` 队列，按旧入口的默认行为冻结并验证 FULL Regular 候选 |
| `tools/atom_runner.py` | 把 `label / expr / settings` 列表交给 `probe`，用可恢复批次保存进度，把全 IS 指标和 checks 写成 `atom-results.jsonl` |
| `tools/atom_screen.py` | 从 ATOM 账本按平台子宇宙检查初筛，同一 Alpha 分数相同时采用最新记录 |
| `tools/spc_client.py` | SPC 表单、提交列表、样例校验和离线输出对比 |

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

两个队列的 `--once` 每次最多处理一份 spec。Campaign 优先处理到期的检查点，再处理未开始的规格。省略该选项时，worker 会持续轮询。
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

Campaign 的 `campaign-freeze-ALPHA_ID.json` 和 `promote-ALPHA_ID.json` 回执也放在 `backlog-results/`。

ATOM 的默认批次路径在 `checkpoints/`，以臂清单内容生成名称。已有同名批次时使用
`--recover` 或 `--resume`。`--recover` 回读已知任务，`--resume` 也可继续已知未提交的臂。
`--dry-run` 可用于新扫描和续跑预览，跳过结果账本写入。
`atom-results.jsonl` 保存带 `metrics_scope: full_is` 的指标与平台检查快照，供子宇宙初筛。
常规训练筛选使用训练期回执，提交资格仍需实时平台 checks。

### 做 SPC 提示词实验

```bash
python tools/spc_client.py recon
python tools/spc_client.py validate SPEC.json
python tools/spc_client.py experiment SPEC.json --run RUN-A.json --run RUN-B.json
python tools/spc_client.py experiment SPEC.json --run RUN-A.json --reference UNIVERSE.json
```

`recon` 和 `validate` 读取实时 SPC 表单。Spec 包含 `name`、`prompt`、`modelFamily`、
`updateFrequency`、`weight` 和 `sampleOutput`。`sampleOutput` 是装有真实模型输出数组的
JSON 字符串。`RUN.json` 文件直接保存模型返回的数组。固定 prompt、模型和信息时点做重复运行，
可在 spec 里记录 `as_of`，以便后续复核。
`recon` 优先读取表单和提交列表，再用剩余的探测预算检查其他端点。跳过的端点会标出原因。

`experiment` 在本地检查每条记录、ISIN 校验位、MIC 格式、重复标的和预测数值。
回执包含输入文件与 prompt 的 SHA-256，以及每个有效运行相对样例的标的重叠率、多空变化、
置信度、期限和收益预测的平均绝对变化。失败的运行也会出现在回执中。退出码为 `0` 时输入通过
本地检查，`2` 表示有输入错误。回执里的哈希用于核对输入版本，模型与时点来自实验人员的记录。

可选的 `UNIVERSE.json` 是本地对照表。地区和行业使用 SPC 指南的分段名称，缺失分类可填 `null`。

```json
[{"isin":"US0378331005","mic":"XNAS","region":"Americas","sector":"Information Technology"}]
```

覆盖按置信度非零的输出行统计。回执保留命中和缺漏，这份统计用于实验比较，领奖资格由平台核算。
[分段定义](https://support.worldquantbrain.com/hc/en-us/articles/44793597008919-What-is-the-segment-coverage-requirement)

SPC 平台提交通过独立的 `submit SPEC.json --yes` 入口，需要明确授权。成功会输出平台条目 ID。
本地校验失败、HTTP 拒绝或无法确认条目 ID 时退出非零。POST 结果不明会停止本批次，重试前先用
`recon` 核对提交列表。
