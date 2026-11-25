# 批量回测与台账复盘

批量 QUICK 先记录训练汇总。台账分析给出样本量、主要失败项和代表候选。
对入围 Alpha 按需补日 PnL，复核训练期逐年 Sharpe，再从原批次起草 FULL。

在仓库根目录执行命令。下文的 `SPEC.json`、`QUICK-BATCH.json`、`FULL-SPEC.json`
和 `QUICK_ALPHA_ID` 是占位符。若有多个研究目录，每条命令使用同一个 `--research-dir PATH`。
先用 `brief` 核对 `data.research_dir`、`data.ledger.path` 和 `data.ledger.exists`。

## 预览、入队和恢复

```bash
python research_cli.py campaign preview SPEC.json
python research_cli.py campaign status --limit 10
```

规格的模拟设置写为 `"simulationMode": "QUICK"`。预览返回解析后的请求，并检查精确历史重复。
查看 `data.would_block`、`data.blocking` 和 `data.gate_blocking`。若已有同表达式、同设置
的结果或可恢复任务，沿 `data.results[].gate` 和 `data.recovery_actions` 回查证据。

把通过预览的规格放入研究目录的 `backlog/` 后启动队列。

```bash
python research_cli.py campaign run --watch --defer-quick-pnl --stream
```

需要优先启动指定对照时，可加 `--prefer control.json`。参数填写 `backlog/` 中的文件名，
重复指定时按给定顺序安排尚未启动的 BRAIN 规格。恢复任务仍先推进。

| 场景 | 读哪个状态 |
| --- | --- |
| 首次推进 | `campaign status` 的 `data.jobs` 显示规格、批次和已保存的执行状态。首次启动使用配置的等待预算。 |
| 回合结束时仍在等平台 | `--watch` 每轮只读一次新受理或已到期的 native Multi 父任务，将平台的 `Retry-After` 保存到检查点后推进队列。直接运行 `probe` 时默认按指定预算等待。 |
| 有 `next_poll_at` | 这是检查点记录的下次轮询时间。监视器按时间推进，原生 Multi 的时间来自父任务。 |
| 进程或连接中断 | 先读原批次。用 `campaign run --recover-only` 补回已受理的结果。需要继续未启动的臂时，再让普通队列推进。 |
| QUICK 训练已记录，日 PnL 尚欠 | `data.archived_quick_pnl` 有待补批次、臂数和不可读批次数。用其中的 `jobs[].batch` 定位原批次。 |

同一队列的在途或待核实请求会占用其精确请求身份，后来的重复规格等待原检查点。
同一账号和研究目录已确认占满槽位时，启动流程先暂缓新请求的字段和算子预检。监视器在同一轮复用刚取得的满槽证据，暂缓同账号、同并发设置且尚未启动的 native Multi 批次。每轮重新核对，检查点的回收仍可继续。
要单独查看预检结果，可以运行 `probe --check`。
平台槽位、限流和结果可见时间仍由账号与平台决定。若 `campaign status` 的列表被截断，
看 `omitted` 和汇总计数。运行中的状态可从另一终端查询，`--stream` 事件写到 stderr。

队列通常把批次和回执保存为 `backlog-results/<spec-stem>-batch.json` 和
`backlog-results/<spec-stem>-result.json`。取实际路径时优先用保存的 Probe 回执
`data.path` 或状态中的 `batch`。待补列表只列仍欠证据的归档批次。

## 用台账找候选

```bash
python research_cli.py ledger analyze --group-by scope --limit 8
python research_cli.py ledger analyze --region ASI --delay 1 --mode QUICK \
  --group-by mechanism --cost-bps 10 --max-turnover 0.20 --limit 8
```

先用 `scope` 核对覆盖，再用相同地区、股票池、delay 和模式比较机制或批次。
可以用 `--universe` 缩到同一股票池。需要复盘近期记录时，用 `--since YYYY-MM-DD`
限定台账记录日期。向 DeepSeek 交接时，给上述摘要、筛选参数和要验证的研究问题。

| 报告字段 | 怎么读 |
| --- | --- |
| `data.counts` | `entries` 按最新台账键计数，`distinct_alpha_ids` 按去重 ID 计数。机制数按已有的机制标签计数。 |
| `data.funnel` | 每个台账键只归入一个阶段。查缺训练窗口、缺持仓规模、成本吃完 margin 和已知硬失败各有多少。 |
| `data.failure_drivers` | 同一行可计入多个失败项。关注 `sole_failure_entries` 和 `otherwise_screened_entries`，区分普遍失败和只差一项的线索。 |
| `data.groups` | 用组内记录数、可用训练证据数和过筛数判断支持量。组别排序优先显示过筛数多的组。 |
| `data.representatives` | 先按 Alpha ID 去重，再按机制标签取代表。无标签共用一个桶。排序以训练 Sharpe 为先，随后用扣成本 margin、记录时间和键打破并列。 |
| `data.single_failure_leads` | 这一行通过训练初筛，已存检查里只有一项硬失败，且其余已存检查均有结论。用它查失败原因。 |

默认初筛要求有检查记录且无已知硬失败，训练 Sharpe 达到下限，多空持仓数为正，
扣成本后仍有正的训练 margin。换手上限为可选参数。待定检查列在 `unresolved_checks`。
同一 Alpha ID 的诊断标记对它的所有台账键生效。

### 检查缺失、历史证据与研究标签

检查名称、正式 `result`、实际 `value`、阈值 `limit` 和说明随载荷保留。
`PASS` 和非阻断的 `WARNING` 都是已有结论；`FAIL` 阻断推进；`ERROR` 也阻断，
但仍是未解决状态；`PENDING`、未知结果及缺失的必需检查不能当成通过。
数值与阈值的比较是诊断信息，不覆盖平台的正式 `result`。
端点是否完成、返回的检查是否都有结论、研究或提交门槛是否满足，分别判断。

同名检查重复出现或夹杂无名、格式错误的记录时，解析器保留原始记录并标记
`source=client_parser`、`client_parse_error` 和空的 `result`，保持待确认。
无名记录集中在 `__CLIENT_CHECK_PARSE_ERROR__` 下；这是客户端诊断，不是平台检查名称，
不会变成后续载荷必须提供的检查。正常刷新可以解除该状态。

每条台账的 `checks` 只表示当前载荷，`checks_state` 描述其覆盖情况：

| `checks_state` | 含义 |
| --- | --- |
| `observed` | 当前有检查，未发现相对于上一份已保存的同请求快照的缺项；仍须看每项结果，不能直接当作全清。 |
| `missing` | 当前没有检查，也没有可匹配的历史快照。 |
| `stale` | 当前没有检查；同表达式、同设置的上一份检查保存在 `last_check_snapshot`，供追溯。 |
| `partial` | 当前检查比上一份同请求快照少了名称；缺项仍待确认，历史快照单独保留。 |

`last_check_snapshot` 最多保留一份检查及其来源、模式和观察时间，不补进当前
`checks`，也不跨表达式或设置（包括 QUICK/FULL）复用。完整事件历史仍在 JSONL 中。
自动 `candidate` 不接受缺失、待定或部分返回的检查；通过 `ledger label --verdict candidate`
或 `ledger record --verdict candidate` 手动标记的条目带 `verdict_manual=true`，表示研究判断。
候选标签本身不证明平台通过，更不证明可以提交。

### 时间字段怎么用于复盘

| 字段 | 含义 |
| --- | --- |
| `recorded_at` | 这一次台账事件的时间，改标签和去重也会更新。`ledger analyze --since` 仍按它筛选近期事件。 |
| `first_recorded_at` | 该台账键最早可追溯的事件时间。 |
| `measurement_observed_at` | 当前指标载荷首次被观察到的时间；指标及切分载荷不变的重复读取及纯改标签不移动它。 |
| `checks_observed_at` | 当前检查载荷首次被观察到的时间；当前缺失时为空，旧时间留在历史快照中。 |
| `first_measurement_observed_at` / `first_checks_observed_at` | 该键最早有时间依据的指标／检查观察，不代表当前数值当时已知。 |

这些是本地证据的观察时间，不是平台实际执行时间。旧台账若只有改标签后的行，
无法补出原始测量时间，相关字段保留为空。做模型时间切分时，应按当时可见的事件
重建特征与标签；不能把最新标签配到最早时间，也不能把缺失检查编码成“全清”。
当前代码不替外部实验选择训练集或证明预测能力。

### 训练初筛参数

| 参数或单位 | 取值 |
| --- | --- |
| 默认 `--min-train-sharpe` | 1.0 |
| 默认 `--cost-bps` | 10 bps，每成交名义金额的单边成本 |
| `net_margin_bps` | `train_margin × 10000 − cost_bps`，要求大于 0 |
| 示例 `--max-turnover 0.20` | 20% 的训练换手上限 |
| `--limit` | 每个输出列表 1–100 行，默认 12。总体计数覆盖选中范围。 |

`representatives[].batch` 是批次名。把摘要里的 `alpha_id` 对回保存的回执或批次。
研究同一机制时，可以先请模型指出最集中的失败项、证据覆盖和一条能区分解释的对照。
候选的经济独立性要用持仓或收益核对。平台检查和提交资格以相应的实时验证为准。

## 补日 PnL、看逐年、起草 FULL

```bash
python research_cli.py probe --recover QUICK-BATCH.json \
  --pnl-alpha-id QUICK_ALPHA_ID --summary
python research_cli.py ledger analyze --region ASI --delay 1 --mode QUICK \
  --group-by mechanism --cost-bps 10 --max-turnover 0.20 --limit 8 --yearly
```

`--pnl-alpha-id` 可重复，用于同一批次的已保存 QUICK Alpha。命令补读已选 ID 的日 PnL，
并可核对已受理任务。重复使用原来的分析范围和阈值可复核同一份候选名单。

`--yearly` 只检查报告已选中的代表。它用本地有完整性凭据的日 PnL，截取
`train_start <= date < test_start`。排名仍由初筛口径决定。读代表条目各自的 `yearly_train`。

| 年度证据 | 口径 |
| --- | --- |
| 可计分年度 | 完整日历年位于训练窗口内，有 1 月和 12 月的有效观测，且有至少 145 个有效日值。 |
| `pass` | 至少有 3 个完整训练年度，且每个完整训练年度都可计分。最差年 Sharpe 大于 0，最近 3 个训练年均值至少 1.0，且没有连续 3 年均低于 0.3。 |
| `fail` | 年度证据齐备，结果触发了上述门槛。 |
| `unavailable` / `insufficient_full_train_years` | 前者检查本地缓存和完整性凭据，后者看年份窗口、观测量和年度可计分性。两者都保留为未知。 |

要集中补归档批次时，可用 `campaign recover-pnl --limit 4 --stream`。这个上限按批次计，
每批会尝试补该批欠缺的日 PnL。通常先按候选 ID 补，避免把下载量扩大到全库。

```bash
python research_cli.py campaign prepare-full QUICK-BATCH.json --out FULL-SPEC.json --limit 3
python research_cli.py campaign preview FULL-SPEC.json
```

`prepare-full` 按原批次的训练 Sharpe 排序，筛掉缺失训练证据及非正训练指标的候选。
对照实验会保留原对照，并计入上限。检查 `data.quick_source.arms[].alpha_id` 是否与
逐年复核后选定的 QUICK ID 一致。该命令按批次取臂。若名单有差异，依据已选的臂另行起草规格，
保留其来源与原对照，再预览、确认和入队。FULL 完成后用其新 Alpha ID 冻结和验证。对照、逐年及成本证据先于平台相关性核验。

## 索引与故障定位

原始记录在 `alpha-registry.jsonl`。同目录的 `alpha-registry.jsonl.index.sqlite3`
是可重建的 SQLite 投影。首次建索引、源被替换或截断、索引版本变化时会重放台账。
正常追加只重放新字节。索引同步需要台账锁，实际报告读取固定的 SQLite 快照。
大库冷建安排在队列可以暂缓写入的时段。全库报告耗时随覆盖行数增长。
索引版本 4 和直接 JSONL 读取共用证据折叠逻辑，旧记录可在读取时补出有依据的来源与时间；
原始 JSONL 不会被改写。升级后的首次读取会重建旧版本索引。

| 看到的情况 | 核对方式 |
| --- | --- |
| `NOT_FOUND` | 核对 `brief` 的研究目录和台账路径。 |
| `the local ledger projection is unavailable` | 核对读写权限、磁盘空间与台账锁。若要排查旧投影，用一份完整 JSONL 副本，在可写目录执行 `ledger analyze --registry COPY.jsonl` 独立建索引。 |
| `data.counts.entries = 0` | 查询成功，当前筛选范围没有记录。先用 `--group-by scope` 查看覆盖。 |
| `archived_quick_pnl.unreadable_batches` 大于 0 | 核对原检查点及回执引用。归档摘要按文件版本复用，源恢复可读后会重新检查。 |

台账、检查点、缓存、回执和研究报告存放在研究目录。版本库保留工具、测试与操作说明。
