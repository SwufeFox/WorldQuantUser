# 批量回测与台账复盘

批量 QUICK 先记录训练汇总。台账分析给出样本量、主要失败项和代表候选。
对入围 Alpha 按需补日 PnL，复核训练期逐年 Sharpe，再从原批次起草 FULL。

在仓库根目录执行命令。下文的 `SPEC.json`、`QUICK-BATCH.json`、`FULL-SPEC.json`
和 `QUICK_ALPHA_ID` 是占位符。若有多个研究目录，每条命令使用同一个 `--research-dir PATH`。
先用 `brief` 核对 `data.research_dir`、`data.ledger.path` 和 `data.ledger.exists`。

## 固定工作链：看状态、预览、单队列执行、复盘

```bash
python research_cli.py brief
python research_cli.py campaign status --limit 10
python research_cli.py campaign preview SPEC.json
```

先确认是否已有相同问题的结果、排队规格或受理任务。新的规格写清 `idea`、机制、
每条臂的稳定标签和实际设置。QUICK 初筛的模拟设置写为 `"simulationMode": "QUICK"`。
预览是离线检查，返回解析后的请求，并检查精确历史重复。
查看 `data.would_block`、`data.blocking` 和 `data.gate_blocking`。若已有同表达式、同设置
的结果或可恢复任务，沿 `data.results[].gate` 和 `data.recovery_actions` 回查原证据；
改标签、改输出路径和更换运行器都不能把旧请求变成新实验。

把通过预览的规格放入研究目录的 `backlog/` 后，由一个执行者启动队列。

```bash
python research_cli.py campaign run --watch --defer-quick-pnl --stream
```

启动后保留原规格、臂标签、设置和批次路径。子代理可以准备不同的草稿、读取状态、
分析结果；同一队列和批次的发单、恢复由当前执行者负责。另一个终端可以运行
`campaign status`，不应同时对这个批次运行另一个 Campaign、Probe 恢复循环或兼容运行器。
需要交接时，先确认原写入者已经退出，再用它保存的检查点继续。仅看到锁文件不能判断谁仍在运行。

需要优先启动指定对照时，可加 `--prefer control.json`。参数填写 `backlog/` 中的文件名，
重复指定时按给定顺序安排尚未启动的 BRAIN 规格。恢复任务仍先推进。

### 看见一个状态，只选择它对应的下一步

`ok: true` 表示命令被正常处理。是否取得结果，要读 `execution`、任务 ID 和证据状态。
`campaign status` 的 `data.jobs` 给出规格、原批次、执行计数和下一步；列表截断时同时看 `omitted`。
已有批次优先读 `diagnosis.state`、`diagnosis.reason` 和 `diagnosis.next_action.argv`。
这些从当前保存的批次推导；尚无批次的新规格显示 `state: queued`。
具体命令由回执给出，仍由当前执行者推进。
Probe 的 `next_actions[].argv` 同样带实际 `--research-dir`；执行时保留整个参数数组，
不要手工重建批次路径或改用当前 shell 的默认目录。CLI 子命令是单数 `alpha`。
等待参数统一接受 `--max-wait-seconds` 和兼容别名 `--max-wait`；默认值、是否允许 0
及按整次调用／逐任务计算的预算含义，以对应 `help COMMAND` 为准。

缩略回执的 `truncated: true` 表示有证据被省略，逐列表的 `*_total` / `*_omitted`
给出数量。`summary.full_evidence.available=true` 时，`path` 指向已保存的完整检查点；
dry-run/check 不会把计划路径冒充本次已保存的证据。

| 看到的状态或现象 | 下一步 | 保留什么 |
| --- | --- | --- |
| `queued` / `queued_work`，尚未受理 | 交给现有 `campaign run --watch` 推进；没有执行者时才启动一个 | 原规格、标签、设置与批次路径 |
| `gate_blocking` 或事件 `matching_request` | 阅读 gate 的引用，或事件的 `owner_batch` 与恢复命令，推进原检查点 | 精确请求身份；等待次数不能使重复请求获得新身份 |
| `batch_identity_mismatch` | 对照原规格与检查点；恢复原规格，或明确另建一个不同实验 | 原请求指纹；不靠改名绕过身份校验 |
| `preflight_backoff` / `preflight_deferred` | 看 `retry_at`、失败次数和未核验字段，由队列到期续跑 | 已成功的精确字段证据及原有效期；不连续提前恢复 |
| `platform_pending`，或有 `next_poll_at` | 由原执行者按时间恢复；手动接管时使用原批次 | 平台任务 ID、URL 和受理记录；本地超时不取消远端任务 |
| `batch_locked` / `runner_locked` | 读取锁冲突回执，确认现有写入者；交接后再继续 | `lock_path` 与观察到的 owner；不并行启动第二个写入者 |
| `shared_slots_occupied` | 从原检查点回读平台终态，再由恢复路径更新 reservation | 受理身份；任务年龄和空指标都不能证明可以释放槽位 |
| `submission_cooldown` | 等保存的发单退避时间；原执行者仍可恢复已受理任务 | 原检查点与退避时间 |
| 模拟 `UNKNOWN_SUBMISSION` / `submission_unknown` | 核对原 POST 的任务身份，读取返回的恢复指引 | 原提交意图；在身份不明时不再次 POST |
| 最终 Alpha 提交 `UNCERTAIN` / `UNKNOWN` | 运行 `alpha submit ALPHA_ID --max-wait 30` 只读检查，查看 `submission_recovery`；详见下文 | 最终提交与模拟状态分开；`UNSUBMITTED` 本身不证明此前 POST 未发生 |
| `terminal_failure`，已确认终态 `ERROR` / `FAIL` / `FAILED` | 复盘错误及输入证据；修正研究构造时另建规格 | 失败臂、idea／机制和原始错误；执行失败不自动证明经济假设已被证伪 |
| 晋级 `metadata_required`，`buffered=false` | 给 `next_action.argv` 补齐 `required_options` 中的候选说明参数，沿原冻结选择运行 `promote` | 已完成的验证回执和 selection；不重新模拟、不循环重试空元数据 |
| QUICK 训练已记录，日 PnL 尚欠 | 从 `data.archived_quick_pnl.jobs[].batch` 找原批次，按候选 ID 补读 | 已保存的训练汇总；不为补 PnL 重跑模拟 |

手动恢复只由当前执行者使用。需要继续已确认排队和已受理的臂时用
`probe --resume BATCH.json`；只读取已有平台任务和补证据时用 `probe --recover BATCH.json`。
`campaign run --recover-only` 用于集中补回已有任务，不启动未受理的臂。
原生 Multi 的监视轮次会保存平台的 `Retry-After`，未到期时让出队列；直接 Probe 则按本次预算等待。

最终 Alpha 提交沿用 `alpha submit`。无 `--yes` 时只读检查待核实的提交，不改提交日志。
若读到已完成且明确拒绝的提交结果，回执会给出 `REJECTED` 和失败项。针对已有的未决提交，
在原有提交授权下用同一命令加 `--yes` 可持久化这次核实结果，该轮不会再 POST。
只有能证明在 POST 前结束的准备失败才记为 `NOT_SENT`；传输不明、空响应、404 或仍在等待的
结果保留不确定性。`post_attempted` 描述本次调用，不能单凭本次为 `false` 推断上一轮没有发出请求。

同一队列的在途或待核实请求会占用其精确请求身份，后来的重复规格等待原检查点。
删除或归档原规格不会释放这个身份；只有原批次的终态证据能解除占用。
同一账号和研究目录已确认占满槽位时，启动流程先暂缓新请求的字段和算子预检。监视器在同一轮复用刚取得的满槽证据，暂缓同账号、同并发设置且尚未启动的 native Multi 批次。每轮重新核对，检查点的回收仍可继续。
要单独查看预检结果，可以运行 `probe --check`。
平台槽位、限流和结果可见时间仍由账号与平台决定。`--stream` 事件写到 stderr。

队列通常把批次和回执保存为 `backlog-results/<spec-stem>-batch.json` 和
`backlog-results/<spec-stem>-result.json`。取实际路径时优先用保存的 Probe 回执
`data.path` 或状态中的 `batch`。待补列表只列仍欠证据的归档批次。

### 区分等待与零进展

比较相邻两次状态：新增受理多少条、新增终态多少条、新增有效指标／检查证据多少条，
以及最后一次有实质变化的时间。正常等待应能指向原任务、阻塞原因和下次轮询／重试时间。
进程仍在运行、日志继续增长或又得到一份 `deferred` 回执，都不能替代这些变化。

`jobs[].last_attempt` 保存最近一次执行观察，不代表当前平台状态。它的 `observed_at`、
`detail`、`diagnosis` 和 `progress` 用于复盘；`matches_current_progress` 说明这份观察是否
仍匹配当前批次。`progress` 包含 `changed`、`unchanged_rounds`、`last_progress_at` 和 `stalled`。
连续至少三轮已结束的尝试没有实质变化，且没有未来轮询时间解释等待时，会标记 `stalled`。
这个标记只要求排查，不会解锁、重发或宣判远端失败。单纯保存文件、更新时间和追加调度日志不算进展。
`last_runner_event` 同样是上一次 runner 冲突的记录，不是现在仍有活进程的证明。

交接或报告时用固定格式：**原批次；本轮新增受理／终态／记录；当前阻塞；下一步与时间**。
如果连续观察没有变化，就明确报告“无新进展”，核对对应的状态行，不再复制原命令循环。
OMP 正在等子代理属于宿主编排状态，应另报代理任务和最后一条状态；仓库的 Campaign
回执不能用来判断 VPS 已死，也不能证明宿主代理仍在推进。

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

### 失败也留种子，缺失 idea 明确报告

已确认终态的执行失败写入同一个 `alpha-registry.jsonl`，保留表达式、设置、原始错误和来源。
这类记录使用稳定的 `failure:` 键，`execution_outcome=failed`、`alpha_id=null`；若原请求带有
Alpha ID，它只作为 `related_alpha_id` 保留。记录不编造指标或检查，也不因执行错误自动
标记经济假设为 `falsified` / `rejected`。仍在等待的任务或一次结果读取失败不因此成为终态。

```bash
python research_cli.py ledger list --idea "研究问题" --format markdown
python research_cli.py ledger list --expression "表达式片段" --format markdown
```

复盘失败时不用最低 Sharpe/Fitness 过滤，否则无指标的记录会被筛掉。
Probe 优先保存臂上的 `idea`，没有时用规格的 `idea`，不从文件名或标签猜经济解释。
`provenance` 保留 `run_id`、`request_fingerprint`、`source_spec` 和 `arm_label`，可回查原实验。
`recording` 摘要报告 `recorded_failures`、`missing_idea_count` 和最多二十个
`missing_idea_labels`；缺失时应补齐下一份规格的问题定义，保留原始缺失事实。

这里的“留种子”是保存研究记忆，包括失败尝试。已验证的提交候选另有 buffer/seed store，
由下文的 `promote` 保存候选说明；不用为 QUICK 和 ERROR 再建一套人工留种子流程。

`simulate --record` 整批失败仍返回 `SIMULATION_FAILED`，其 `error.details.recording`、
`error.details.summary` 和检查点引用可用于找回已记录的失败。Probe 的 `execution.terminal`
单列无 Alpha ID 且已确认终态的臂数；`probe.complete` 要求所需训练证据齐全。
记录了错误不等于拿到了有效训练证据，也不能算一次成功出分。

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
保留其来源与原对照，再预览、确认和入队。FULL 完成后使用其新 Alpha ID。
对照、逐年及成本证据先于平台相关性核验。

## 选定 FULL 候选后，一次补齐说明并晋级

Campaign 为满足训练门槛的 FULL Regular 候选保存冻结选择，在 `training-frozen` 事件中
返回 `selection_path`。沿用该路径和候选 ID，在原执行者结束该批次后运行一次 `promote`。
以下名称、标签和说明是格式示例，实际填写该候选的机制与证据：

```bash
python research_cli.py promote FULL_ALPHA_ID --selection SELECTION.json \
  --name "分析师预期修正" --category ANALYST --tags analyst revision \
  --description "研究假设是分析师预期修正的信息逐步进入价格；使用事先指定的修正字段、地区和时间窗，避免把数值缺失或覆盖变化当成信号。训练选择已经冻结，留出期只用于核对这一选择。复盘时需要结合原对照、成本、分期表现和相关性检查解释结果，并保留不支持该假设的证据。"
```

`name` 非空；`category` 取 `ANALYST`、`FUNDAMENTAL`、`PRICE_VOLUME`、`SENTIMENT` 或
`VOLUME`；`tags` 至少一个；`description` 去掉首尾空白后至少 100 个字符。
说明应交代真实机制、输入、对照与证据，缺失信息明确写缺失，不为凑长度编造发现。
`promote` 使用原冻结选择做验证、核对平台检查，再保存 buffer 和 seed metadata。
无须再手动执行一次 `buffer add`，也不手改 `seeds.jsonl`。

若验证和检查通过，但说明未齐，回执为 `outcome=metadata_required`、`buffered=false`，
`metadata_required` 列出缺项，`next_action` 给出原冻结选择的命令和 `required_options`。
补齐后运行一次该命令。`campaign run --validate` 也可能停在这个状态；无说明的 `--dry-run`
同样不会报告 `would_buffer`。若使用自定义 `--seeds PATH`，晋级和后续提交沿用同一路径。

本地 seed metadata 和平台 Alpha 的名称、分类、标签是两份不同记录；本地入库不会自动
修改远端属性。真正提交仍走既有授权入口，并按回执补齐平台要求的属性。
已有未决提交的只读核实和持久化恢复不因缺少新 seed metadata 而阻塞。

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
