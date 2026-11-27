# 状态机实验流程与台账复盘（v4）

v4 版本采用**常驻单写入者 Daemon + 非阻塞 CLI + SQLite WAL/Delta-Parquet 三层存储 + 持久化门禁状态机**替代旧版文件队列与阻塞式长命令。运行状态、槽位日志与 BRAIN Labs 工单持久化在 `<research_dir>/workspace.db`（SQLite WAL），日度 PnL 主存储使用 `<research_dir>/pnl/` 下的不可变 `base-*.parquet` 和 `deltas/*.parquet`，由 `meta.sqlite` 发布生效的版本，不可变冻结凭证保留在 `runs/<run_id>-selection.json`，台账证据与失败记录持久化在 `<research_dir>/alpha-registry.jsonl`。

> 在仓库根目录执行命令；若有多个研究目录，请始终传递同一个 `--research-dir PATH`。先运行 `python research_cli.py brief` 核对 `data.research_dir` 与工作区状态；当遇到连续逻辑合理但模拟异常、字段分布/覆盖不确定或多字段共线性检验需求时，通过 `python research_cli.py labs open`（或在门禁态执行 `act RUN_ID request-labs`）向人类协作者委派 BRAIN Labs 诊断工单，停止盲试。

## 1. 常驻 Daemon 与短命令工作链

```bash
python research_cli.py daemon start
python research_cli.py brief
python research_cli.py propose SPEC.json
python research_cli.py wait --max-wait 30
python research_cli.py inbox
python research_cli.py inspect RUN_ID
```

- `propose SPEC.json` 进行离线 AST / Python `@alpha(data=[...])` / 重复请求指纹校验，并立即创建处于 `AUTO_PREFLIGHT` 的运行文档返回，**不等待模拟或网络请求**；若 daemon 尚未启动且未在测试禁用模式下，CLI 会按需拉起或使用 `daemon start`。
- 后台 worker 只在读取快照、记录回执与合并写回时持有 `daemon/state.lock`。平台 I/O 和本地 SPC 计算在锁外运行，CLI 可同时响应 `propose`、`cancel` 和配置更新。调度优先级如下。
  1. 主动取消 (`AUTO_CANCELLING`) 与提交核实 (`AUTO_SUBMITTING`)
  2. 入架写描述 (`AUTO_BUFFER`)
  3. 进行中/待启动 FULL 验证 (`AUTO_FULL_VALIDATE`)
  4. 进行中 QUICK (`AUTO_QUICK_SIM`)、PnL 补水 (`AUTO_HYDRATE_PNL`)、预检 (`AUTO_PREFLIGHT`)、未发送新 QUICK (`AUTO_QUICK_SIM`)
  5. 本地 SPC 对比 (`AUTO_SPC`)，不需要 BRAIN 会话。
- 所有需要智能体做出研究/风控决断的节点均自动暂停在 `GATE_*` 态，通过 `wait` 的条件变量即时唤醒或 `inbox` 查询。
- 调度配额护栏可通过 `python research_cli.py daemon config --quota-floor 20 --concurrency 8` 配置：`quota_floor` 仅暂停未启动的新 QUICK 臂；FULL 验证与已受理任务回读仅在平台 `blocked=true` 时暂停。

## 2. 四门禁状态机全图与转换表

```mermaid
stateDiagram-v2
    [*] --> AUTO_PREFLIGHT: propose
    [*] --> AUTO_SPC: propose SPC
    AUTO_SPC --> COMPLETED: local comparison complete
    GATE_PREFLIGHT_BLOCKED --> GATE_LABS: act request-labs
    GATE_TRIAGE --> GATE_LABS: act request-labs
    GATE_APPROVE_FULL --> GATE_LABS: act request-labs
    AUTO_PREFLIGHT --> GATE_PREFLIGHT_BLOCKED: preflight failed or 3x unchanged defers
    GATE_PREFLIGHT_BLOCKED --> AUTO_PREFLIGHT: act retry-preflight
    GATE_PREFLIGHT_BLOCKED --> DROPPED: act drop
    AUTO_PREFLIGHT --> AUTO_QUICK_SIM: preflight ok
    AUTO_QUICK_SIM --> FAILED: all arms failed
    AUTO_QUICK_SIM --> GATE_TRIAGE: quick complete (train-only visible)
    GATE_TRIAGE --> AUTO_HYDRATE_PNL: act hydrate-pnl
    GATE_TRIAGE --> ARCHIVED: act drop --verdict ...
    AUTO_HYDRATE_PNL --> GATE_APPROVE_FULL: yearly train hydrated
    GATE_APPROVE_FULL --> GATE_TRIAGE: act back-to-triage
    GATE_APPROVE_FULL --> ARCHIVED: act drop --verdict ...
    GATE_APPROVE_FULL --> AUTO_FULL_VALIDATE: act approve-full (freezes selection, opens test view)
    AUTO_FULL_VALIDATE --> REJECTED_AT_VALIDATION: failed checks, self-correlation, test Sharpe or Power Pool budget
    AUTO_FULL_VALIDATE --> GATE_DESCRIBE: validation passed
    GATE_DESCRIBE --> AUTO_BUFFER: act describe --name --category --tags --description
    GATE_DESCRIBE --> ARCHIVED: act drop --verdict ...
    AUTO_BUFFER --> GATE_CONFIRM_SUBMIT: shelved & seed stored
    GATE_CONFIRM_SUBMIT --> SHELVED: act hold
    GATE_CONFIRM_SUBMIT --> AUTO_SUBMITTING: act submit --yes
    AUTO_SUBMITTING --> SUBMITTED: confirmed
    AUTO_SUBMITTING --> REJECTED_AT_SUBMIT: rejected checks
```

| 状态 | 类型 | 触发条件 / 协议保证 | 下一步 CLI |
| --- | --- | --- | --- |
| `AUTO_PREFLIGHT` | 自动 | 账本查重 + 在线字段/算子预检；连续 3 轮无进展且无 `Retry-After` 触发熔断 | 等待或 `cancel RUN_ID` |
| `GATE_PREFLIGHT_BLOCKED` | 门禁 | 预检报错或退避熔断 | `act RUN_ID retry-preflight` 或 `act RUN_ID drop --reason ...` |
| `AUTO_QUICK_SIM` | 自动 | 执行 QUICK（遵守 `quota_floor`），将已完成/终态失败写入 `alpha-registry.jsonl` | 等待 `wait` / `inbox` |
| `GATE_TRIAGE` | 门禁 | 仅暴露训练期指标；`inspect` 强制隐藏测试期及验证字段 | `act RUN_ID hydrate-pnl --top 3`（或 `--arms a,b`）或 `drop` |
| `AUTO_HYDRATE_PNL` | 自动 | 对选中候选拉取日频 PnL 并评估训练窗口逐年 Sharpe | 等待 `wait` |
| `GATE_APPROVE_FULL` | 门禁 | **测试窗口仍关闭**。只能批准一条有明确训练期逐年通过证据的 arm。 | `act RUN_ID approve-full [--arms LABEL]` / `back-to-triage` / `drop` |
| `AUTO_FULL_VALIDATE` | 自动 | 按冻结身份跑一条 FULL，要求有效正值 test Sharpe、完整检查与通过的自相关证据。成本审计覆盖 5/10/20 bps，数据缺失时结果为空。 | `inspect RUN_ID` 查看含 `train+test` 与 `cost_audit` 完整结果 |
| `GATE_DESCRIBE` | 门禁 | 验证通过；等待填写候选元数据与官方三段式模板说明 | `act RUN_ID describe --name ... --category ... --tags ... --description ...` |
| `AUTO_BUFFER` | 自动 | `set_alpha_metadata` 写入平台并回读校验一致，再写入 `seeds.jsonl` 与 `submit-buffer.jsonl` | `shelf list` / `shelf plan` |
| `GATE_CONFIRM_SUBMIT` | 门禁 | 候选已安全入架；**可选择直接提交或暂存候补** | `act RUN_ID hold`（转 `SHELVED`）或 `act RUN_ID submit --yes`（需 `BRAIN_ALLOW_AGENT_SUBMIT=true`） |
| `GATE_LABS` | 门禁 | 等待人类协作者回传汇总统计。工单处于 `OPEN` 时，拦截同 Region/Universe/Delay 且字段重叠的提案。 | `labs show TICKET_ID` / `labs resolve TICKET_ID --summary ...` |
| `AUTO_CANCELLING` | 自动 | 等待平台确认取消、已不存在或进入终态。有未决平台回执时继续退避。 | `inspect RUN_ID` |
| `AUTO_SPC` | 自动 | 对指定本地结果文件运行 SPC 对比。 | `wait --run-id RUN_ID` |
| `AUTO_SUBMITTING` | 自动 | 持久化发送 `POST /alphas/{id}/submit`；遇 `UNCERTAIN` 仅走只读回查，绝不重发 `POST` | 等待终态 `SUBMITTED` 或 `REJECTED_AT_SUBMIT` |

## 3. 协议级训练/测试隔离（Train/Test Firewall）

- `public_run_view`（被 `inspect`、`inbox`、`wait`、`propose`、`act` 共用）在 `PRE_FREEZE_STATES`（`AUTO_PREFLIGHT` 至 `GATE_APPROVE_FULL`）或 `selection_frozen == false` 时，协议层自动剔除行内的 `test`、`test_sharpe`、`test_fitness`、`test_turnover`、`retention`、`checks`、`cost_audit`、`validation` 与 `out_of_sample.test`。
- 提案中的所有 arm 必须使用相同的有效模拟设置。不同中性化、衰减或范围配置分开提案。QUICK 的训练指标只取显式训练分段。
- 只有在 `act RUN_ID approve-full` 不可逆地冻结 `runs/<run_id>-selection.json` 并置 `selection_frozen = true` 后，`inspect RUN_ID` 才打开 `test_view_open: true` 并暴露测试期留存、平台实时 checks 与成本净 Sharpe。批准动作和显式查看测试窗口写入 `views.jsonl`。

## 4. Power Pool 官方说明模板与 `GATE_DESCRIBE`

提交元数据在 `GATE_DESCRIBE` 阶段即做严格离线校验，不通过则拒绝进入 `AUTO_BUFFER`：
- `name` 非空；`category` 为 `ANALYST`、`FUNDAMENTAL`、`PRICE_VOLUME`、`SENTIMENT` 或 `VOLUME`；`tags` 至少一项。
- `description` 去掉首尾空白后至少 100 字符，且逐字包含三个标题行：
  - `Idea:`
  - `Rationale for data used:`
  - `Rationale for operators used:`
- 每个标题独占一行后接至少一个 `- ` 列表项；旧版 `IDEA:` / `RATIONALE:` 会被拒绝。`AUTO_BUFFER` 会调用 `set_alpha_metadata` 写入平台并回读 `regular.description` 严格一致后才落盘 `seeds.jsonl` 和 `submit-buffer.jsonl`。

## 5. 台账分析、失败留痕与旧版工作区迁移

```bash
python research_cli.py ledger analyze --group-by scope --limit 8
python research_cli.py ledger analyze --region ASI --delay 1 --mode QUICK --group-by mechanism --cost-bps 10 --max-turnover 0.20 --limit 8 --yearly
```

- **失败记录**：终态失败写入 `alpha-registry.jsonl`，带 `failure:` 稳定键与 `execution_outcome=failed`，不会把技术报错自动当成假设被证伪（`falsified`）。
- **检查覆盖态 (`checks_state`)**：区分 `observed`、`missing`、`stale`、`partial`；缺失和 `PENDING` 均保持未知，不能视为通过。
- **训练期参数与逐年口径**：默认 `--min-train-sharpe 1.0`、`--cost-bps 10`；逐年 `pass` 要求至少 3 个完整训练日历年（含 1 月和 12 月观测且 ≥145 个有效日值），最差年 > 0，近三年均值 ≥ 1.0，无连续三年 < 0.3。
- **候选架独立性规划**：入架候选可随时通过 `shelf list` 和 `shelf plan` 基于训练期日频 PnL 计算最大权重独立子集并规划提交顺序。
- **v3 工作区迁移**：旧目录存在 `backlog/` 或 `backlog-results/` 时，先预览再迁移：
  ```bash
  python research_cli.py migrate-workspace --dry-run
  python research_cli.py migrate-workspace
  ```
  预览不写入存储。正式迁移把旧批次和旧版 `runs/*.json` 导入 `workspace.db`。已有训练证据的 QUICK 批可进入 `GATE_TRIAGE`，缺少冻结证据的旧 FULL 批保留在 `ARCHIVED`。旧 run JSON 保留，经过验证的数据缓存和原队列目录可归档至 `legacy-v3-archive/`。迁移统计包含 run、PnL、槽位和 datafields。
