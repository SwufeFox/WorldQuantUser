# 错误码、状态诊断与恢复指南

`research_cli.py` 始终向 `stdout` 输出**单个 JSON 文档**（日志、进度与 `--stream` 事件写向 `stderr`）。
Agent 与脚本必须按结构化字段分流处理，**不要盲目重试、删除锁文件、改名重发或切换到另一个 runner 绕过门禁**。

先看单条命令参数与机器契约：

```bash
python research_cli.py help COMMAND
python research_cli.py schema COMMAND
```

主流程规范见 [批量回测与台账复盘](ledger-workflow.md)，命令清单见 [工具地图](tooling.md)。

---

## 1. JSON 响应信封与 `error` 结构

### 成功响应（退出码 `0`）

```json
{
  "ok": true,
  "command": "campaign status",
  "data": { ... }
}
```

> [!IMPORTANT]
> `ok: true` 只说明命令被正常处理，**不代表实验已经出分或提交已成功**。对批次回测与晋级任务，必须继续读取 `data.jobs[].diagnosis`、`data.execution`、`data.outcome` 或 `data.would_block`。

### 失败响应（退出码非 `0`）

```json
{
  "ok": false,
  "command": "pairs",
  "error": {
    "code": "INVALID_INPUT",
    "message": "spec must contain a list of pairs or a dict with 'pairs'",
    "hint": "provide --spec PATH to JSON containing pairs or pass expressions directly",
    "details": { ... }
  }
}
```

- `error.code`：稳定错误码，用于程序分支与退出码映射。
- `error.message`：具体失败事实。
- `error.hint`：直接给出下一步可执行的修复或恢复建议（`CliError` 与主要 `BrainError` 子类均附带 `hint`；极少数平台透传异常在无额外上下文时为 `null`）。
- `error.details`：可选结构化现场信息（例如检查点路径 `checkpoint`、`recording` 失败记录摘要、`protected_path`、`unknown_submission` 或 `retry_after`）。

使用 `--save PATH` 保存回执时，失败时的 `error` 信封同样会原子写入 `PATH`，但写向 batch/checkpoint/selection/shelf/seed 等受保护状态路径会被 `INVALID_INPUT`（`OutputConflict`）拒绝。

---

## 2. 全局 `ERROR_CODES` 与 `EXIT_CODES` 映射

下表与 `research_app/runtime.py` 中的 `EXIT_CODES` / `ERROR_CODES` 一一对应：

| `error.code` | 退出码 | 含义 | 常见诱因 | 标准处理方式 |
| --- | --- | --- | --- | --- |
| `OK` | `0` | 命令成功执行 | 正常返回或仅进入等待/预检态 | 继续读取 `data` 中的实际证据与 `next_action` |
| `USAGE` | `2` | 命令行参数或子命令语法错误 | 缺少必填参数、未知选项、互冲突参数 | 按 `hint` 运行 `help COMMAND` / `schema COMMAND` 修正参数 |
| `INVALID_INPUT` | `2` | 输入文件、表达式、设置、描述或路径校验失败 | spec 缺少字段、FASTEXPR 语法/类型错误、`--save` 覆盖受保护状态、Power Pool 描述不符合模板 | 按 `hint` 修正输入或使用 `lint` / `campaign preview` 预检 |
| `CONFIRM_REQUIRED` | `2` | 涉及平台写入或不可逆操作的确认门禁 | 首次调用需要二次确认或缺少 `--yes` | 核对回执摘要无误后，按 `hint` 重复同一确认请求 |
| `AUTH_MISSING` | `3` | 未配置 BRAIN 凭据 | 环境变量或与 `machine_lib.py` 同目录的 `.env` 缺少 `BRAIN_USERNAME` / `BRAIN_PASSWORD` | 按 `.env.example` 配置凭据后运行 `auth check` |
| `AUTH_FAILED` | `3` | 凭据被 BRAIN 拒绝或当前会话失效 | 密码/账号错误或会话长期失效 | 核对 `.env` 或环境变量并运行 `auth check --relogin` |
| `AUTH_CAPTCHA` | `3` | BRAIN 登录要求先完成人机验证 | 账号触发安全校验 | 按 `error.hint` 中的 URL 在浏览器完成认证后再重试，不可脚本绕过 |
| `ACCESS_DENIED` | `3` | 账号无权限访问目标接口 | 普通用户调用 Consultant/Power Pool 专属端面等 | 用 `auth profile` / `auth competitions` 确认账号权限边界 |
| `RATE_LIMITED` | `4` | 请求触发 BRAIN 限流且退避预算耗尽 | 并发过高、密集拉取目录/指标/子模拟 | 停止盲目重发，等待 `retry_after` / `Retry-After` 后由原 owner 恢复 |
| `SESSION_BUSY` | `4` | 同账号另一个进程正在登录 BRAIN | 并发启动多个会触发重登录的命令 | 稍后重试；确保共用缓存会话而非并发 `--relogin` |
| `SERVICE_ERROR` | `4` | HTTP / 网络层与 BRAIN 通信异常或远端 5xx | 网络抖动、超时、服务端临时错误 | 若发生在只读查询可稍后重试；若涉及模拟或提交 POST，**必须先按第 4 节核对是否已受理** |
| `NOT_FOUND` | `5` | 指定的 Alpha ID、数据集、文件或接口对当前账号不存在 | ID 拼写错误、研究目录 `--research-dir` 选错 | 先跑 `brief` 核对 `research_dir` 与台账路径，或 `alpha list` 确认 ID |
| `SIMULATION_FAILED` | `6` | 本次模拟批次全部终态失败 | 表达式/Python 运行时错误、平台拒绝全部子模拟 | 检查 `error.details.recording`、`error.details.checkpoint` 与原文错误；修复假设/写法另建规格，勿把执行报错等同于经济假设 `falsified` |
| `SUBMIT_DISABLED` | `7` | 当前环境或策略禁止真实平台提交 | 缺少提交确认开关或只读环境限制 | 确认提交权限、是否应只保留到 `buffer add` 阶段 |
| `INTERRUPTED` | `130` | 进程收到中断信号（SIGINT / SIGTERM） | 手动 `Ctrl+C` 或外部超时终止 | 本地等待中断不取消平台已有任务；下次用原检查点恢复 |

---

## 3. `ok: true` 时的状态诊断（预览、队列、Probe 与晋级）

即使 `ok: true`，也要按阶段读取实际诊断结果：

### 3.1 `campaign preview` 阻塞信号

预览会检查 `data.would_block`、`data.blocking` 与 `data.gate_blocking`：

- `blocking` / `gate_blocking`：表示仓库/台账中已有完全相同表达式 + 设置的结果、在途任务或待核实记录。
- 正确动作：沿 `data.results[].gate` 与 `data.recovery_actions` 恢复或复用原证据；**禁止通过修改 arm label、微调无意义括号、或换运行器来绕过**。

### 3.2 `campaign status` 中的 `diagnosis.state` 与 `diagnosis.reason`

`campaign status` 优先根据已保存的 batch checkpoint 推导 `jobs[].diagnosis`（未建 batch 的新规格直接显示 `state: "queued"`）：

| `diagnosis.state` | `diagnosis.reason` | 含义与正确动作 |
| --- | --- | --- |
| `ready` | `queued_work` | 有尚未提交的 queued 臂，交由唯一执行者按 `diagnosis.next_action.argv`（`probe --resume` 或 `campaign run`）推进 |
| `waiting` | `platform_pending` | 平台模拟仍在执行中，遵守 `next_poll_at` 等待，不要提前连环轮询 |
| `waiting` | `preflight_backoff` | 字段/算子预检进入退避窗口，等 `preflight_deferred` 的 `retry_at` 到期后自动续跑 |
| `waiting` | `shared_slots_occupied` | 同账号+研究目录共享模拟槽位已满，必须先通过原检查点回读平台终态来释放占用 |
| `waiting` | `submission_cooldown` | 触发发单冷却退避，等冷却结束再发新单；已有已受理任务仍可先回收 |
| `blocked` | `submission_unknown` | **曾有模拟 POST 结果不明**！立即停止新 POST，运行 `diagnosis.next_action.argv` 核对平台是否已受理 |
| `done` | `complete` | 该 batch 所有需要执行的臂已达到完成状态，进入 `ledger analyze` / `triage` 复盘 |
| `failed` | `terminal_failure` | 该 batch 已有确认终态失败；检查日志与台账失败记录，切勿反复重跑同一失败规格 |

### 3.3 Probe 暂停与阻断原因（`pause` / `execution`）

Direct `probe` / `campaign run` 在跳过或暂停一轮调度时，会在结构化回执中给出以下原因（对应 `research_app/experiments.py`）：

- `deferred` / `reconciled` / `skipped` / `unavailable`（步骤状态）
- 常见原因字段 `reason`：
  - `accepted_slot_reconciliation`：已有已受理槽位需要先对齐回收，当前先不对新臂发 POST。
  - `batch_locked`：目标 batch 检查点被锁；检查现有 owner 进程，确认退出后再交接，不可直接删锁。
  - `duplicate_arm_labels`：规格存在重复 arm label；修正 spec 文件（新实验按新 spec 处理）。
  - `exact_repeat`：当前候选与历史/在途请求完全一致，需复用原检查点或结果。
  - `python_source`：Python 策略源码声明或静态解析门禁拦截（如未声明 `@alpha(data=[...])`、`lookback` 或语法/隐式字段错误）。
  - `reconcile_recording` / `recording_disabled`：本地台账核对记录未完成或当前处于只读/禁写模式（例如 `--dry-run`）。
  - `shared_slots_occupied`：并发账号槽位占满，暂停新增发单。
  - `submission_cooldown`：发单冷却中。

- **`campaign diagnose` 与单写入者配额门禁**
  - `campaign diagnose --campaign-dir DIR` 离线检查顶层规格、`backlog/failed/.reasons/` 失败记录、`backlog-reserve/eurusa-approved`、`native-multi` 到期时间及 `issues`（`no_ready_top_level_specs`、`approved_reserve_empty`、`unreadable_queue_entries`、`specs_outside_top_level`、`failed_specs_missing_recent_reason`、`native_dispatch_deadline_elapsed`）。
  - `campaign publish` 会依次验证 16+ 臂同 scope QUICK、结构限额、`probe --check --strict-gate` 与同账号三数据集 catalog 证据；扩展阶段（`--phase expansion --pilot-batch`）还会校验相同 scope 下同一 `mechanism` 是否在已完成的 16 臂 pilot 中达到训练 Sharpe ≥ 1.0、正 Fitness 且换手率 ≤ 35%。
  - 账号日配额账本由单写入者锁保护：在同一 reset 窗口内收到 `429` 或配额耗尽（`blocked=true`）时停止发单，等待 `reset_at` 之后再恢复。

### 3.4 `promote` 与 `campaign run --validate` 的 `metadata_required`

当训练、Test 留存与实时平台 checks 均已通过，但候选说明未提供完整时：

- 回执中 `outcome == "metadata_required"` 且 `buffered == false`。
- `required_options` 列出缺失参数（例如 `--name`、`--category`、`--tags`、`--description`）。
- 正确动作：**先运行 `alpha describe` 写入官方模板描述，再把参数补齐到回执返回的 `next_action.argv`（保留原 `--selection`）执行一次 `promote` / `buffer add`**。不要重新跑模拟，也不要空参循环重试。

---

## 4. 禁止盲目重试的高危场景（Agent 红线）

1. **模拟发单状态不明（`submission_unknown` / `UNKNOWN_SUBMISSION`）**
   - 必须从保存的 `BATCH.json` 运行 `probe --recover BATCH.json`（或 `probe --from-simulation ID`）确认平台是否已生成任务，确认前禁止任何新的 `POST /simulations`。
2. **最终 Alpha 提交状态不明（`UNCERTAIN` / `UNKNOWN`）**
   - 先用只读模式执行 `python research_cli.py alpha submit ALPHA_ID --max-wait 30` 检查 `submission_recovery`。若平台已返回明确 `REJECTED` 或确认已提交，只在原有授权下执行同一命令加 `--yes` 持久化结果；`UNSUBMITTED` 本身不证明上一轮 POST 未发出。
3. **锁冲突（`batch_locked` / `runner_locked` / `SESSION_BUSY`）**
   - `lock_path` 和 `last_runner_event` 只是核查线索；禁止按文件修改时间自动删锁或手动删除 reservation。
4. **本地等待超时 ≠ 取消远端任务**
   - `--max-wait` / `--max-wait-seconds` 到期只结束本地等待进程，不会向平台发送 `DELETE`。要主动取消任务必须显式执行 `alpha cancel-sim SIM_ID`。
5. **Power Pool 模板与检查红线**
   - 描述必须逐字包含三段标题 `Idea:`、`Rationale for data used:`、`Rationale for operators used:`，每段后跟 `- ` 列表项，去除首尾空白后总长至少 100 字符（旧全大写 `IDEA:` / `RATIONALE:` 会直接报 `INVALID_INPUT`）。
   - `buffer add` 会回读平台 `regular.description` 校验一致；任意检查项为 `FAIL`（如 `LOW_FITNESS`）时禁止入库和提交。
