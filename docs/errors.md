# 错误码、状态诊断与恢复指南

`research_cli.py` 始终向 `stdout` 输出**单个 JSON 文档**（日志与进度写向 `stderr`）。
Agent 与脚本必须按结构化字段分流处理，**不要盲目重试、删 `daemon/daemon.lock`、改名重发绕过重复检测，或直接修改 `runs/*.json`**。

先看单条命令参数与机器契约：

```bash
python research_cli.py help COMMAND
python research_cli.py schema COMMAND
```

主流程规范见 [状态机实验流程与台账复盘](ledger-workflow.md)，命令清单见 [工具地图](tooling.md)。

---

## 1. JSON 响应信封与 `error` 结构

### 成功响应（退出码 `0`）

```json
{
  "ok": true,
  "command": "inbox",
  "data": { ... }
}
```

> [!IMPORTANT]
> `ok: true` 只说明 CLI / Daemon IPC 正常响应，**不代表实验已经推进到门禁或已提交**。必须继续检查 `data.state`、`data.execution`、`data.stalled`、`data.next_wake_at` 或 `data.actionable`。

### 失败响应（退出码非 `0`）

```json
{
  "ok": false,
  "command": "act",
  "error": {
    "code": "INVALID_INPUT",
    "message": "missing non-empty `idea` on arms: c1",
    "hint": "set `idea` on the spec or every arm explaining the hypothesis",
    "details": { ... }
  }
}
```

- `error.code`：稳定错误码，用于程序分支与退出码映射。
- `error.message`：具体失败事实。
- `error.hint`：直接给出下一步可执行的修复或恢复建议（所有 `CliError` 均附带 `hint`）。
- `error.details`：可选结构化现场信息（例如 `protected_path`、`problems`、`missing_idea_labels`、`run_id`、`suggestions`）。

使用 `--save PATH` 保存回执时，写向 `<research_dir>/runs/`、`<research_dir>/daemon/`、`alpha-registry.jsonl`、`submit-buffer.jsonl` 或 `seeds.jsonl` 会被 `INVALID_INPUT`（`OutputConflict`）拒绝。

---

## 2. 全局 `ERROR_CODES` 与 `EXIT_CODES` 映射

下表与 `research_app/runtime.py` 中的 `EXIT_CODES` / `ERROR_CODES` 一一对应：

| `error.code` | 退出码 | 含义 | 常见诱因 | 标准处理方式 |
| --- | --- | --- | --- | --- |
| `OK` | `0` | 命令成功执行 | 正常返回、入队、等待唤醒或完成状态转移 | 按 `data.state` 与 `allowed_actions` 推进 |
| `USAGE` | `2` | 命令行参数或子命令语法错误 | 缺少必填参数、未知选项、调用已移除的 v3 命令 | 按 `hint` 运行 `help COMMAND` / `schema COMMAND` 修正参数 |
| `INVALID_INPUT` | `2` | 输入、动作、模板或路径校验失败 | spec 缺 `idea` / 语法错、`act` 在错误门禁态调用、`--save` 撞保护路径、Power Pool 描述不合模板 | 按 `hint` 修正输入并用 `lint` 预检 |
| `CONFIRM_REQUIRED` | `2` | 不可逆平台提交缺少显式确认 | 调用 `act RUN_ID submit` 未带 `--yes` | 确认 `inspect RUN_ID` 指标与 `cost_audit` 后附带 `--yes` |
| `AUTH_MISSING` | `3` | 未配置 BRAIN 凭据 | 环境变量或 `.env` 缺少 `BRAIN_USERNAME` / `BRAIN_PASSWORD` | 按 `.env.example` 配置凭据后运行 `auth check` |
| `AUTH_FAILED` | `3` | 凭据被 BRAIN 拒绝或会话失效 | 密码错误或会话过期 | 核对 `.env` 并运行 `auth check --relogin` |
| `AUTH_CAPTCHA` | `3` | BRAIN 登录要求先完成人机验证 | 账号触发安全校验 | 在浏览器完成认证后再重试，禁止脚本绕过 |
| `ACCESS_DENIED` | `3` | 无权限访问目标接口或 Daemon Bearer Token 无效 | 普通账号访问受限端面，或 loopback IPC 缺少/错误 `Authorization` | 检查账号权限，或重启当前工作区的 `daemon` |
| `RATE_LIMITED` | `4` | 请求触发 BRAIN 限流或配额耗尽 | 高频请求目录/检查，或触发 `quota_floor` | 尊重 `next_wake_at` / `reset_at` 等待自动恢复 |
| `SESSION_BUSY` | `4` | 同账号正在登录或另一 Daemon 已持有 `daemon.lock` | 重复调用 `daemon start` 或并发 `--relogin` | 运行 `daemon status` 复用已有 Daemon |
| `SERVICE_ERROR` | `4` | HTTP / 网络异常或远端 5xx | 临时通信失败、读取超时、Daemon 启动超时 | 若为 `AUTO_SUBMITTING`，绝不重发 `POST`，等待只读回查 |
| `NOT_FOUND` | `5` | 指定的 `run_id`、Alpha ID、数据集或文件不存在 | ID 拼写错误或 `--research-dir` 选错 | 运行 `brief` 与 `runs` 核对工作区和 ID |
| `SIMULATION_FAILED` | `6` | 模拟全部终态失败 | 表达式/Python 运行时错误导致无可用 Alpha | 在 `inspect RUN_ID` 查看 `records[].error` 与台账 `failure:` 记录；修复后重提新 spec |
| `SUBMIT_DISABLED` | `7` | 当前环境策略禁止真实提交 | 未设置 `BRAIN_ALLOW_AGENT_SUBMIT=true` | 保持在架（`act RUN_ID hold` -> `SHELVED`），或在授权环境下开启变量 |
| `INTERRUPTED` | `130` | 进程收到中断信号 | `Ctrl+C` 或短命令提前退出 | 状态已在 Daemon 与 `runs/*.json` 持久化，随时可用 `wait` / `inbox` 接续 |

---

## 3. 状态机阻塞与熔断诊断

### 3.1 `GATE_PREFLIGHT_BLOCKED`

`AUTO_PREFLIGHT` 会在以下三类情况下切入 `GATE_PREFLIGHT_BLOCKED` 并等待人工或智能体处理（查看 `inspect RUN_ID` 的 `preflight_block`）：
- `reason == "exact_repeat"`：账本中所有臂均已有相同表达式 + 设置的已完成记录。复用已有 Alpha，或直接 `act RUN_ID drop --reason ...`。
- `reason == "preflight_rejected"` / `"exploration_admission"`：在线字段/算子目录校验失败或未通过探索准入。检查 `preflight_block.details` 并弃用改写，勿重命名绕过。
- `reason == "stalled_deferral"`：在没有明确未来 `Retry-After` 的情况下，连续 `max_unchanged_defers`（默认 3）次轮询均未改变进度哈希。排查网络或平台状态后执行 `act RUN_ID retry-preflight` 或 `drop`。

### 3.2 `AUTO_QUICK_SIM` / `AUTO_FULL_VALIDATE` 暂停与 `stalled`

- **配额保护 (`quota_floor`)**：若 `daemon config` 设了 `quota_floor > 0` 且剩余配额 ≤ `quota_floor`，未发送的新 QUICK 模拟暂停并设置 `next_wake_at`（直至 `reset_at`）；已批准的 `AUTO_FULL_VALIDATE` 会忽略软下限继续推进，除非平台返回 `blocked: true`。
- **进程重启恢复**：若宿主机重启或 Daemon 崩溃，重新运行 `daemon start` 会自动重新扫描 `runs/*.json`，继续推进所有 `AUTO_*` 任务并定期回收孤儿槽位。

---

## 4. 禁止盲目重试的高危场景（Agent 红线）

1. **最终提交不确定 (`submit_attempt.state == "UNCERTAIN"`)**
   - 当 `POST /alphas/{id}/submit` 遭遇超时、504 或回读失败时，状态保持在 `AUTO_SUBMITTING` 且标记 `post_attempted: true`。
   - Worker 下一轮仅调用只读 `fetch_alpha_metrics` 与 `read_alpha_submission_status` 对账，**严禁重复调用 `POST /submit`** 或新建相同 spec 再提交。
2. **协议级 Train/Test 隔离 (`selection_frozen == false`)**
   - 在执行 `act RUN_ID approve-full` 之前，不得通过直接读底层原始文件绕过 `inspect RUN_ID` 查看未冻结的测试期表现。
3. **Power Pool 元数据模板 (`GATE_DESCRIBE`)**
   - 必须逐字包含 `Idea:`、`Rationale for data used:`、`Rationale for operators used:` 三段标题，每段后跟 `- ` 列表，总长 ≥ 100 字符；`AUTO_BUFFER` 会回读远端 `regular.description` 严格一致后才写盘。
4. **锁冲突（`SESSION_BUSY` / `daemon.lock`）**
   - 禁止手动删除 `daemon/state.lock`、`daemon/daemon.lock` 或 `simulation-slots/*.lock`；使用 `daemon status` 与 `daemon stop` 管理常驻进程。
