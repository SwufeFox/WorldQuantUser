# Alpha Research CLI

用结构化 JSON 批量运行 `machine_lib.py` 中已有的 Alpha 模拟流程，并取回每条表达式的 Alpha 指标。CLI 只提供 `simulate` 子命令，没有提交 Alpha 的命令。

## 安装依赖

```sh
pip install -r requirements.txt
```

运行依赖 `requests` 和 `pandas`（见 `requirements.txt`）。

## 凭据配置

凭据不再写在源码里。`machine_lib.login()` / `login_hk()` 通过 `machine_lib.get_credentials()` 读取 `BRAIN_USERNAME` 和 `BRAIN_PASSWORD`：优先取真实环境变量，其次取 `machine_lib.py` 同目录下 `.env` 文件中的同名键（真实环境变量优先）。任何时候都不要把凭据提交进仓库。

任选一种方式：

```sh
# 方式一：环境变量
export BRAIN_USERNAME="your-username"
export BRAIN_PASSWORD="your-password"

# 方式二：.env 文件（已被 .gitignore 忽略）
cp .env.example .env
"$EDITOR" .env
```

缺少任一凭据时，`get_credentials()` 会抛出 `RuntimeError` 说明如何配置，而不是用空字符串去请求服务。

## 桌面 GUI

使用 Python 3 启动标准库 Tkinter 界面：

```sh
python research_gui.py
```

在表达式编辑区每行输入一条 Alpha；可直接输入 `expression | decay`（例如 `rank(close) | 8`），省略 decay 时默认为 `6`。GUI 可加载和保存与下方 CLI 相同的 JSON 格式。点击“开始模拟”前会显示真实服务操作确认；确认后会登录并向 WorldQuant BRAIN 提交模拟，界面日志不会输出认证凭据。模拟结束后 GUI 按每条表达式返回的 `status` 汇总显示（例如 `COMPLETE 3`）。GUI 不提供 Alpha 提交流程。

## CLI 使用

```sh
python research_cli.py --help
python research_cli.py simulate --help
python research_cli.py simulate research.example.json
```

示例文件：[research.example.json](research.example.json)。每个 alpha 可独立指定 `decay`，不指定时使用 `6`。运行 CLI simulate 会调用真实 BRAIN 服务；请仅在有意执行模拟且已按上文「凭据配置」设置好 `BRAIN_USERNAME` / `BRAIN_PASSWORD` 时运行。帮助命令不会导入 `machine_lib` 或访问服务。

## 输入格式

```json
{
  "settings": {
    "region": "USA",
    "universe": "TOP3000",
    "neutralization": "SUBINDUSTRY",
    "start": 0
  },
  "alphas": [
    {"expression": "rank(ts_delta(close, 5))", "decay": 6}
  ]
}
```

`settings` 可省略，默认值依次为 `USA`、`TOP3000`、`SUBINDUSTRY`、`0`。`region` 接受 USA、GLB、EUR、ASI、CHN、HKG、JPN、KOR、TWN；`universe` 必须为非空字符串；`neutralization` 接受 NONE、MARKET、SECTOR、INDUSTRY、SUBINDUSTRY、COUNTRY、SLOW、FAST；`start` 为非负整数。`alphas` 必须是非空数组，表达式为非空字符串，`decay` 为非负有限数值。

CLI 按 `--batch-size`（默认 3）分组，并调用 `single_simulate(alpha_pools, neut, region, universe, start)`。该函数返回每条表达式一条记录：`{"expression", "decay", "alpha_id", "status", "metrics", "error"}`，其中 `metrics` 是完成后的 Alpha 设置与样本内指标（`sharpe`、`fitness`、`turnover`、`checks` 等），`error` 在失败时给出原因。CLI 成功 JSON 的 `result` 即该记录数组，`result_summary` 按状态计数，`simulation_status` 为 `complete` / `partial` / `failed`（无结果时为 `unknown`）。提交或轮询失败的表达式会按条记录错误，而不是被静默丢弃；会话过期时会重新登录并重试一次。

输入错误退出码为 `2`，模拟调用错误退出码为 `3`。验证时可运行 `python research_cli.py --help`、`python research_cli.py simulate --help`；不要为验证而运行 simulate，因为它会请求真实 BRAIN 服务。
