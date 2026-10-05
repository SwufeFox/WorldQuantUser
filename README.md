# Alpha Research CLI

用结构化 JSON 批量运行 `machine_lib.py` 中已有的 Alpha 模拟流程。CLI 只提供 `simulate` 子命令，没有提交 Alpha 的命令。

## 桌面 GUI

使用 Python 3 启动标准库 Tkinter 界面：

```sh
python research_gui.py
```

在表达式编辑区每行输入一条 Alpha；可直接输入 `expression | decay`（例如 `rank(close) | 8`），省略 decay 时默认为 `6`。GUI 可加载和保存与下方 CLI 相同的 JSON 格式。点击“开始模拟”前会显示真实服务操作确认；确认后会登录并向 WorldQuant BRAIN 提交模拟，界面日志不会输出认证凭据。当前 `machine_lib.single_simulate` 返回 `None`，因此 GUI 会如实显示模拟状态未知。GUI 不提供 Alpha 提交流程。

## CLI 使用

```sh
python research_cli.py --help
python research_cli.py simulate --help
python research_cli.py simulate research.example.json
```

示例文件：[research.example.json](research.example.json)。每个 alpha 可独立指定 `decay`，不指定时使用 `6`。运行 CLI simulate 会调用真实 BRAIN 服务；请仅在有意执行模拟且已配置 `machine_lib.login()` 凭据时运行。帮助命令不会导入 `machine_lib` 或访问服务。

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

CLI 按 `--batch-size`（默认 3）分组，并按 `single_simulate(alpha_pools, neut, region, universe, start)` 调用现有函数。该函数返回 `None`，且内部通过打印报告进度。CLI 成功 JSON 中 `result` 为 `null`；`simulation_status` 固定为 `unknown`，无法据其返回值确认任务完成。GUI 同样不会声称模拟完成。

输入错误退出码为 `2`，模拟调用错误退出码为 `3`。验证时可运行 `python research_cli.py --help`、`python research_cli.py simulate --help`；不要为验证而运行 simulate，因为它会请求真实 BRAIN 服务。
