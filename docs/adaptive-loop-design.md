# 自适应 Alpha 研究闭环：最小设计

目标：让「研究策略」成为可记录、可复现、可比较的对象，而不是散在对话里的判断。
不引入 Rust、GP 或新依赖；回测仍走现有 `probe` / daemon / 配额账本。

## 1. 先审计：已经有什么

| 需要的能力 | 现有组件 | 缺口 |
|---|---|---|
| 状态持久化 | `runs/*.json` + `workspace_db.py`（runs/events） | 没有「策略状态」这一层 |
| 动作执行 | `propose` → `AUTO_*` → `GATE_*` | 动作是人工逐个发起 |
| 候选身份 | `semantic_identity.py`（新）、骨架计数 | 身份未写进记录 |
| 评价结果 | `alpha-registry.jsonl`、`ledger` | 缺 evaluator context、缺血缘 |
| 评价可比性 | 无 | 不同池/窗口/检查版本的分数混在一起 |
| 选择策略 | `batch_admission.build_admission` 固定规则 | 单函数，不可替换，不可对照 |
| 实验控制 | 无 | 没有对照组，无法归因 |

结论：**执行层够用，记录层缺三个字段，策略层完全空白。**

## 2. 五个显式对象

字段尽量少，能少一个是一个。全部是 JSON，落盘在 `research_dir/strategy/`。

### Candidate（候选）

```json
{"expr": "...", "settings": {...}, "identity": "sha256:...",
 "parents": ["alpha:ABC"], "policy": "pkg.mod:fn@1", "seed": 7, "batch": "b-042"}
```

`identity` 用现有 `semantic_identity.fingerprint`，`parents` 给血缘，`policy` 记录是谁生成的。

### Evaluation（评价）

```json
{"alpha_id": "ABC", "context": {"region": "HKG", "universe": "TOP500", "delay": 1,
  "train_start": "2014-01-01", "train_end": "2022-12-31", "checks_version": "2026-11",
  "cost_bps": 0},
 "metrics": {"sharpe": 2.44, "fitness": 2.62, "turnover": 0.088, "margin": 0.0033},
 "checks": {"IS_LADDER_SHARPE": "PASS"}, "verdict": "pass|watch|fail"}
```

**只有 `context` 完全相同的两条 Evaluation 才允许比较**——这是 AlphaWinnow 的 evaluator context
落到我们这里的最小形态，也是对过去「拿不同池的 Sharpe 比大小」的直接修正。

### State（状态）

```json
{"budget": {"arms_left": 3800, "quota_floor": 20},
 "counters": {"arms": 1730, "pass": 61, "explore": 120},
 "parents": ["alpha:AAA", "alpha:BBB"],
 "last_batch": "b-042", "history": ["b-041", "b-042"]}
```

状态是**可重建的**：`state = replay(events)`，不手工编辑。

### Action（动作）

最小集合三个：`propose(batch)`、`select(parents, k)`、`stop(reason)`。
每个 Action 带 `policy` 与 `seed`，因此同一条记录能重放出同一个决策。

### Update（状态更新）

`state' = update(state, evaluations)`，纯函数：给定同样的 evaluations 必得同样的 state'。
不做隐式累加，所有计数都在这一处改。

## 3. 三个插拔点

策略是**纯函数模块**，用 `模块:函数` 字符串引用（如 `policies.baseline:select`），
在批次配置里指定。这样不需要插件框架，也不需要改 daemon。

| 插点 | 签名 | 默认实现 |
|---|---|---|
| Proposer | `propose(state, budget) -> [Candidate]` | 当前模板族（8 臂同题变体） |
| Selector | `select(state, candidates, k) -> [Candidate]` | 按训练分排序截断 |
| Updater | `update(state, evaluations) -> State` | 计数 + 父代替换 |

三条硬约束：纯函数（不读时钟、不写文件）、只用传入的 state、输出必须能反序列化。
违反任何一条，这条策略的记录就不可复现。

## 4. 探索预算

`exploration_every: N`：Selector 每 N 个槽位强制选一个「身份不在历史 parents 里」的候选；
`N=1` 表示完全不用历史父代（纯探索），`N=0` 表示关闭强制探索。

这不改变 Proposer，只在选择处生效——所以可以用同一批候选做两种预算的对照。

## 5. 实验控制：先离线，再在线

### E0（零配额，先做）

用台账回放：取历史上某一批候选池，比较「我们的选择器会选哪 8 条」与「随机选 8 条」，
看两条路各自选中的候选在台账里已有的实测结果。不新增任何回测。
它只能回答「在同一历史池上，选择器是否比随机好」，样本有偏，但成本为零。

### E1（小规模在线）

冻结 Proposer（同一个模板族），只换 Selector：`score-ranked` vs `uniform-random`。
两臂各 400 次回测，同一 evaluator context，配对比较每千臂 PASS 数。
这是 `LatestBatchUniformV1` 的最小移植：**对照组不是可选项，是实验的一部分**。

### E2（探索配额）

同一 Proposer，`exploration_every = 0 / 4`，各 400 臂，比 PASS 产出与机制覆盖。

## 6. 与现有代码的接法

- Proposer 的产物就是 `propose` 接受的 spec，不改协议。
- Selector 在 `GATE_TRIAGE` 处生效：决定哪些臂进 `hydrate-pnl` / `approve-full`。
- Evaluation 由现有 `ledger record` 写入，只多三个字段：`context`、`identity`、`parents`。
- Updater 写在 `strategy/state.json`，不动 `runs/*.json`。

## 7. 不做的事

不做 GP、不做自动改表达式、不做在线学习、不引入新服务。先把「记录 + 对照」跑通，
再谈学习；没有可比记录，学出来的东西也没法验证。

## 8. 风险

策略显式化会让「研究」一开始变慢：要写策略、要跑对照、要等结果。短期产出下降是预期内的。
真正的收益是能回答「我们的策略有用吗」——这个问题现在一次都回答不了。
