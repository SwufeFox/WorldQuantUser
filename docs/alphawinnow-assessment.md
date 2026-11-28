# AlphaWinnow 评估（2026-11-28）

外部项目：`github.com/eslazarev/alphawinnow`，MIT，Rust CLI + 库，1.7 MB / 30 个 rs 文件。
定位是**本地公式化 alpha 挖掘引擎**：类型化表达式语法、确定性遗传编程、语义去重、
结构新颖性搜索、Rayon 并行、可选本地评估。明确不做 BRAIN 客户端、不做调度、不做提交。

## 它处在我们的哪一层

| 层 | 我们的现状 | AlphaWinnow |
|---|---|---|
| 候选生成 | LLM + 模板 + 人工设计 | 类型化 GP，确定性可复现 |
| 去重 | 骨架计数（结构） | 语义指纹（等价规则下同一身份） |
| 新颖性 | 无 | 有显式结构新颖度 |
| 反馈 | ledger 里 24k 条实测 alpha | ask/tell：按实测分数更新父代 |
| 平台交互 | 我们自己做（probe/daemon） | 明确不做，留给 adapter |

它把「生成 + 去重 + 按实测反馈选下一代」做成了闭环，正好是我们最薄的一段。

## 直接对得上我们痛点的三件事

1. **8 臂同骨架的问题**。我们现在的闸门数的是骨架（结构），它数的是语义身份——
   换窗口、换等价算子都会被折叠，能逼着我们用 8 个真不同的想法填满一批。
2. **冷启动重复烧**。它支持从已审计的历史表达式 warm-start 并排除已探索身份；
   我们 ledger 里有 24,141 个 alpha id、40,781 条台账，正好是它的输入形态。
3. **可复现**。确定性 GP + lineage，批次能重放出同一个候选集；我们现在做不到。

## 明确的限制

- 0.3.0 的 `compile` 用内置 parser，**不接受自定义 catalog**；内置算子覆盖
  `rank/zscore/winsorize/group_rank/group_count` 和 `ts_*` 一族（mean/delta/corr/
  cov/decay_linear/delay/ema/kurt/mad/max/median/min/rank/skew/std_dev/sum/var/wma/zscore），
  但**没有** `group_mean/group_zscore/group_neutralize/group_backfill`、`if_else`、
  `trade_when`、`vec_avg`、`signed_power`、`bucket`、`ts_argmax`。要覆盖我们常用写法得加映射层。
- 它不给任何收益承诺：文档里写明 structural score ≠ Sharpe，measured outcome 由调用方提供。
- pre-1.0，公开错误变体仍在加；要用就得锁版本。
- 需要 Rust 1.92+ 工具链；VPS 只有 2 核，编译要几分钟。

## 建议

不做整体替换。两条路：

**A. 只借思想**：把「语义身份」写进我们的闸门，用它替代现在的骨架计数；warm-start 用
ledger 自己实现（我们已有 40k 条台账，只差一个身份函数）。

**B. 做一次 spike**：本机编出二进制，导出单区（HKG/TOP500）的字段与算子 catalog，
生成 1,000 条候选，只比较两件事——合法率、与我们现有批次的语义重合率。不烧配额。
如果语义去重确实比骨架计数强，再考虑接 ask/tell。

倾向先做 B：它给出的是可比较的数字，而不是又一次「看起来不错」。
