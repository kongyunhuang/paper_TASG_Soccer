# 表 7 的 2024/25 配对自助区间（202 场，按场次重抽 2,000 次）

写于 2026-10-06 10:30，本机数值收尾会话（xcv2-96）写，任务卡 `00_任务卡_本机数值收尾.md` 第二件。评估报告第六节第 7 条要的，Javier 10-05 批注 2.1 也相关。tex 没动，数和建议句交主会话转改稿会话。

## 怎么算的

- 脚本 `scripts/eval/paired_bootstrap_2425.py`，日志 `logs/paired_bootstrap_2425_2026-10-06.log`，全部结果（点估计、区间、逐种子 AUC、每个对比用了哪几个种子）在 `audit/2026-10-06_paired_bootstrap_2425.json`。重跑命令 `cd 01_dev_workspace && PYTHONPATH=. python -u scripts/eval/paired_bootstrap_2425.py --n_boot 2000`，约半分钟。
- 做法照搬 `scripts/eval/holdout_2526_summarize.py` 的 bootstrap。AUC 用它的 `AUCFast`（按场次整数权重重算，并列算一半）；随机数 `np.random.RandomState(20261003)`，和留出确认同一个种子；每次重抽从 202 场里有放回抽 202 场，七个任务、全部模型、全部种子共用同一次重抽；区间取 2,000 个重抽值的 2.5 和 97.5 分位。
- 差在种子均值上做。每次重抽里先算每个模型每个种子的 AUC，再按下面写的种子取均值，再相减。配对规则也照搬留出脚本的 `paired_diff`，单任务模型之间按共有种子配对，统一模型对单任务模型是种子均值之差、不配对。
- 202 场取法和表 7 相同（`summarize_u1_clean.clean_matches`，状态正常且射门带帧率不低于 70%）。七个任务的事件数 Pass 26,482、Shot 4,929、Interception 2,605、Ball recovery 16,714、Pressure 26,484、Dribble 5,060、Tackle 6,110，和表 7 的 Events 列相同。统一模型和单任务模型的测试事件逐场比过事件数和正例数，全部相同。
- 只读 `data/cache/u1_clean/` 和 `data/cache/single_clean_rich/preds/`，没写 `data/`。卡上列的 `single_clean/`（薄标量单任务）这两组对比用不到，没读。

### 用了哪几个种子

| 对比 | 左边 | 右边 | 配对 |
|---|---|---|---|
| A 薄统一模型减加厚 XGBoost | u1 种子 0 到 3（四个）的均值 | 加厚 XGBoost 种子 0 到 4（五个）的均值 | 不配对，和表 9 C3 同一个量 |
| B 加厚 CNN 减加厚 XGBoost（主用） | 加厚 CNN 种子 0 到 2 | 加厚 XGBoost 种子 0 到 2 | 按种子 0 到 2 配对，等于两边三种子均值之差，就是正文 5.2 和改稿记录第十六节的口径 |
| B5 同 B，只作参照 | 加厚 CNN 种子 0 到 2 | 加厚 XGBoost 种子 0 到 4 | 不配对，是表 7 两格均值直接相减的口径，**不要和 B 混用** |
| C 加厚统一模型减加厚 XGBoost（卡外） | u1rich 种子 0 到 3 | 加厚 XGBoost 种子 0 到 4 | 不配对，正文 5.3 第二句的量，顺手算 |

### 核对

- 四个模型（u1、u1rich、加厚 XGBoost、加厚 CNN）七个任务共 28 个种子均值，和表 7 出处登记 `scripts/paper_v2/out/tab_outcome_main_sources.json` 四位一致（脚本里写死的闸，对不上就停）。
- 点估计和正文、表 9 对过。B 的 Pass 负 0.0070、Interception 0.0093、Ball recovery 和 Pressure 各 0.0015，逐种子 Interception 0.0036 到 0.0143，和正文 5.2 一致；A 的三任务平均 0.0003 和表 9 C3 的 2024/25 列一致；C 的 Shot 负 0.0057、Pressure 0.0027 和正文 5.3 一致。
- 加权 AUC 另用 `sklearn.metrics.roc_auc_score(sample_weight=…)` 在第一次重抽上抽四个运行独立重算，最大差 1.1e-16。
- 同一命令重跑一遍，2,000 次重抽的全部区间逐位相同。

## 结果

点估计、区间都是 AUC 之差，四位小数。“差为正的重抽样占比”是 2,000 次里差大于零的比例，只作参考。

### A 薄统一模型（四种子）减加厚 XGBoost（五种子）

| 任务 | 事件数 | 点估计 | 95% 区间 | 区间含 0 | 差为正的重抽样占比 |
|---|---:|---:|---|---|---:|
| Pass | 26,482 | -0.0040 | [-0.0051, -0.0027] | 不含 | 0.000 |
| Shot | 4,929 | -0.0022 | [-0.0129, +0.0085] | 含 | 0.338 |
| Interception | 2,605 | +0.0036 | [-0.0100, +0.0163] | 含 | 0.696 |
| Ball recovery | 16,714 | -0.0004 | [-0.0060, +0.0053] | 含 | 0.428 |
| Pressure | 26,484 | +0.0051 | [+0.0007, +0.0099] | 不含 | 0.989 |
| Dribble | 5,060 | -0.0050 | [-0.0126, +0.0027] | 含 | 0.102 |
| Tackle | 6,110 | +0.0002 | [-0.0064, +0.0066] | 含 | 0.524 |
| 七任务平均 |  | -0.0004 | [-0.0033, +0.0025] | 含 | 0.387 |
| 三任务平均（Pass、Ball recovery、Pressure） |  | +0.0003 | [-0.0021, +0.0027] | 含 | 0.589 |

### B 加厚 CNN 减加厚 XGBoost，种子 0 到 2 配对（主用）

| 任务 | 事件数 | 点估计 | 95% 区间 | 区间含 0 | 差为正的重抽样占比 |
|---|---:|---:|---|---|---:|
| Pass | 26,482 | -0.0070 | [-0.0083, -0.0056] | 不含 | 0.000 |
| Shot | 4,929 | +0.0026 | [-0.0079, +0.0131] | 含 | 0.666 |
| Interception | 2,605 | +0.0093 | [-0.0040, +0.0221] | 含 | 0.914 |
| Ball recovery | 16,714 | +0.0015 | [-0.0046, +0.0074] | 含 | 0.676 |
| Pressure | 26,484 | +0.0015 | [-0.0026, +0.0059] | 含 | 0.772 |
| Dribble | 5,060 | +0.0029 | [-0.0036, +0.0098] | 含 | 0.810 |
| Tackle | 6,110 | +0.0039 | [-0.0019, +0.0100] | 含 | 0.901 |
| 七任务平均 |  | +0.0021 | [-0.0008, +0.0050] | 含 | 0.922 |
| 三任务平均（Pass、Ball recovery、Pressure） |  | -0.0014 | [-0.0038, +0.0012] | 含 | 0.135 |

B 的逐种子差（不重抽，202 场原样）

| 任务 | 种子 0 | 种子 1 | 种子 2 |
|---|---:|---:|---:|
| Pass | -0.0072 | -0.0071 | -0.0068 |
| Shot | +0.0012 | +0.0034 | +0.0034 |
| Interception | +0.0143 | +0.0098 | +0.0036 |
| Ball recovery | +0.0024 | +0.0004 | +0.0015 |
| Pressure | +0.0016 | +0.0006 | +0.0022 |
| Dribble | +0.0051 | -0.0009 | +0.0044 |
| Tackle | +0.0058 | +0.0039 | +0.0021 |

### B5 加厚 CNN（三种子）减加厚 XGBoost（五种子），只作参照

| 任务 | 点估计 | 95% 区间 |
|---|---:|---|
| Pass | -0.0073 | [-0.0086, -0.0058] |
| Shot | +0.0020 | [-0.0083, +0.0124] |
| Interception | +0.0066 | [-0.0067, +0.0200] |
| Ball recovery | +0.0010 | [-0.0049, +0.0071] |
| Pressure | +0.0018 | [-0.0022, +0.0061] |
| Dribble | +0.0034 | [-0.0032, +0.0102] |
| Tackle | +0.0041 | [-0.0013, +0.0102] |
| 七任务平均 | +0.0017 | [-0.0011, +0.0046] |

结论和 B 相同（只有 Pass 不含 0），只是点估计换了口径。正文和表注用 B。

### C 加厚统一模型（四种子）减加厚 XGBoost（五种子），卡外

| 任务 | 点估计 | 95% 区间 | 区间含 0 |
|---|---:|---|---|
| Pass | -0.0047 | [-0.0059, -0.0035] | 不含 |
| Shot | -0.0057 | [-0.0169, +0.0051] | 含 |
| Interception | +0.0010 | [-0.0120, +0.0129] | 含 |
| Ball recovery | 0.0000 | [-0.0058, +0.0055] | 含 |
| Pressure | +0.0027 | [-0.0011, +0.0068] | 含 |
| Dribble | -0.0023 | [-0.0091, +0.0045] | 含 |
| Tackle | -0.0006 | [-0.0066, +0.0054] | 含 |
| 七任务平均 | -0.0014 | [-0.0042, +0.0014] | 含 |

## 和正文措辞对照（不改结论，报主会话定）

下面的 L 行号是 `02_manuscript/manuscript.tex` 在 2026-10-06 10:30 的行号，并入 Miguel 修订后会挪。

1. **正文 5.2、摘要、结论的“剩下 0.0015 到 0.009 的种子均值残差”**，点估计对，区间不冲突。
2. **讨论 L429“positive in 17 of the 18 seed-task pairs”和 L435“with a convolutional branch adding a small residual on six tasks”，有张力。** 六个 CNN 领先的任务，按场次重抽的区间全部含 0（差为正的占比 0.67 到 0.91），七任务平均 +0.0021 [-0.0008, +0.0050] 也含 0。17/18 个种子为正说的是同一批 202 场上训练种子之间方向一致，不是换一批比赛还会为正。“adding a small residual”读起来像一个已确立的正效应，场次重抽撑不住；留出冻结方案第二节 C2 那条也早写过“小而一致”的后半句只由 24/25 多种子支持。这条反倒和“大部分可被十五个汇总量替代”的主结论同向。可选改法，在 L429 或 L435 那句后补“which bootstrap intervals over the 202 test matches do not separate from zero”，或把 adding a small residual 改成 leaving a small residual that the match-level intervals do not separate from zero。
3. **“the unified model stays within 0.006 AUC of task-specific trees”（摘要 L66、5.3 L374、讨论 L431 和 L435、结论 L443），有张力但不算冲突。** 七个种子均值差都在 0.006 以内是点估计，成立；按场次重抽，Shot、Interception、Pressure、Dribble、Tackle 五个任务的区间伸出了 ±0.006（Interception 上到 +0.0163，Dribble 下到 -0.0126），只有 Pass 和 Ball recovery 在里面，七任务和三任务平均的区间在里面。所以它不是一个“等价在 0.006 内”的结论，审稿人若问区间会被看出来。讨论 L435 已写 seed-mean AUC，摘要和结论没写 seed-mean，可以考虑统一加上。另外 Pressure 区间 [+0.0007, +0.0099] 不含 0，统一模型在 24/25 上确实领先，和留出季 +0.0083 [0.0047, 0.0118] 同向，表 9 C3 判不确定的原因在 24/25 里已经看得到。
4. **正文 5.2 L360“On pass the CNN is behind by 0.0070 in every seed”**，区间 [-0.0083, -0.0056] 不含 0，成立。正文 5.3 和讨论里“behind ... on pass”同理（A 的 Pass 区间也不含 0）。
5. **C（卡外）**，加厚统一模型只有 Pass 的区间不含 0，按预先规则判没追平是点估计规则，和区间不冲突。

## 建议句（给改稿会话，英文，按 B 和 A 口径）

**这一节是 10:30 的第一版，已被文末“建议句第二版”取代，只留作记录，改稿以第二版为准。**

表 7 表注末尾加一句（推荐）

> Over 2,000 bootstrap resamples of the 202 test matches, the 95\% interval of the enriched CNN minus enriched XGBoost (paired over seeds 0 to 2) excludes zero only on pass ($-0.0070$, $[-0.0083, -0.0056]$), and that of the thin unified model minus enriched XGBoost (four against five seeds) excludes zero on pass ($-0.0040$, $[-0.0051, -0.0027]$) and pressure ($+0.0051$, $[+0.0007, +0.0099]$); all intervals are given in the Supplementary Material.

补充材料放一张七行两列的表（每格“点估计 [区间]”），数照上面 A、B 两表，表题建议

> Seed-mean AUC differences to enriched XGBoost on the 202 test matches of 2024/25, with 95\% bootstrap intervals over matches (2,000 resamples, the same procedure as for the held-out season).

注意 A 的 Pass 和 Pressure 点估计按原值舍入是 -0.0040 和 +0.0051，表 9 C3 的 2024/25 列和正文 5.3 写的是 -0.0039 和 +0.0052，那是表 7 两格四位均值相减（0.9155 减 0.9194、0.6099 减 0.6047）。新句子要么照原值写 -0.0040 和 +0.0051，要么为了和表 9、正文一致改写成 -0.0039 和 +0.0052 并在表注说明是表中均值之差，两种都行，不要混。这件事见下面“可能漏的”第一条。

## 没做

- 没把区间写进 tex，没改表 7、表 9 的生成脚本。
- 薄标量的对比（薄 CNN 减薄 XGBoost、薄增益）没算，卡没要，正文 5.2 第一段用到，要的话同一脚本加两行就能出。
- 自助法只重抽测试比赛，模型固定，不含训练种子之间的波动；种子波动另由表 7 的标准差和 B 的逐种子差表现。区间是百分位法，没有对七个任务做多重比较校正。

## 可能漏的

1. 表 9 C3 的 2024/25 列和正文 5.3 的“+0.0052 (pressure)”是四位均值相减得来的，原值是 +0.0051（Pass 原值 -0.0040，表 9 写 -0.0039）。来源是留出脚本 `holdout_2526_summarize.py` 的 `ref_2425()`，那里是从冻结方案抄的手写常数，冻结方案正文不能改。和 0.279 是同一类“先舍再算”的差，量级只有 0.0001，要不要动由主会话定；改的话只改表 9 显示和正文，不动冻结方案和它的阈值。
2. 种子少（加厚 CNN 只有三个），B 的点估计对种子很敏感（Interception 三个种子 0.0036 到 0.0143），场次区间里看不到这部分不确定。
3. 七个任务、两组对比一共 14 个区间，挑出“不含 0”的几个来写时，审稿人可能问多重比较；建议句只描述哪几个不含 0，不说显著。

## 建议句第二版（2026-10-06 10:35，按主会话 xcv2-6a 定的口径改，没重跑）

口径是主会话定的。表 7 表注只写区间、不写点估计，点估计读者从表 7 相减可得，这样绕开 -0.0039 对 -0.0040 那 0.0001 的差，表 9 和冻结方案的常数都不动。补充材料的表给原值点估计加区间，表注写明点估计按未舍入的种子均值算。下面的数都由 `audit/2026-10-06_paired_bootstrap_2425.json` 直接生成，没有手抄。

### 表 7 表注末尾加的一句

> Over 2,000 bootstrap resamples of the 202 test matches, the 95\% interval of the enriched CNN minus enriched XGBoost (paired over seeds 0 to 2) excludes zero only on pass ($[-0.0083, -0.0056]$), and that of the thin unified model minus enriched XGBoost (four against five seeds) excludes zero only on pass ($[-0.0051, -0.0027]$) and pressure ($[0.0007, 0.0099]$); the intervals for all tasks are given in the Supplementary Material.

### 补充材料的表

表题

> Seed-mean AUC differences to enriched XGBoost on the 202 test matches of 2024/25, with 95\% bootstrap intervals over matches.

表身（markdown 和 LaTeX 两种写法，数相同）

| Task | Events | Enriched CNN minus enriched XGBoost, seeds 0 to 2 paired | Thin unified model (4 seeds) minus enriched XGBoost (5 seeds) |
|---|---:|---|---|
| Pass | 26,482 | -0.0070 [-0.0083, -0.0056] | -0.0040 [-0.0051, -0.0027] |
| Shot | 4,929 | 0.0026 [-0.0079, 0.0131] | -0.0022 [-0.0129, 0.0085] |
| Interception | 2,605 | 0.0093 [-0.0040, 0.0221] | 0.0036 [-0.0100, 0.0163] |
| Ball recovery | 16,714 | 0.0015 [-0.0046, 0.0074] | -0.0004 [-0.0060, 0.0053] |
| Pressure | 26,484 | 0.0015 [-0.0026, 0.0059] | 0.0051 [0.0007, 0.0099] |
| Dribble | 5,060 | 0.0029 [-0.0036, 0.0098] | -0.0050 [-0.0126, 0.0027] |
| Tackle | 6,110 | 0.0039 [-0.0019, 0.0100] | 0.0002 [-0.0064, 0.0066] |

```latex
\begin{tabular}{l r c c}
\toprule
Task & Events & \makecell{Enriched CNN minus\\enriched XGBoost} & \makecell{Thin unified model minus\\enriched XGBoost} \\
Seeds & & 0 to 2, paired & 4 against 5 \\
\midrule
Pass & 26,482 & $-$0.0070 [$-$0.0083, $-$0.0056] & $-$0.0040 [$-$0.0051, $-$0.0027] \\
Shot & 4,929 & 0.0026 [$-$0.0079, 0.0131] & $-$0.0022 [$-$0.0129, 0.0085] \\
Interception & 2,605 & 0.0093 [$-$0.0040, 0.0221] & 0.0036 [$-$0.0100, 0.0163] \\
Ball recovery & 16,714 & 0.0015 [$-$0.0046, 0.0074] & $-$0.0004 [$-$0.0060, 0.0053] \\
Pressure & 26,484 & 0.0015 [$-$0.0026, 0.0059] & 0.0051 [0.0007, 0.0099] \\
Dribble & 5,060 & 0.0029 [$-$0.0036, 0.0098] & $-$0.0050 [$-$0.0126, 0.0027] \\
Tackle & 6,110 & 0.0039 [$-$0.0019, 0.0100] & 0.0002 [$-$0.0064, 0.0066] \\
\bottomrule
\end{tabular}
```

表注

> Each cell gives the difference of seed-mean AUC and, in brackets, the 2.5 and 97.5 percentiles over 2,000 resamples of the 202 test matches (random seed 20261003), drawn once and shared by all tasks and models, the same procedure as for the held-out season. The enriched CNN and enriched XGBoost are paired over their shared seeds 0 to 2; the thin unified model (seeds 0 to 3) is compared with the mean of the five enriched XGBoost seeds. Point estimates are computed from unrounded seed means, so for the unified model they can differ by 0.0001 from the difference of the rounded means in Table 7; for the enriched CNN they differ from that difference because Table 7 averages enriched XGBoost over five seeds. The intervals reflect the sampling of test matches only, not the variation between training seeds.

说明

- 表注第二句里“for the enriched CNN they differ”这一句，是因为表 7 里加厚 XGBoost 是五个种子的均值，加厚 CNN 那列直接相减得到的是本文的 B5 口径（例如 Interception，表 7 两格相减是 0.6477 减 0.6410 得 0.0067，B5 原值 0.0066），不是配对的 0.0093。表 7 现有表注末句已经解释了这一点，改稿会话若嫌重复，可以删掉这半句，只留 0.0001 那半句。
- 表 7 表注那句只写了不含 0 的三个区间，含 0 的不逐个列，指向补充表。
- 数和第一版完全相同，只换了写法。
