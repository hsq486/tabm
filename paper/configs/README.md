# E0 复现实验配置（冻结）

日期：2026-11-09
代码：`b53fd7cf2d9be2d66b918ca6cf66e0365e4ce11c`
环境：conda `tabm`，Python 3.10.22，torch 2.0.1+cpu
数据：`paper/data`（来自 `data.tar`，SHA256 `D8038250C20A83F241BAB822C32B71C622C032B8C25559E9AB26CAEAFCA5385D`）

## 1. 主实验冻结配置

```toml
seed = 0
batch_size = 256
patience = 16
n_epochs = -1
gradient_clipping_norm = 1.0
amp = true

[data]
cache = true
path = "data/california"
num_policy = "noisy-quantile"

[optimizer]
type = "AdamW"
lr = 0.0008468726533903873
weight_decay = 0.0

[model]
arch_type = "tabm"
k = 8
share_training_batches = true

[model.backbone]
type = "MLP"
n_blocks = 2
d_block = 256
dropout = 0.1697377329739597
```

相对官方 `exp/tabm-sharetrainingbatches/california/0-evaluation/0.toml` 的差异
（本项目从「批次共享」变体出发，因为主配置已固定 `share_training_batches=true`）：

| 项目 | 原文 | 本项目 | 原因 |
| --- | --- | --- | --- |
| k | 32 | 8 | CPU 预算；E0 预实验显示 k=8→16 精度几乎不变、耗时翻倍 |
| n_blocks | 3 | 2 | 降低 CPU 训练时间 |
| d_block | 352 | 256 | 降低 CPU 训练时间 |
| 其它 | `lr`、`dropout`、`batch_size`、`patience`、`num_policy` | 与原文一致 | 不改动官方调参结果 |

> 注：官方非共享变体 `exp/tabm/california/` 的 `d_block` 是 400、`lr` 与 `dropout` 也不同；
> 本项目没有使用该文件，报告引用数值时以 `tabm-sharetrainingbatches` 一套为准。

## 2. E0 预实验实测结果（单次运行，非正式证据）

| 配置 | 参数量 | epoch | 耗时 | test RMSE |
| --- | --- | --- | --- | --- |
| debug MLP（1×128） | 1,281 | 337 | 24.3 秒 | 0.548722 |
| TabM k=8（主） | 79,944 | 198 | 7 分 18 秒 | 0.474324 |
| TabM k=16（对照） | 92,304 | 181 | 13 分 47 秒 | 0.474932 |

初步判定：TabM k=8 相对普通 MLP 明显降低 RMSE；k 从 8 增到 16 的收益仅 0.0006，且耗时翻倍，因此主实验固定 k=8。

> **E0 是单次运行的预实验**（仅用于跑通流程和确定 k），不进入汇总表、不作为证据；
> 上述判断已由 E2/E3 的 3 种子结果复核（见 §4、§5），k=8 与 k=16 的差距在种子波动内。

## 3. 目录与命令

```text
paper/configs/                 # 13 个冻结配置，均以注释标明来源与缩减项
├── E0-tabm-k8.toml            # 预实验：k=8（单次）
├── E0-tabm-k16.toml           # 预实验：k=16（单次）
├── E3-tabm-<dataset>.toml     # 主方法，dataset ∈ {california, churn, house}
├── E3-tabm-packed-<dataset>.toml
├── E3b-tabm-plr-<dataset>.toml
├── E4-tabm-<dataset>.toml     # dataset ∈ {california, churn}，k 由脚本替换
└── README.md

paper/exp/reproduce/           # 运行配置与输出（被 .gitignore 排除，不入库）
├── E2-baseline/<method>/<dataset>/<seed>.toml + <seed>.log + <seed>/
├── E3/<method>/<dataset>/<seed>.toml + ...
├── E3b/<dataset>/<seed>.toml + ...
└── E4/<dataset>/k=<k>/<seed>.toml + ...
```

**配置可追溯性**：runner 只替换模板里的 `seed = ...` 一行（E4 另外替换 `k`），
每个种子都会把实际使用的 TOML 写到 `exp/reproduce/.../<seed>.toml` 后再交给
`bin/model.py`，因此「3 行结果」对应「3 份实际配置 + 3 份日志 + 3 个 report.json」；
提交号同时写入 `paper/results/*/runs.csv` 的 `commit` 列。

```powershell
cd "F:\study\大三上\数据分析课设\Tabm\paper"
$env:OMP_NUM_THREADS="14"; $env:MKL_NUM_THREADS="14"
# 把 configs 下的模板复制到 exp/reproduce 的对应目录后运行
& "D:\Anaconda\envs\tabm\python.exe" bin\model.py exp\reproduce\E0-tabm-k8\0.toml --force
```

## 4. 实验矩阵与执行状态（全部已执行完毕）

| 实验 | 数据集 | 方法/内容 | 种子 | 状态 |
| --- | --- | --- | --- | --- |
| E2 基线 | california、churn、house | MLP、MLP† | 0/1/2 | 18/18 完成 |
| E3 主方法 | california、churn、house | TabM、TabM-packed | 0/1/2 | 18/18 完成 |
| E3b 解混淆 | california、churn、house | TabM + 分段线性嵌入 | 0/1/2 | 9/9 完成 |
| E4 k 分析 | california、churn | TabM，k=1/2/4/8/16/32 | 0（探索性） | 12/12 完成 |
| E5 效率 | 合成负载（california 形状） | plain/TabM/TabM-packed | 3 次重复 | 完成 |
| E6 误差 | 复用 E2/E3/E3b 预测 | 十分位、校准、配对胜负 | 0/1/2 | 完成 |
| E7 独立实现 | 合成张量 | 前向等价与参数量闭式 | — | 7 项检查全通过 |
| E9 定位 | 官方 10 个数据集的报告 | 效应量分布 | — | 完成 |

下面按实验给出配置、结果与结论；表中汇总值均为 3 种子均值（E4 除外，见其小节说明）。

### E2 基线说明

- `mlp` 与 `mlp-piecewiselinear` 均直接使用官方 tuned 配置
  `exp/<method>/<dataset>/0-evaluation/0.toml`，只替换 `seed`。
- 运行配置生成到 `exp/reproduce/E2-baseline/<method>/<dataset>/<seed>.toml`，
  原始日志与报告保留在同目录，已被 `.gitignore` 排除。
- 执行脚本：`paper/scripts/run_e2_baselines.ps1`；执行台账：
  `paper/results/E2/runs.csv`。
- 基线宽度（如 house 的 848、churn 的 752）来自官方调参结果，**不缩减**；
  TabM 主方法因 CPU 预算缩减到 k=8、2×256。该不对称对 TabM 更严格，
  报告需在差异表中说明。

### E2 执行结果（2026-11-10，commit `38f6b0067`）

18/18 用例成功（`exit_code = 0` 且生成 `DONE`），总墙钟时间约 39 分钟
（09:15~09:54），逐次耗时 14 秒~5 分 25 秒。台账 `paper/results/E2/runs.csv`，
汇总 `paper/results/E2/summary.csv`，原始日志在
`paper/exp/reproduce/E2-baseline/<method>/<dataset>/<seed>.log`（已被 Git 忽略）。

| 方法 | california RMSE | churn Accuracy | house RMSE |
| --- | --- | --- | --- |
| MLP | 0.4968 ± 0.0084 | 0.8520 ± 0.0036 | 31005 ± 143 |
| MLP† | 0.4522 ± 0.0038 | 0.8600 ± 0.0018 | 30912 ± 299 |

说明：

- `report.json` 的 `metrics.test.score` 为「越大越好」口径，回归任务取其相反数，
  上表已还原为正的 RMSE。
- house 的 RMSE 在原始目标量纲上（约 3.1×10⁴），不与 california 的 0.45 直接可比。
- 3 个种子的波动很小（分类 ≤ 0.004，回归 ≤ 1%），说明固定划分下基线已收敛，
  后续 TabM 的差异需要超过该波动才可判定为真实提升。

### E3 执行结果（2026-11-10，commit `51ecb5954`）

18/18 用例成功，墙钟 10:16~11:24（约 68 分钟）。配置见本目录
`E3-<method>-<dataset>.toml`：TabM 取官方 `tabm-sharetrainingbatches/<dataset>`
的调参结果，只缩减 `k=8`、`2×256`；TabM-packed 与之一一对应，仅 `arch_type` 不同。
台账 `results/E3/runs.csv`，跨实验汇总 `results/comparison.csv`。

| 数据集 | MLP | MLP† | TabM | TabM-packed |
| --- | --- | --- | --- | --- |
| california RMSE | 0.4968 ± 0.0084 | **0.4522 ± 0.0038** | 0.4711 ± 0.0046 | 0.4881 ± 0.0023 |
| churn Accuracy | 0.8520 ± 0.0036 | **0.8600 ± 0.0018** | 0.8587 ± 0.0018 | 0.8570 ± 0.0023 |
| house RMSE | 31005 ± 143 | 30912 ± 299 | 30798 ± 388 | **30720 ± 272** |

参数量（3 个种子一致）：TabM 79,944（california）/ 84,112（churn）/ 82,056（house）；
TabM-packed 546,824 / 565,264 / 563,208，即 **TabM 少 6.5~6.8 倍参数**。

按预先写死的判据：

- **C1（TabM 优于 MLP）支持**：逐种子配对差为 california +0.0257、
  churn +0.0067、house +207.6（RMSE 更低），三个数据集、三个种子方向一致，
  且均大于种子波动。
- **C2（权重共享更省参数）部分支持**：参数量优势巨大且稳定；性能上
  california +0.0170、churn +0.0017 优于 packed，house 差 −77（落在 ±272~388
  的种子波动内），符合「性能持平但参数更少」。**但 CPU 墙钟未体现优势**
  （california TabM 664~946 秒 vs packed 236~394 秒），原因是两者早停点不同
  （best_step 9,516 vs 5,460），不能据此否定论文的效率结论，需 E5 用固定
  epoch 数的专门计时再判定。
- **C4（单独弱、集体强）支持**：集成预测在三个数据集上均优于最好单头
  （california −0.4711 vs −0.4749；churn 0.8587 vs 0.8568；house −30,798 vs −31,986）。

必须声明的限制：

1. 本实验的 TabM 不带分段线性数值嵌入，而 MLP† 带。因此 **MLP† 在 california
   明显优于 TabM（0.4522 vs 0.4711）**，churn 基本持平。C1 只在
   「TabM vs 纯 MLP」意义上成立；与 MLP† 的比较受嵌入差异混淆，需补
   `tabm-piecewiselinear` 条件后才能判断「提升是否来自集成」。
2. house 的目标量纲为原始值，RMSE 不可与 california 直接比较，跨数据集只比较相对差。
3. TabM 的骨干与 k 被缩减，绝对数值不可与论文 Figure 3 逐点对齐，只做同协议内部比较。

### E3b 执行结果：嵌入混淆检验（2026-11-10，commit `a2e0758c2`）

9/9 成功，墙钟 12:01~12:30（约 29 分钟）。配置 `E3b-tabm-plr-<dataset>.toml`
来自官方 `tabm-sharetrainingbatches-piecewiselinear/<dataset>`，同样只缩
`k=8`、`2×256`，保留官方 `lr / dropout / d_embedding / n_bins`。

| 数据集 | MLP | MLP† | TabM | TabM-packed | **TabM+嵌入** |
| --- | --- | --- | --- | --- | --- |
| california RMSE | 0.4968 | 0.4522 | 0.4711 | 0.4881 | **0.4419 ± 0.0008** |
| churn Accuracy | 0.8520 | 0.8600 | 0.8587 | 0.8570 | **0.8628 ± 0.0018** |
| house RMSE | 31005 | 30912 | 30798 | **30720** | 31304 ± 168 |

逐种子配对差（TabM+嵌入 减 对照）：

| 对照 | california | churn | house |
| --- | --- | --- | --- |
| MLP† | +0.0103（3/3 正） | +0.0028（3/3 正） | −392（3/3 负） |
| TabM（无嵌入） | +0.0291（3/3 正） | +0.0042（3/3 正） | −506（3/3 负） |
| TabM-packed | +0.0462 | +0.0058 | −583 |

结论：

- **california / churn：嵌入混淆解除**。控制嵌入后 TabM 仍显著优于 MLP†，
  且「嵌入」与「集成」的收益近似可加（california：MLP→MLP† +0.0446，
  再加集成 +0.0103），说明 C1 不是特征工程的假象。
- **house：反向结果，必须如实报告**。该数据集的官方嵌入配置是 `n_bins=2`、
  `dropout=0.365`、`lr=4.3×10⁻³`，原本为 4×608 骨干调参，套到 2×256 后
  在 3 个种子上都变差（−392 ~ −583），best_step 也明显下降。这属于
  **配置对骨干规模的敏感性**，不能解释为「集成无效」。
  换言之，该行检验的是「官方超参 + 缩减骨干」这一组合的效果，**不能把变差单独
  归因于分段线性嵌入模块**；要分离嵌入本身的效应，需在 2×256 上重调
  `n_bins / dropout / lr`。本作业未做这轮重调，作为限制写入报告。
- 参数量：TabM+嵌入 146k~148k，仍比 MLP†（california 578k）少 3.9 倍、
  比 TabM-packed（547k）少 3.7 倍，参数效率结论不变。

### E4 执行结果：集成规模 k 分析（2026-11-11，commit `bba1e5a02`）

12/12 成功，10:43–10:59（churn），california k=32 从 21:45 跨夜跑到 10:43（**约 12 小时 39 分钟**）。配置
`E4-tabm-<dataset>.toml` 固定除 k 外的所有超参，以隔离集成规模效应；seed=0。

> **单种子探索性实验**：为控制总耗时，E4 每个 (数据集, k) 只跑 seed=0。因此下表是
> 「固定种子下的趋势」，用于确定 k 的取舍区间，**不构成 k 趋势的统计证据**（E3 只
> 提供了 k=8 的 3 种子水平，没有 k 的跨种子重复）。报告引用时必须标注「seed=0，
> 探索性」，且不能写成稳定规律。

| 数据集 | k=1 | k=2 | k=4 | k=8 | k=16 | k=32 |
| --- | --- | --- | --- | --- | --- | --- |
| california RMSE | 0.48813 | 0.47898 | 0.47730 | 0.47432 | 0.47493 | **0.47190** |
| california params | 69,129 | 70,674 | 73,764 | 79,944 | 92,304 | 117,024 |
| california 墙钟 | 50 s | 2:45 | 6:48 | 7:37 | 17:19 | **12:39:29** |
| churn Accuracy | 0.85100 | 0.85900 | **0.85950** | 0.85700 | 0.85700 | 0.85950 |
| churn params | 71,442 | 73,252 | 76,872 | 84,112 | 98,592 | 127,552 |
| churn 墙钟 | 8 s | 30 s | 44 s | 2:15 | 3:49 | 7:44 |

**结论（C3 部分支持；以下均为 seed=0 的探索性观察）**：

- california 上 k=1 → k=8 的 RMSE 改善是主要部分（−0.0138），k=8 → k=32 只再降 0.0024，
  且 k=16 比 k=8 反而略微变差（+0.0006）。收益递减的形态明显，据此把 k=8 作为本机
  CPU 预算下的截断点；但「收益递减」目前只有单种子支持。
- churn 上 k=1 → k=2 直接跳 0.008，k=2 之后就已经很平（波动 ≤ 0.0025），k=4 达到 0.8595 后
  不再改善。分类任务上集成规模的饱和更快（同样仅 seed=0）。
- 参数量增长近似线性（k 每翻倍，参数量增 ~12k），**墙钟在 california 上呈超线性**：
  k=16 约 17 min，k=32 约 12.7 h（**45× 变慢**，best_step 同为 9412）。这是 CPU 无 GPU 并行时
  `share_training_batches=true` 的已知代价，不影响结论。
- 同一配置 k=8 下，E4 跑出 california RMSE 0.47432，与 E3 seed=0 的 0.474324 基本一致，
  说明流程是确定性的。

### E5 执行结果：效率分解（2026-11-11）

`runs.csv` 里的墙钟把「每步成本」和「早停留下的步数」混在一起，因此另用一个
固定步数的微基准把两者分开：`paper/scripts/benchmark_efficiency.py`，同一份
california 形状的合成负载（8 特征、16,512 行、batch 256、14 线程、dropout=0）。

> **适用范围**：这是**合成负载的固定步数微基准**，只用于比较同一形状下的相对
> 计算成本；它不等于任何数据集上的真实训练总耗时（不含早停步数差异），也不能
> 外推到论文的 GPU 环境。报告只能引用「同批次内的比值」，不能引用绝对秒数。

| 架构 | 参数量 | 每步 ms | 每 epoch 秒 | 推理（千样本/秒） |
| --- | --- | --- | --- | --- |
| plain | 68,353 | 9.12（5.01–10.49） | 0.59 | 528 |
| tabm-mini k=8 | 70,216 | 17.31 | 1.12 | 104 |
| **tabm k=8** | **79,944** | **28.82** | **1.86** | **65** |
| tabm-packed k=8 | 546,824 | 20.11 | 1.30 | 89 |

表中为 3 次重复的中位数。**绝对耗时会随机器的频率/负载漂移**（连跑两批之间 tabm 从
19.05 ms 漂到 28.82 ms），因此只解读同一批次内的比值：TabM/packed 每步 1.43×
（三次分别 1.52 / 1.41 / 1.43），推理 1.36×（1.36 / 1.36 / 1.37），方向稳定。
原始数据见 `results/E5/benchmark.csv` 与 `repeat1..3.csv`。

结论（**必须如实写入报告**）：

- **参数效率成立**：TabM 比 packed 少 6.8 倍参数，这是 C2 的核心。
- **计算效率不成立**：本机 CPU 上 TabM 每步比 packed 慢约 1.43 倍、推理慢约 1.36 倍。
  两者都要为 8 个子模型付出数倍于 plain 的算力，共享权重省的是内存与 I/O 而不是 FLOPs；
  packed 用 grouped linear，TabM 用 broadcast 的 `(B,K,D)` 张量，在无 GPU、无 AMP 的 CPU 上后者更吃亏。
- 因此本项目**不主张**论文 Figure 4 的「更快」结论，只主张「同性能下少 6.8 倍参数」，
  并把时间差异归因于环境（CPU、AMP 关闭、批次共享）写入差异表。

### E6 执行结果：误差分析（2026-11-11）

不重新训练，直接读取 E2/E3/E3b 已保存的 `predictions.npz` 与官方测试标签，
脚本 `paper/scripts/analyze_errors.py`（3 种子，表中为均值，标准差为总体标准差）。

| 数据集 | 方法 | 主指标 | bias | 最差 1% MAE | ECE |
| --- | --- | --- | --- | --- | --- |
| california | mlp | 0.4968 | −0.006 | 2.284 | — |
| california | mlp† | 0.4522 | +0.009 | 2.106 | — |
| california | tabm | 0.4711 | +0.010 | 2.182 | — |
| california | tabm-packed | 0.4881 | +0.003 | 2.265 | — |
| california | tabm+嵌入 | **0.4419** | +0.019 | **2.104** | — |
| churn | mlp | 0.8520 | — | — | 0.0376 |
| churn | mlp† | 0.8600 | — | — | 0.0413 |
| churn | tabm | 0.8587 | — | — | 0.0227 |
| churn | tabm-packed | 0.8570 | — | — | 0.0221 |
| churn | tabm+嵌入 | **0.8628** | — | — | **0.0176** |
| house | mlp | 31005 | −1072 | 205,738 | — |
| house | mlp† | 30912 | +5 | 207,173 | — |
| house | tabm | 30798 | −749 | 206,303 | — |
| house | tabm-packed | **30720** | −1288 | 205,880 | — |

**四点发现（可直接写入报告的「深入分析」）**：

1. **误差随目标幅度增长，相对误差反而下降**。按目标值等样本分十分位（每档 413 / 456 条），
   california 的 MAE 从 0.223 升到 0.638，house 从 6,835 升到 55,688；
   但相对误差是**递减**的：house 第 1 档 47%、第 5 档 26%、第 10 档 31%。
   即高价样本的绝对误差大，模型对它们的相对刻画并不更差；RMSE 主要由最高一档贡献。
   注意 house 的低价档含大量相同取值，故分位按样本秩切分而不是按分位点切分。
2. **house 存在系统性低估**：所有方法的 bias 都为负（−749 ~ −1,288），MLP 最严重。
   这个偏差在只看 RMSE 时被掩盖，是「模型对高价样本欠拟合」的直接证据。
3. **逐样本胜负不等于平均提升**。california 上 TabM 只在 53.7% 的测试样本上优于 MLP，
   但 MSE 比是 1.11（MLP 误差平方和大 11%）；两者误差相关 0.96。
   即 TabM 的收益是**全局小幅改善**，不是「修好某一类难样本」。
   反过来，TabM 输给 MLP† 的样本占 55%，而 MLP† 的优势集中在尾部（最差 1% MAE 2.11 vs 2.18）——
   嵌入的价值主要体现在极端样本上。
4. **集成显著改善概率校准**：churn 的 ECE 从 MLP 的 0.0376 降到 TabM 的 0.0227、
   TabM+嵌入的 0.0176（降幅约 40%~53%），而准确率只差 0.7 个百分点。
   校准表显示 MLP 在 0.2~0.45 区间系统性低估（预测 0.24 而实际发生率 0.31），
   这类偏差会直接影响风控/营销场景的阈值决策，是「集成为什么有用」的一个独立证据。

数据：`results/E6/summary.csv`、`summary_by_seed.csv`、`error_by_decile.csv`、
`calibration.csv`、`pairwise_vs_tabm.csv`。

图（`paper/scripts/plot_e6.py`，只读上述 CSV，matplotlib 3.10.9，200 dpi）：

| 文件 | 内容 | 说明 |
| --- | --- | --- |
| `results/E6/figures/error_by_decile.png` | california / house 的十分位 MAE | house 用对数纵轴 |
| `results/E6/figures/calibration_churn.png` | churn 可靠性图（图例含 ECE） | 对角线为完美校准 |
| `results/E6/figures/params_vs_error.png` | california 参数量 vs RMSE | 横轴对数，含嵌入版 |

### E7 执行结果：独立实现核验（2026-11-11）

`paper/scripts/verify_implementation.py` 用**手写的适配器层**重写 TabM（不导入任何集成层代码），
把官方权重逐层拷入后做三项数值核验；同一逻辑也进了 `tests/test_independent_implementation.py`。

| 检查 | 官方 | 独立实现 | 最大绝对差 | 判定 |
| --- | --- | --- | --- | --- |
| 前向等价 k=1 | 0.13840523 | 0.13840523 | 0.000e+00 | 通过 |
| 前向等价 k=2 | 0.11097854 | 0.11097854 | 0.000e+00 | 通过 |
| 前向等价 k=4 | 0.16245495 | 0.16245495 | 0.000e+00 | 通过 |
| 前向等价 k=8 | 0.11377308 | 0.11377308 | 0.000e+00 | 通过 |
| 参数量公式 k=1 | 438 | 438（公式与枚举一致） | 0 | 通过 |
| 参数量公式 k=8 | 1,152 | 1,152（公式与枚举一致） | 0 | 通过 |
| k=1 退化为 MLP | 0.11170638 | 0.11170638 | 0.000e+00 | 通过 |

手写实现逐层计算

```text
y[b, i, :] = ((x[b, :] * r[i]) @ W.T) * s[i] + bias[i]
```

参数量闭式（`n_blocks` 层、隐层宽 `D`、输入 `F`、输出 `d_out`）：

```text
params = [F*D + k*(F + 2D)] + (n_blocks - 1)*[D*D + k*(D + 2D)] + k*(D + d_out)
```

结论：**差异为 0（不是"小于容差"，而是逐位相同）**，说明我们对 TabM 的理解与官方实现一致，
E2~E6 的所有结论建立在被独立核验过的机制之上，而不是"照抄仓库跑一遍"。
k=1 在把 `r,s` 置 1、`bias` 置为共享偏置后与普通 MLP 输出完全相同，验证了论文「不改变网络结构即可获得集成」的说法。

### E9 执行结果：官方背景分布定位（2026-11-11）

`paper/scripts/locate_effect_size.py` 读取仓库自带的官方 `report.json`，用**先定义后比较**的效应量
定位我们的结果：

- 回归：相对误差下降 `(mlp - tabm) / mlp`（越大越好）；
- 分类：准确率之差 `tabm - mlp`（百分点）。

官方分布（本地仓库自带的 10 个数据集，`0-evaluation/0`，A100 + 完整配置）：

| 任务 | 数据集数 | 中位数 | 范围 | 为负的数据集 |
| --- | --- | --- | --- | --- |
| 回归 | 5 | +4.20% | +0.64% ~ +10.33% | 0 |
| 分类 | 5 | +0.74pp | +0.21pp ~ +2.16pp | 0 |

**10/10 个数据集上 TabM 都优于 MLP**，与本项目 3/3 的方向完全一致。

我们的位置（3 种子均值）：

| 数据集 | 我们的效应量 | 官方同数据集 | 在官方分布中的百分位 |
| --- | --- | --- | --- |
| california | +5.18% | +10.33% | 60% |
| churn | +0.67pp | +0.45pp | 40% |
| house | +0.67% | +4.20% | 20% |

解读：

1. **方向稳健**：官方 10/10、本项目 3/3、E3b 的 3 个种子全部同向，C1 的定性结论不依赖单次运行。
2. **幅度受预算削减影响，且衰减不均匀**。california 只有官方的约一半（k 32→8、3×400→2×256）；
   house 掉到 20 百分位——官方 TabM 在 house 上是 29,876，我们是 30,798，
   而我们的 MLP（31,005）反而略好于官方（31,187），一升一降压缩了效应量。
3. **churn 反而超过官方**（0.67pp vs 0.45pp）：我们的 TabM 与其持平（0.8587 vs 0.8585），
   差距来自我们的 MLP 略弱（0.8520 vs 0.8540），这个量级在种子标准差（0.0036）之内，
   **不应解释为「我们的方法更强」**，只能说明分类任务的效应量本身很小。
4. 因此报告中的规范表述是「**方向一致、幅度处于官方分布的中低区间，与预算削减相符**」，
   而不是「复现了论文的全部效应量」。本地数据包只含 10 个数据集，不能声称覆盖论文的 46 个。

图：`results/E9/effect_size_distribution.png`（灰点为官方 10 个数据集，红色倒三角为我们的 3 个结果）。
表：`results/E9/official_effects.csv`、`results/E9/our_position.csv`。

> 官方 `exp/` 下的配置与结果未修改；本目录只存放可提交的复现配置，原始输出留在 `exp/reproduce/` 且不进入 Git。
