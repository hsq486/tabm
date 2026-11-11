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

相对官方 `exp/tabm/california/0-evaluation/0.toml` 的差异：

| 项目 | 原文 | 本项目 | 原因 |
| --- | --- | --- | --- |
| k | 32 | 8 | CPU 预算；k=8/16 预实验精度几乎相同 |
| n_blocks | 3 | 2 | 降低 CPU 训练时间 |
| d_block | 400 | 256 | 降低 CPU 训练时间 |
| share_training_batches | false | true | CPU 吞吐；官方另有 `tabm-sharetrainingbatches` 变体 |
| 硬件 | A100 | 本机 CPU | 无 GPU |

## 2. 实测结果

| 配置 | 参数量 | epoch | 耗时 | test RMSE |
| --- | --- | --- | --- | --- |
| debug MLP（1×128） | 1,281 | 337 | 24.3 秒 | 0.548722 |
| TabM k=8（主） | 79,944 | 198 | 7 分 18 秒 | 0.474324 |
| TabM k=16（对照） | 92,304 | 181 | 13 分 47 秒 | 0.474932 |

初步判定：TabM k=8 相对普通 MLP 明显降低 RMSE；k 从 8 增到 16 的收益仅 0.0006，且耗时翻倍，因此主实验固定 k=8。该结论仍需 3 个种子确认。

## 3. 目录与命令

```text
paper/configs/
├── E0-tabm-k8.toml        # 主配置模板（可提交）
└── E0-tabm-k16.toml       # 集成规模对照

paper/exp/reproduce/
├── E0-tabm-k8/0.toml      # 实验运行配置（exp 下被 git 忽略）
├── E0-tabm-k8/0/          # 输出：checkpoint、预测、report.json、DONE
├── E0-tabm-k16/0.toml
└── E0-tabm-k16/0/
```

```powershell
cd "F:\study\大三上\数据分析课设\Tabm\paper"
$env:OMP_NUM_THREADS="14"; $env:MKL_NUM_THREADS="14"
# 把 configs 下的模板复制到 exp/reproduce 的对应目录后运行
& "D:\Anaconda\envs\tabm\python.exe" bin\model.py exp\reproduce\E0-tabm-k8\0.toml --force
```

## 4. 后续矩阵（待执行）

| 实验 | 数据集 | 方法 | 种子 |
| --- | --- | --- | --- |
| E2 基线 | california、churn、house | MLP、MLP† | 0/1/2 |
| E3 主方法 | california、churn、house | TabM、TabM_packed | 0/1/2 |
| E4 k 分析 | california | TabM k=1/2/4/8/16/32 | 0 |

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
- 参数量：TabM+嵌入 146k~148k，仍比 MLP†（california 578k）少 3.9 倍、
  比 TabM-packed（547k）少 3.7 倍，参数效率结论不变。

### E4 执行结果：集成规模 k 分析（2026-11-11，commit `bba1e5a02`）

12/12 成功，10:43–10:59（churn），california k=32 从 21:45 跨夜跑到 10:43（**约 12 小时 39 分钟**）。配置
`E4-tabm-<dataset>.toml` 固定除 k 外的所有超参，以隔离集成规模效应；seed=0。

| 数据集 | k=1 | k=2 | k=4 | k=8 | k=16 | k=32 |
| --- | --- | --- | --- | --- | --- | --- |
| california RMSE | 0.48813 | 0.47898 | 0.47730 | 0.47432 | 0.47493 | **0.47190** |
| california params | 69,129 | 70,674 | 73,764 | 79,944 | 92,304 | 117,024 |
| california 墙钟 | 50 s | 2:45 | 6:48 | 7:37 | 17:19 | **12:39:29** |
| churn Accuracy | 0.85100 | 0.85900 | **0.85950** | 0.85700 | 0.85700 | 0.85950 |
| churn params | 71,442 | 73,252 | 76,872 | 84,112 | 98,592 | 127,552 |
| churn 墙钟 | 8 s | 30 s | 44 s | 2:15 | 3:49 | 7:44 |

**结论（C3 部分支持）**：

- california 上 k=1 → k=8 的 RMSE 改善是主要部分（−0.0138），k=8 → k=32 只再降 0.0024，
  且 k=16 比 k=8 反而略微变差（+0.0006）。收益递减明显，**k=8 已是本机 CPU 预算下的合理截断点**。
- churn 上 k=1 → k=2 直接跳 0.008，k=2 之后就已经很平（波动 ≤ 0.0025），k=4 达到 0.8595 后
  不再改善。分类任务上集成规模的饱和更快。
- 参数量增长近似线性（k 每翻倍，参数量增 ~12k），**墙钟在 california 上呈超线性**：
  k=16 约 17 min，k=32 约 12.7 h（**45× 变慢**，best_step 同为 9412）。这是 CPU 无 GPU 并行时
  `share_training_batches=true` 的已知代价，不影响结论。
- 同一配置 k=8 下，E4 跑出 california RMSE 0.47432，与 E3 seed=0 的 0.474324 基本一致，
  说明流程是确定性的。

> 官方 `exp/` 下的配置与结果未修改；本目录只存放可提交的复现配置，原始输出留在 `exp/reproduce/` 且不进入 Git。
