# TabM 复现环境与冒烟记录

记录时间：2026-11-09
执行目录：`F:\study\大三上\数据分析课设\Tabm\paper`
代码版本：`b53fd7cf2d9be2d66b918ca6cf66e0365e4ce11c`（官方仓库 `main`）

## 1. 硬件与系统

| 项目 | 值 |
| --- | --- |
| 系统 | Windows 11 家庭版 中文版（10.0.26300） |
| CPU | Intel Core Ultra 5 125H，14 核 18 线程 |
| 内存 | 31.5 GB |
| GPU | 无 CUDA（Intel Arc 核显，PyTorch 使用 CPU） |
| 线程设置 | `OMP_NUM_THREADS=14`、`MKL_NUM_THREADS=14` |

## 2. 软件环境

conda 环境：`D:\Anaconda\envs\tabm`（Python 3.10.22）

| 包 | 版本 |
| --- | --- |
| torch | 2.0.1+cpu |
| numpy | 1.26.3 |
| scipy | 1.11.3 |
| scikit-learn | 1.3.1 |
| pandas | 2.1.1 |
| loguru | 0.7.2 |
| delu | 0.0.25 |
| rtdl_num_embeddings | 0.0.11 |
| rtdl_revisiting_models | 0.0.2 |
| optuna | 3.3.0 |
| tensorboard | 2.15.2 |
| tomli / tomli-w | 2.0.1 / 1.0.0 |
| tqdm | 4.66.1 |
| typing_extensions | 4.8.0 |

`torch.cuda.is_available() == False`；训练日志中 `AMP enabled: False`。

安装命令：

```powershell
conda create -n tabm python=3.10 -y
& "D:\Anaconda\envs\tabm\python.exe" -m pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu
& "D:\Anaconda\envs\tabm\python.exe" -m pip install numpy==1.26.3 scipy==1.11.3 scikit-learn==1.3.1 pandas==2.1.1 `
  loguru==0.7.2 tqdm==4.66.1 tensorboard==2.15.2 tomli==2.0.1 tomli-w==1.0.0 optuna==3.3.0 `
  typing-extensions==4.8.0 delu==0.0.25 rtdl_num_embeddings==0.0.11 rtdl_revisiting_models==0.0.2
```

## 3. 数据

| 项目 | 值 |
| --- | --- |
| 来源 | `https://hf-mirror.com/datasets/rototoHF/tabm-data/resolve/main/data.tar`（官方 HuggingFace 数据包的镜像） |
| 文件 | `paper/data/data.tar` |
| 大小 | 1,034,147,840 字节（约 0.96 GiB） |
| SHA256 | `D8038250C20A83F241BAB822C32B71C622C032B8C25559E9AB26CAEAFCA5385D` |
| 解压位置 | `paper/data/<dataset>/`，含 `california`、`adult`、`churn`、`house` 等 |

数据集规模（来自 `info.json`）：

| 数据集 | 任务 | train / val / test |
| --- | --- | --- |
| california | regression | 13,209 / 3,303 / 4,128 |
| churn | binclass | 6,400 / 1,600 / 2,000 |
| house | regression | 14,581 / 3,646 / 4,557 |
| adult | binclass | 26,048 / 6,513 / 16,281 |

## 4. 冒烟测试结果

### 4.1 官方 debug 配置（E0 门槛）

命令：

```powershell
cd "F:\study\大三上\数据分析课设\Tabm\paper"
$env:OMP_NUM_THREADS="14"; $env:MKL_NUM_THREADS="14"
& "D:\Anaconda\envs\tabm\python.exe" bin\model.py exp\debug\0.toml --force
```

结果：**通过**（`DONE` 已生成）

| 指标 | 值 |
| --- | --- |
| 配置 | plain MLP，1 层 128，无 dropout，lr=0.0003 |
| 参数量 | 1,281 |
| 训练轮数 | 约 337 epoch（best_step 16,744） |
| 总耗时 | **24.3 秒** |
| test RMSE | 0.548722 |
| 日志 | `logs/E0_debug.log` |

结论：Windows + CPU 链路可用，数据、缓存、训练、评估、报告生成均正常。

### 4.2 官方 TabM k=32 配置（不可直接用于全矩阵）

配置：`exp/tabm/california/0-evaluation/0.toml`（k=32，3 层，d=400，`share_training_batches=false`）。

实测：约 9 分钟只到 epoch 23（约 22 秒/epoch），按早停规律预计单次 **超过 1 小时**，已终止。
结论：该配置不适合作为本机 CPU 的主实验配置，保留为「原文配置」对照，不纳入批量运行。

### 4.3 共享批次 TabM k=32（仍偏慢）

配置：`exp/tabm-sharetrainingbatches/california/0-evaluation/0.toml`（k=32，3 层，d=352）。

实测：约 10 分钟到 epoch 28（约 17 秒/epoch），预计单次 **30 分钟以上**，已终止。
结论：即使共享批次，k=32 + 3×352 仍然偏慢，需要缩减。

### 4.4 缩减配置 TabM k=8（主实验候选）

配置：`exp/reproduce/E0-tabm-k8/0.toml`

```toml
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

| 指标 | 值 |
| --- | --- |
| 参数量 | 79,944 |
| 训练轮数 | 198 epoch（best_step 9,516） |
| 总耗时 | **437.9 秒（7 分 18 秒）** |
| test RMSE | **0.474324** |
| test R² | 0.827461 |
| 日志 | `logs/E0_tabm_k8.log` |
| 输出 | `exp/reproduce/E0-tabm-k8/0/` |

与 debug MLP 对比：test RMSE 从 0.5487 降到 0.4743，差距明显大于随机波动，说明 TabM 链路与增益方向正常。

### 4.5 缩减配置 TabM k=16（对照）

配置：与 k=8 相同，仅 `k=16`。

| 指标 | 值 |
| --- | --- |
| 参数量 | 92,304 |
| 训练轮数 | 181 epoch（best_step 9,412） |
| 总耗时 | **13 分 47 秒** |
| test RMSE | **0.474932** |
| test R² | 0.827018 |
| 日志 | `logs/E0_tabm_k16.log` |
| 输出 | `exp/reproduce/E0-tabm-k16/0/` |

对比 k=8：RMSE 0.474324 vs 0.474932，差距仅 0.0006，小于种子波动量级；但训练时间几乎翻倍。初步说明在该配置下 **k 从 8 增到 16 的收益已饱和**（可作为 C3 的预实验证据，正式结论仍需多种子）。

## 5. 配置决策（E0 冻结）

| 项目 | 决策 |
| --- | --- |
| 主实验 | TabM，`k=8`、`n_blocks=2`、`d_block=256`、`share_training_batches=true`（7 分 18 秒/次） |
| 备选 | k=16（13 分 47 秒/次，精度无显著提升，不用于主矩阵） |
| 基线 | MLP / MLP†，使用官方调参配置（单次通常 <1 分钟） |
| k 分析 | {1, 2, 4, 8, 16, 32}，只跑 1 个种子；k=32 单次预计 30 分钟左右 |
| 不跑 | 官方 k=32 + 3×400 + `share_training_batches=false` 的全量配置 |
| 记录要求 | 所有缩减写入「原文-本项目差异表」，不声称逐位复现原文 Figure 3 |

**初步结论**：k=8 与 k=16 的 test RMSE 几乎相同，而 k=16 耗时翻倍；因此主实验选 k=8，把 k=16/32 留给 k 分析的饱和趋势验证。

## 6. E2 基线批量执行（2026-11-10）

| 项目 | 值 |
| --- | --- |
| 代码版本 | `38f6b0067`（分支 `experiment/e2-baselines`） |
| 脚本 | `paper/scripts/run_e2_baselines.ps1`（顺序执行、`OMP/MKL_NUM_THREADS=14`） |
| 矩阵 | `mlp` / `mlp-piecewiselinear` × `california` / `churn` / `house` × seed 0/1/2 = 18 |
| 结果 | **18/18 成功**（`exit_code=0` 且存在 `DONE`），0 失败 |
| 墙钟 | 09:15~09:54，约 39 分钟；单次 14 秒~5 分 25 秒 |
| 台账 | `Tabm/paper/results/E2/runs.csv`；汇总 `summary.csv` |
| 原始日志 | `Tabm/paper/exp/reproduce/E2-baseline/<method>/<dataset>/<seed>.log`（未提交） |
| 运行台 | `logs/E2_runner.log` |

| 方法 | california RMSE | churn Accuracy | house RMSE |
| --- | --- | --- | --- |
| MLP | 0.4968 ± 0.0084 | 0.8520 ± 0.0036 | 31005 ± 143 |
| MLP† | 0.4522 ± 0.0038 | 0.8600 ± 0.0018 | 30912 ± 299 |

执行过程中修复的两个缺陷（均为已提交的脚本问题，不是模型问题）：

1. `Set-Content -Encoding UTF8` 写入 BOM，`tomli` 拒绝解析 → 改用 `UTF8Encoding($false)`（`2b57c1ac9`）。
2. `$ErrorActionPreference = "Stop"` 把 tqdm 的 stderr 输出当作终止错误，Python 在首个 epoch 前被杀 → 调用期间临时切换为 `Continue`，并把 stderr 归一化为字符串写入日志（`38f6b0067`）。

## 7. E3 主方法批量执行（2026-11-10）

| 项目 | 值 |
| --- | --- |
| 代码版本 | `51ecb5954`（分支 `experiment/e2-baselines`） |
| 脚本 | `paper/scripts/run_e3_main.ps1`；配置 `paper/configs/E3-*.toml` |
| 矩阵 | `tabm` / `tabm-packed` × california / churn / house × seed 0/1/2 = 18 |
| 结果 | **18/18 成功**，0 失败；墙钟 10:16~11:24（68 分钟） |
| 台账 | `Tabm/paper/results/E3/runs.csv`；跨实验汇总 `Tabm/paper/results/comparison.csv` |
| 冒烟 | 两架构各跑 1 epoch（`n_epochs=1`、`k=2`）均通过，参数量 70,674 vs 136,706 |

| 数据集 | MLP | MLP† | TabM | TabM-packed |
| --- | --- | --- | --- | --- |
| california RMSE | 0.4968 | 0.4522 | 0.4711 | 0.4881 |
| churn Accuracy | 0.8520 | 0.8600 | 0.8587 | 0.8570 |
| house RMSE | 31005 | 30912 | 30798 | 30720 |

判定：C1 支持（TabM 在 3 数据集 3 种子上均优于 MLP）；C2 部分支持（参数少 6.5~6.8 倍、性能持平或更优，但 CPU 墙钟受早停点影响未体现优势，需 E5 固定步数计时）；C4 支持（集成优于最好单头）。

已知混淆：TabM 未带分段线性嵌入，MLP† 带，故 california 上 MLP† 显著更好；需补 `tabm-piecewiselinear` 条件。

## 8. E3b 嵌入混淆检验（2026-11-10）

| 项目 | 值 |
| --- | --- |
| 代码版本 | `a2e0758c2` |
| 脚本 | `paper/scripts/run_e3b_plr.ps1`；配置 `paper/configs/E3b-tabm-plr-*.toml` |
| 矩阵 | TabM+嵌入 × 三数据集 × 3 种子 = 9 |
| 结果 | 9/9 成功；12:01~12:30（29 分钟）；单次 75~492 秒 |
| 台账 | `Tabm/paper/results/E3b/runs.csv` |

| 数据集 | MLP | MLP† | TabM | TabM-packed | TabM+嵌入 |
| --- | --- | --- | --- | --- | --- |
| california RMSE | 0.4968 | 0.4522 | 0.4711 | 0.4881 | **0.4419** |
| churn Accuracy | 0.8520 | 0.8600 | 0.8587 | 0.8570 | **0.8628** |
| house RMSE | 31005 | 30912 | 30798 | **30720** | 31304 |

判定：california / churn 上「嵌入 + 集成」收益可加，C1 在控制嵌入后仍成立；house 上嵌入版本 3 个种子全部变差（−392~−583），归因于官方嵌入配置（`n_bins=2`、`dropout=0.365`、`lr=4.3e-3`）针对 4×608 骨干调参，缩减到 2×256 后失配，属配置敏感性而非集成失效。

## 9. E4 k 规模分析（2026-11-11）

| 项目 | 值 |
| --- | --- |
| 代码版本 | `bba1e5a02` |
| 脚本 | `paper/scripts/run_e4_kablation.ps1`；模板 `paper/configs/E4-tabm-*.toml` |
| 矩阵 | TabM k ∈ {1,2,4,8,16,32} × california / churn × seed=0 = 12 |
| 结果 | **12/12 成功**。churn 全部 < 8 min；california k=32 跨夜 12 小时 39 分钟 |
| 台账 | `Tabm/paper/results/E4/runs.csv`；汇总 `Tabm/paper/results/E4/summary.csv` |

| k | california RMSE | california 墙钟 | churn Accuracy | churn 墙钟 |
| --- | --- | --- | --- | --- |
| 1 | 0.48813 | 50 s | 0.85100 | 8 s |
| 2 | 0.47898 | 2:45 | 0.85900 | 30 s |
| 4 | 0.47730 | 6:48 | 0.85950 | 44 s |
| 8 | **0.47432** | 7:37 | 0.85700 | 2:15 |
| 16 | 0.47493 | 17:19 | 0.85700 | 3:49 |
| 32 | 0.47190 | **12:39:29** | 0.85950 | 7:44 |

判定：C3 部分支持——k 从 1 增到 8 的收益占主要部分（california −0.0138，churn +0.006），k=8 之后收益递减明显；california 上 k=16 比 k=8 略差，k=32 虽最好但仅再降 0.0024，且墙钟 45× 变慢，**k=8 是本机 CPU 预算下的合理截断点**。

## 10. E5 效率分解（2026-11-11）

`runs.csv` 的墙钟同时包含「每步成本」与「早停步数」，无法单独归因。为此新增固定步数微基准
`paper/scripts/benchmark_efficiency.py`（合成 california 形状负载：8 特征、16,512 行、batch 256、
14 线程、dropout=0、AdamW、30 步取中位数、预热 5 步），每种架构重复 3 次。

| 架构 | 参数量 | 每步 ms | 每 epoch 秒 | 推理（千样本/秒） |
| --- | --- | --- | --- | --- |
| plain | 68,353 | 9.12 | 0.59 | 528 |
| tabm-mini k=8 | 70,216 | 17.31 | 1.12 | 104 |
| tabm k=8 | 79,944 | 28.82 | 1.86 | 65 |
| tabm-packed k=8 | 546,824 | 20.11 | 1.30 | 89 |

表中为 3 次重复中位数；绝对耗时会随机器频率/负载漂移（两批之间 tabm 从 19.05 ms 漂到 28.82 ms），
故只解读同批次比值：TabM/packed 每步 1.43×（1.52/1.41/1.43），推理 1.36×（三次一致）。

判定：参数效率成立（TabM 比 packed 少 6.8 倍参数）；**计算效率不成立**（每步慢 1.43×、推理慢 1.36×）。
两者相对 plain 都要付数倍算力，共享权重省的是内存而非 FLOPs；packed 的 grouped linear 在 CPU 上比
TabM 的 `(B,K,D)` broadcast 更友好。故本项目不主张 Figure 4 的「更快」，只主张「同性能下少 6.8 倍参数」，
并把时间差异写入环境差异表。

## 11. E6 误差分析（2026-11-11）

| 项目 | 值 |
| --- | --- |
| 脚本 | `paper/scripts/analyze_errors.py`（只读 `predictions.npz`，不训练） |
| 输入 | E2 基线、E3 主方法、E3b 嵌入版 × 3 数据集 × 3 种子 |
| 输出 | `Tabm/paper/results/E6/`：summary、summary_by_seed、error_by_decile、calibration、pairwise_vs_tabm |
| 耗时 | 约 4 秒（纯 numpy） |

主要结论：误差随目标幅度增长但相对误差递减（house 第 1 档 47%、第 5 档 26%、第 10 档 31%）；house 全部
方法系统性低估（bias −749 ~ −1,288）；california 上 TabM 只在 53.7% 样本胜过 MLP，但 MSE 低 11%（误差相关
0.96），收益是全局小幅改善；churn 的 ECE 从 MLP 0.0376 降至 TabM 0.0227、TabM + 嵌入 0.0176，
**集成的价值在概率校准上比准确率更明显**。

作图：新增依赖 `matplotlib 3.10.9`（仅用于出图，不参与训练），脚本 `paper/scripts/plot_e6.py` 只读
`results/E6` 的 CSV 与 `results/comparison.csv`，输出 `results/E6/figures/` 下 3 张 200 dpi PNG
（十分位误差、churn 可靠性图、参数量-RMSE 散点）。

## 12. E7 独立实现核验（2026-11-11）

| 项目 | 值 |
| --- | --- |
| 脚本 | `paper/scripts/verify_implementation.py`（手写适配器层，不导入官方集成层） |
| 输出 | `Tabm/paper/results/E7/verification.csv` |
| 回归测试 | `tests/test_independent_implementation.py`（3 个用例，随 CI 运行） |
| 耗时 | 约 1 秒 |

结果：前向等价 k=1/2/4/8 的最大绝对差**均为 0.0**（逐位相同）；参数量闭式与官方枚举完全一致
（k=1 → 438，k=8 → 1,152）；把 r、s 置 1 且 bias 取共享值后，k=1 的 TabM 与普通 MLP 输出完全相同。
说明本项目对 TabM 机制的实现与论文/官方代码一致，E2~E6 的结论有独立代码路径支撑。

## 13. E9 官方背景分布定位（2026-11-11）

| 项目 | 值 |
| --- | --- |
| 脚本 | `paper/scripts/locate_effect_size.py` |
| 官方样本 | 本地仓库自带 `exp/{tabm,mlp}/<dataset>/0-evaluation/0/report.json`，共 10 个数据集 |
| 输出 | `Tabm/paper/results/E9/`：official_effects.csv、our_position.csv、effect_size_distribution.png |
| 耗时 | 约 2 秒 |

效应量定义：回归 `(mlp-tabm)/mlp` 相对误差下降；分类 `tabm-mlp` 准确率差。
官方回归 5 个数据集（中位 +4.20%，范围 +0.64%~+10.33%）、分类 5 个（中位 +0.74pp，范围 +0.21~+2.16pp），
**10/10 全部为正**。我们的位置：california +5.18%（60 百分位）、churn +0.67pp（40）、house +0.67%（20）。
结论：方向一致，幅度处于官方分布中低区间且与预算削减相符；house 衰减最大（我们的 MLP 略强、TabM 被削减）。
本地数据包只有 10 个数据集，报告中不得声称覆盖论文的 46 个。

