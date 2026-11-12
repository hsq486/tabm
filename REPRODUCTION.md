# 复现说明（REPRODUCTION）

本文件说明如何在**一台没有 GPU 的 Windows 机器**上，从零复现本作业的全部实验与结论。
所有命令均在 `paper/` 目录下执行，路径以本项目实际位置为例，换机器时只需替换绝对路径。

- 论文：*TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling*（ICLR 2025）
- 上游仓库：<https://github.com/yandex-research/tabm>
- 复现基线 commit：`b53fd7cf2d9be2d66b918ca6cf66e0365e4ce11c`（2025-11-07）
- 本作业分支：`experiment/e2-baselines`，相对基线共 21 个提交
- 实验总次数：**57 次训练**（E2 18 + E3 18 + E3b 9 + E4 12）+ 5 个只读分析脚本（E5/E6×2/E7/E9）

---

## 1. 环境

### 1.1 硬件

| 项目 | 值 |
| --- | --- |
| 系统 | Windows 11（10.0.26300） |
| CPU | Intel Core Ultra 5 125H，14 核 18 线程 |
| 内存 | 31.5 GB |
| GPU | 无 CUDA（`torch.cuda.is_available() == False`） |
| 线程 | `OMP_NUM_THREADS=14`、`MKL_NUM_THREADS=14` |

### 1.2 Python 环境

conda 环境名 `tabm`，Python 3.10.22：

```powershell
conda create -n tabm python=3.10 -y
$py = "D:\Anaconda\envs\tabm\python.exe"
& $py -m pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu
& $py -m pip install numpy==1.26.3 scipy==1.11.3 scikit-learn==1.3.1 pandas==2.1.1 `
  loguru==0.7.2 tqdm==4.66.1 tensorboard==2.15.2 tomli==2.0.1 tomli-w==1.0.0 `
  optuna==3.3.0 typing-extensions==4.8.0 delu==0.0.25 `
  rtdl_num_embeddings==0.0.11 rtdl_revisiting_models==0.0.2
# 仅 E6/E9 出图与 lint 需要，不是训练依赖
& $py -m pip install matplotlib==3.10.9 ruff pytest
```

关键版本：torch 2.0.1+cpu、numpy 1.26.3、scikit-learn 1.3.1、pandas 2.1.1、
tomli 2.0.1、matplotlib 3.10.9。`amp = true` 在无 CUDA 时自动降级，日志显示
`AMP enabled: False`，因此所有结果都是 FP32 CPU 结果。

### 1.3 数据

```powershell
cd paper
# 官方数据包（HuggingFace 镜像）
curl -L -o data\data.tar https://hf-mirror.com/datasets/rototoHF/tabm-data/resolve/main/data.tar
# 校验：1,034,147,840 字节；SHA256 应为 D8038250...385D
Get-FileHash data\data.tar -Algorithm SHA256
tar -xf data\data.tar -C data
```

解压后为 `paper/data/<dataset>/{N_train.npy,y_train.npy,...,info.json}`。本作业只用到
`california`（回归，13,209/3,303/4,128）、`churn`（二分类，6,400/1,600/2,000）、
`house`（回归，14,581/3,646/4,557）。**数据与 `exp/` 输出都不入库**（见 `.gitignore`）。

---

## 2. 一键复跑顺序

按下列顺序执行即可得到 `paper/results/` 下全部汇总；每一步都幂等，已完成的 run
（存在 `DONE`）会被跳过。

```powershell
cd "F:\study\大三上\数据分析课设\Tabm\paper"
$env:OMP_NUM_THREADS="14"; $env:MKL_NUM_THREADS="14"
$py = "D:\Anaconda\envs\tabm\python.exe"

# 0) 冒烟（可选，约 25 秒）
& $py bin\model.py exp\debug\0.toml --force

# 1) E2 基线：MLP / MLP†，3 数据集 × 3 种子     （实测约 39 分钟）
.\scripts\run_e2_baselines.ps1

# 2) E3 主方法：TabM / TabM-packed，3 数据集 × 3 种子（实测约 68 分钟）
.\scripts\run_e3_main.ps1

# 3) E3b 解混淆：TabM + 分段线性嵌入            （实测约 29 分钟）
.\scripts\run_e3b_plr.ps1

# 4) E4 集成规模：k=1/2/4/8/16/32，2 数据集，seed=0（实测约 13 小时，k=32 占 12 小时 39 分）
.\scripts\run_e4_kablation.ps1

# 5) 以下均为只读分析，不重新训练（合计 < 1 分钟）
& $py scripts\benchmark_efficiency.py   # E5 固定步数微基准
& $py scripts\analyze_errors.py         # E6 十分位/校准/配对
& $py scripts\plot_e6.py                # E6 三张图
& $py scripts\verify_implementation.py  # E7 独立实现等价性
& $py scripts\locate_effect_size.py     # E9 官方效应量分布
```

### 2.1 单次运行模板

```powershell
& $py bin\model.py exp\reproduce\E3\tabm\california\0.toml --force
```

`--force` 表示覆盖已有输出；去掉它则对已完成的 run 直接跳过。

### 2.2 配置从哪来

- 冻结模板：`paper/configs/E<实验>-<method>-<dataset>.toml`，共 13 个，文件头注释写明
  官方来源与全部缩减项。
- runner 运行时只替换 `seed = ...`（E4 另替换 `k`），并把**实际使用的配置**写到
  `paper/exp/reproduce/<实验>/.../<seed>.toml`；同目录另有 `<seed>.log`、
  `<seed>/report.json`、`<seed>/DONE`。因此每个结果都能追溯到「配置 + 日志 + 报告 + 提交号」。
- 每个 `paper/results/<实验>/runs.csv` 的 `commit` 列记录了当次实验的 commit。

---

## 3. 预期结果（用于核对复跑是否成功）

均值为 3 种子均值，括号内为总体标准差（ddof=0）。house 为原始目标量纲（约 3.1×10⁴），
不与 california 的 RMSE 直接比较。

| 数据集 | MLP | MLP† | TabM | TabM-packed | TabM+嵌入 |
| --- | --- | --- | --- | --- | --- |
| california RMSE | 0.4968 (0.0084) | 0.4522 (0.0038) | 0.4711 (0.0046) | 0.4881 (0.0023) | **0.4419** (0.0008) |
| churn Accuracy | 0.8520 (0.0036) | 0.8600 (0.0018) | 0.8587 (0.0018) | 0.8570 (0.0023) | **0.8628** (0.0018) |
| house RMSE | 31005 (143) | 30912 (299) | 30798 (388) | **30720** (272) | 31304 (168) |

参数量：TabM 79,944 / 84,112 / 82,056；TabM-packed 546,824 / 565,264 / 563,208
（**少 6.5~6.8 倍**）；TabM+嵌入 146k~148k。

E4（单种子探索性，仅用于确定 k）：

| 数据集 | k=1 | k=2 | k=4 | k=8 | k=16 | k=32 |
| --- | --- | --- | --- | --- | --- | --- |
| california RMSE | 0.48813 | 0.47898 | 0.47730 | 0.47432 | 0.47493 | 0.47190 |
| churn Accuracy | 0.85100 | 0.85900 | 0.85950 | 0.85700 | 0.85700 | 0.85950 |

E5（合成负载、固定步数、3 次重复中位数）：TabM 28.82 ms/step、65 千样本/秒；
TabM-packed 20.11 ms/step、89 千样本/秒 → **每步 1.43×、推理 1.36×**（同批次内比值）。

E7：7 项检查的最大绝对差均为 `0.000e+00`（逐位相同），参数量闭式与枚举一致。

E9：官方 10/10 数据集上 TabM 优于 MLP（回归中位 +4.20%，分类中位 +0.74pp）；
本项目 3/3 同向。

---

## 4. 结果目录

```text
paper/
├── configs/                      # 13 个冻结配置 + README（逐实验结论）
├── scripts/                      # 9 个脚本：runner 3 个 + 分析/基准 6 个
├── results/
│   ├── E2/{runs,summary}.csv
│   ├── E3/{runs,summary}.csv
│   ├── E3b/{runs,summary}.csv
│   ├── E4/{runs,summary}.csv
│   ├── E5/{benchmark,repeat1..3}.csv
│   ├── E6/{summary,summary_by_seed,error_by_decile,calibration,pairwise_vs_tabm}.csv
│   ├── E6/figures/{error_by_decile,calibration_churn,params_vs_error}.png
│   ├── E7/verification.csv
│   ├── E9/{official_effects,our_position}.csv + effect_size_distribution.png
│   └── comparison.csv            # E2/E3/E3b 跨实验汇总
└── exp/reproduce/                # 原始输出（不入库）
```

---

## 5. 质量检查

```powershell
cd "F:\study\大三上\数据分析课设\Tabm"
& $py -m ruff check paper\scripts tests
& $py -m ruff format --check paper\scripts tests
& $py -m pytest -q          # 96 passed
```

CI 定义在 `.github/workflows/ci.yml`（ruff + pytest，Python 3.10）。

### 5.1 干净目录复跑（2026-11-12 实测）

在一份全新的 `git worktree`（`HEAD = f97b68c4e`，不含任何本地 `exp/` 输出）中，
只拷入 `data/california` 后运行 E2 的 MLP 命令：

```powershell
git worktree add --detach <临时目录> HEAD
# 拷入 paper/data/california 与 exp/mlp/california/0-evaluation/0.toml
python bin\model.py exp\reproduce\smoke.toml --force
```

结果：`seed=0`，`test RMSE = 0.49699921464857993`，`n_parameters = 45553`，
训练 30.3 s、端到端 41.3 s。与台账中 E2 `mlp/california/seed=0` 的
`0.49699920415878296` 相差 `1.0e-8`（CPU 多线程归约顺序导致），可视为逐位可复现。

---

## 6. 已知差异与限制（复跑时请一并阅读）

1. **缩减**：主方法固定 `k=8`、`2×256`、`share_training_batches=true`，源自官方
   `tabm-sharetrainingbatches` 变体（california 为 `3×352`，churn `5×448`，house `5×816`）。
   基线 MLP/MLP† 使用官方 tuned 宽度**不缩减**，该不对称对 TabM 更严格。
2. **嵌入不对称**：MLP† 带分段线性数值嵌入而 TabM 不带，故 california 上 MLP† 优于
   TabM；E3b 用 `tabm-piecewiselinear` 解混淆，california/churn 上 TabM 反超 MLP†，
   house 上变差（官方嵌入超参是为 4×608 骨干调的，套到 2×256 失效）。
3. **E4 只有 seed=0**：受 CPU 预算限制，k 趋势属探索性观察，不构成统计证据。
4. **E5 是合成负载**：只比较同形状下的相对成本，不等于真实训练总耗时，也不外推到 GPU。
5. **效率结论**：CPU 上 TabM 每步比 TabM-packed 慢 1.43×，本作业只主张参数效率
   （少 6.8 倍），不主张论文 Figure 4 的「更快」。
6. **未做**：官方 k=32/3×400 完整配置（单次预计 >1 小时，已放弃）、Optuna 重调参、
   `torch.compile`、多数据集超参搜索、标签噪声实验（E8）。

---

## 7. 自检清单

- [x] 数据 SHA256 匹配，划分与官方一致，测试集只用一次
- [x] 每个结果都有「配置 + 日志 + report.json + commit」四件套
- [x] 3 种子重复；E4 单种子已显式标注
- [x] 所有结论方向与判据在实验前写定（`G1选题材料.md`）
- [x] 负结果保留：house 上 MLP† > TabM、E3b house 变差、C2 时间不成立
- [x] 96 个测试通过，ruff 全通过，E7 独立实现逐位等价
