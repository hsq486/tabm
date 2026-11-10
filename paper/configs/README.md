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

> 官方 `exp/` 下的配置与结果未修改；本目录只存放可提交的复现配置，原始输出留在 `exp/reproduce/` 且不进入 Git。
