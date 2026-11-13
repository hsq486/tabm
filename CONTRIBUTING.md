# 贡献指南

本项目是 TabM（ICLR 2025）的课程复现工程，在 fork `hsq486/tabm` 上开发。
`main` 始终可运行、可复现；所有改动经分支 + PR 合入。

## 1. 分支策略

| 分支 | 用途 | 命名示例 |
| --- | --- | --- |
| `main` | 稳定、可演示、可验收 | `main` |
| `feature/` | 新增功能、新实验脚本 | `feature/14-gpu-e11` |
| `fix/` | 缺陷修复 | `fix/27-ledger-bom` |
| `docs/` | 文档修改 | `docs/contributing` |
| `experiment/` | 探索性实验，结论稳定后整理并合入 | `experiment/gpu-fullscale` |

约定：

- 从**最新 `main`** 开分支，分支生命周期不超过一个任务。
- 不要直接在 `main` 上提交。
- 合并后**立即删除分支**（本地与远端）。

## 2. 提交信息

格式 `type(scope): summary`，`type` 取 `feat` / `fix` / `docs` / `test` / `refactor` / `perf` / `ci` / `chore`。

```
feat(gpu): add the E10-E14 matrix runner for the second machine
fix(results): use repo-relative ledger paths and BOM-free CSV
```

一次提交只解决一个逻辑问题；禁止 `update`、`修改一下`、`final` 这类消息。

## 3. Pull Request

1. 有对应 Issue 时在正文写 `Closes #<编号>`。
2. 推送后创建 PR，标题写结果而不是动作。
3. 正文必填四项：变更内容、验证方法、结果影响、风险。
4. 等 CI（ruff + pytest）通过；实验相关 PR 附指标对比。
5. 合并方式统一用 Squash merge，合并后删除分支。

## 4. 本地环境与检查

```bash
conda activate tabm
python -m pytest -q          # 96 个用例
ruff check .                 # 与 CI 一致
ruff format --check .
```

GPU 机器上的环境见 `paper/configs/environment-gpu.yml`，CPU 机器见 `paper/configs/environment.md`。

## 5. 数据与产物

- `paper/data/` 不进入 Git（`.gitignore` 中的 `data` 规则），按 `REPRODUCTION.md` 第 2 节获取并校验 SHA256。
- `paper/exp/**/report.json` 与 `paper/results/**/runs.csv` 是实验证据，**要提交**。
- 中间 checkpoint、日志（`*.log`）、`data.tar` 不提交。
- 图表由脚本生成，提交脚本与指标表，不只提交图片。

## 6. 历史说明

- PR [#1](https://github.com/hsq486/tabm/pull/1)、[#2](https://github.com/hsq486/tabm/pull/2) 按上述流程合并，含 CI 记录。
- `acc2663f1`（5 种子主对比与 E8 标签噪声）是早期流程未固化时的本地合并，未走 PR；此后所有改动统一走分支 + PR + CI。
