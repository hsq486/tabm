# 学术诚信与工作量声明（DECLARATION）

| 项目 | 内容 |
| --- | --- |
| 课程 | 数据分析与数据挖掘实践 |
| 题目 | TabM（ICLR 2025）的参数高效集成复现与验证 |
| 姓名 | ______________________ |
| 学号 | ______________________ |
| 班级 | ______________________ |
| 日期 | ______________________ |

---

## 1. 独立完成声明

除第 2、3 节明确列出的外部代码、数据与工具外，本作业的实验设计、环境搭建、全部
训练与评估、结果分析及报告撰写由本人独立完成。实验在本人的个人电脑上实际执行，
报告中的每一个数值都可在 `paper/results/` 与 `paper/exp/reproduce/` 的原始输出中
逐条核对，不存在编造、复制他人结果或从论文表格直接转录的情况。

## 2. 外部代码的使用

| 来源 | 版本/地址 | 使用方式 | 许可 |
| --- | --- | --- | --- |
| TabM 官方仓库 | `b53fd7cf2d9be2d66b918ca6cf66e0365e4ce11c`，<https://github.com/yandex-research/tabm> | 作为复现基线：直接使用其模型实现（`tabm.py`）、训练入口（`paper/bin/model.py`）、数据管线与官方调参结果（`paper/exp/*/0-evaluation/*.toml`） | Apache-2.0，已保留 `LICENSE` |
| 上游依赖 | `rtdl_num_embeddings` 等 | 通过 pip 安装，未修改 | 各自原始许可 |

本人新增的内容（均在上游代码之外，未改动上游模型语义）：
`paper/configs/`（13 个冻结配置）、`paper/scripts/`（9 个脚本）、
`paper/results/`（汇总表与图）、`tests/`（96 个用例）、`.github/workflows/ci.yml`、
`README.md` 的复现章节、本文件与 `REPRODUCTION.md`。

对上游文件的唯一改动是 `.gitignore`：新增 `paper/exp/reproduce/`（本地实验输出不入库）、
`.ruff_cache/`，并用 `!paper/results/**/*.csv` 允许提交小体积结果台账。
`tabm.py`、`paper/lib/`、`paper/bin/` 未作任何修改，可用
`git diff b53fd7cf2 -- tabm.py paper/bin paper/lib` 验证为空。

## 3. 数据来源

实验使用论文作者发布的官方数据包 `data.tar`（HuggingFace 镜像
`rototoHF/tabm-data`），SHA256 `D8038250C20A83F241BAB822C32B71C622C032B8C25559E9AB26CAEAFCA5385D`。
数据集的原始出处见论文附录；本项目只使用其官方划分，未重新划分、未在测试集上调参。
数据文件与训练输出均未提交入库。

## 4. AI 工具使用披露

本作业使用了 AI 编程助手（Trae IDE 内置模型），使用范围与边界如下：

| 环节 | 是否使用 AI | 具体说明 |
| --- | --- | --- |
| 文献阅读与选题 | 是（辅助） | 辅助梳理论文主张与判据；选题、判据与实验矩阵由本人确认 |
| 实验代码（runner、分析脚本） | 是 | AI 起草 `paper/scripts/*.ps1`、`*.py` 与 `tests/`，本人逐行阅读、运行并修正（如 PowerShell BOM 导致 TOML 解析失败、tqdm 输出被误判为错误、十分位切分方式等） |
| 实验执行 | 否 | 全部 57 次训练与 5 个分析脚本（E5/E6×2/E7/E9）在本人的机器上实际运行，耗时与日志见 `logs/environment.md` |
| 结果数据 | 否 | 所有数值来自 `report.json` / `predictions.npz`，AI 未生成、未修改任何实验结果 |
| 图表与文档草稿 | 是（辅助） | 图表由脚本从 CSV 直接绘制；文档由 AI 起草初稿，本人核对数值与表述后定稿 |
| 代码审查与测试 | 是（辅助） | 用 AI 生成测试用例与 ruff 规则检查；E7 用**手写独立实现**做数值核验，最大绝对差为 0 |

AI 生成的内容均经过可执行验证（96 个测试、ruff、E7 逐位等价），本人对其正确性负责。

## 5. 结果真实性声明

- 三种子重复实验（E2/E3/E3b）的全部结果均已报告，包括与论文不一致的部分：
  - CPU 上 TabM 每步比 TabM-packed 慢 1.43×，本作业**不主张**论文的「更快」结论；
  - california 上带数值嵌入的 MLP† 优于 TabM，负结果保留并用 E3b 解混淆；
  - E3b 在 house 上变差（−392 ~ −583），未删除、未调参掩盖。
- E4 只跑了 `seed=0`，所有引用处均标注「探索性、非统计证据」，未写成稳定规律。
- 测试集仅在早停后用一次；早停与选型只用验证集。
- 未使用论文表格中的数值替代本机结果；与论文的对照统一以「差异表」形式给出，
  并在 `paper/configs/README.md` 中注明配置差异与归因。

## 6. 引用

1. Gorishniy, Y., Kotelnikov, A., Babenko, A. *TabM: Advancing Tabular Deep Learning
   with Parameter-Efficient Ensembling.* ICLR 2025.
2. 官方实现：<https://github.com/yandex-research/tabm>（commit `b53fd7cf2`）。
3. 数据包：<https://huggingface.co/datasets/rototoHF/tabm-data>。

---

本人已阅读并遵守课程关于学术诚信与代码规范的要求，上述声明真实有效。

签名：______________________　　日期：______________________
