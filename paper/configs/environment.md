# 运行环境（本项目实测）

本文件是 `REPRODUCTION.md` 与 `README.md` 所引用的环境记录；完整实验日志、耗时与
数据校验过程见作业目录下的 `logs/environment.md`（该文件不属于代码仓库）。

## 硬件

| 项目 | 值 |
| --- | --- |
| 系统 | Windows 11（10.0.26300） |
| CPU | Intel Core Ultra 5 125H，14 核 18 线程 |
| 内存 | 31.5 GB |
| GPU | 无 CUDA，`torch.cuda.is_available() == False` |
| 线程 | `OMP_NUM_THREADS=14`、`MKL_NUM_THREADS=14` |

无 GPU 时 `amp = true` 会被框架自动关闭，日志中为 `AMP enabled: False`，因此全部
结果均为 FP32 CPU 结果。

## Python 环境

- conda 环境名 `tabm`，Python **3.10.22**
- 冻结清单：[`environment.yml`](environment.yml)（`conda env export --no-builds`，已去掉 `prefix:`）
- 重建：

```powershell
conda env create -f paper\configs\environment.yml
conda activate tabm
python -m pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu
```

## 关键版本

| 包 | 版本 |
| --- | --- |
| torch | 2.0.1+cpu |
| numpy | 1.26.3 |
| scikit-learn | 1.3.1 |
| pandas | 2.1.1 |
| tomli | 2.0.1 |
| rtdl_num_embeddings | 0.0.11 |
| rtdl_revisiting_models | 0.0.2 |
| matplotlib（仅出图） | 3.10.9 |

## 数据

`paper/data/data.tar`：1,034,147,840 字节，SHA256
`D8038250C20A83F241BAB822C32B71C622C032B8C25559E9AB26CAEAFCA5385D`。
解压后为 `paper/data/<dataset>/{N_train,y_train,N_val,y_val,N_test,y_test}.npy` 与
`info.json`，划分与论文一致，未重新切分。
