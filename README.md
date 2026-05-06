# 数字经济对企业全要素生产率的影响研究

本项目按给定的 CSMAR 财务表、员工人数表、公司信息表，以及北京大学数字普惠金融指数 2011-2023 构建 2016-2023 年企业面板数据，并完成 TFP 测算、固定效应回归、XGBoost/SHAP 辅助分析、机制检验、异质性分析和稳健性检验。

## 目录结构

```text
DV/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── outputs/
│   ├── tables/
│   ├── figures/
│   └── reports/
├── src/
│   ├── 00_config.py
│   ├── 01_load_clean_merge.py
│   ├── 02_construct_variables.py
│   ├── 03_tfp_estimation.py
│   ├── 04_descriptive_statistics.py
│   ├── 05_baseline_fe.py
│   ├── 06_xgboost_shap.py
│   ├── 07_mechanism_tests.py
│   ├── 08_heterogeneity_tests.py
│   ├── 09_robustness_tests.py
│   ├── 10_visualizations.py
│   └── run_all.py
├── requirements.txt
└── README.md
```

## 环境安装

建议使用 Python 3.10-3.12。进入项目根目录后运行：

```bash
python -m pip install -r requirements.txt
```

macOS 如果运行 `06_xgboost_shap.py` 时提示缺少 `libomp.dylib`，需要先安装 OpenMP 运行库：

```bash
brew install libomp
```

`06_xgboost_shap.py` 会优先使用可选的 `shap` 包；如果未安装，则使用 XGBoost 原生 TreeSHAP (`pred_contribs=True`) 生成 `shap_summary.png` 和 `shap_lndigital_dependence.png`。

## 一键运行

确认以下文件已经放在 `data/raw/`：

- `FS_Combas.xlsx`，或旧版 `BalanceTable.xlsx`
- `FS_Comins.xlsx`，或旧版 `IncomeTable.xlsx`
- `FS_Comscfd.xlsx`，或旧版 `CashFlowTable.xlsx`
- `CG_Ybasic.xlsx`，或旧版 `HeadcountTable.xlsx`
- `STK_LISTEDCOINFOANL.xlsx`，或旧版 `InfoTable.xlsx`
- `北京大学数字普惠金融指数（PKU-DFIIC）2011-2023.xlsx`

然后运行：

```bash
python src/run_all.py
```

如果当前机器缺少 XGBoost 所需的 `libomp`，`run_all.py` 会跳过 `06_xgboost_shap.py` 并继续完成固定效应、机制、异质性、稳健性和图表流程；安装 `libomp` 后可再单独运行 `python src/06_xgboost_shap.py`。

所有路径、字段映射、样本期、变量清单和缩尾变量都集中写在 `src/00_config.py`。CSMAR 文件读取时使用 `openpyxl`，先扫描第 1 行真实字段宽度，再显式传入 `max_col`，不依赖异常的 `ws.max_column`。

新版 `STK_LISTEDCOINFOANL.xlsx` 不含企业产权性质和成立日期：代码会用首次上市日期计算 `Age`，并将 `SOE` 保留为空；因此 SOE 异质性分析会在结果表的 `errors` sheet 中记录为不可估计。

当前样本期设为 `2011-2023`。`Age` 使用 `year - 上市年份` 构造；在企业固定效应和年份固定效应同时存在时会被自动吸收，但仍用于 SA 指数。SA 指数按总资产百万元口径计算，避免将元口径资产直接代入 SA 公式。研发费用缺失样本用 `rd_missing` 控制，技术创新机制另用 `RD_observed` 对披露研发费用的样本做检验。

## 主要输出

- `data/processed/panel_model_data.csv`
- `data/processed/panel_model_data_with_tfp.csv`
- `outputs/tables/descriptive_statistics.xlsx`
- `outputs/tables/correlation_matrix.xlsx`
- `outputs/tables/baseline_fe_results.xlsx`
- `outputs/tables/ml_feature_importance.xlsx`
- `outputs/tables/mechanism_results.xlsx`
- `outputs/tables/heterogeneity_results.xlsx`
- `outputs/tables/robustness_results.xlsx`
- `outputs/figures/*.png`
- `outputs/reports/data_cleaning_report.md`
- `outputs/reports/model_interpretation.md`

## 方法说明

基准模型使用 `linearmodels.PanelOLS` 估计企业固定效应和年份固定效应，并在企业层面聚类稳健标准误；如果 `PanelOLS` 失败，代码会自动切换为 `statsmodels` 加企业和年份虚拟变量的备选实现。

XGBoost/SHAP 仅用于变量重要性和非线性关系发现，不用于因果识别。因果解释以固定效应模型、机制检验、异质性分析和稳健性检验为主。
