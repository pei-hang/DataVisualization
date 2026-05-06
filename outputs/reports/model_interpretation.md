# 模型结果解释

## 基准固定效应模型

- 首选基准模型为 Model 2，因变量为 `tfp_ols_m`，核心解释变量为 `lndigital`，控制企业固定效应和年份固定效应，并在企业层面聚类稳健标准误。
- Model 2 中 `lndigital` 的估计系数为 0.1253***，p=0.0001。
- 当前样本期为 2011-2023。
- 基准控制变量已加入 `rd_missing`，避免把研发费用未披露样本完全等同于真实零研发。
- 表格中的星号含义: *** p<0.01，** p<0.05，* p<0.10。

## XGBoost / SHAP 说明

- XGBoost/SHAP 只用于变量重要性和非线性关系发现，不用于因果识别；因果解释仍以固定效应模型、机制检验、异质性分析和稳健性检验为主。

## 当前数据限制

- 若原始公司信息表不含 `OWNERSHIPTYPE`，则 `SOE` 无法识别；SOE 分组回归会在 `heterogeneity_results.xlsx` 的 `errors` sheet 中记录为不可估计。
- `Age = year - 上市年份` 在企业固定效应和年份固定效应同时存在时会与固定效应共线，回归中被自动吸收；它仍保留用于 SA 指数构造。
- SA 指数已改为按总资产百万元口径计算，避免用元口径直接进入 SA 公式造成数量级失真。

## XGBoost / SHAP 辅助分析

- 测试集 RMSE=0.1895，MAE=0.1163，R2=0.4983。
- SHAP 计算方式: xgboost native TreeSHAP pred_contribs。
- XGBoost 只用于变量重要性和非线性关系发现，不用于因果识别；因果解释仍以固定效应模型及稳健性检验为主。

## XGBoost / SHAP 辅助分析

- 测试集 RMSE=0.1895，MAE=0.1163，R2=0.4983。
- SHAP 计算方式: xgboost native TreeSHAP pred_contribs。
- XGBoost 只用于变量重要性和非线性关系发现，不用于因果识别；因果解释仍以固定效应模型及稳健性检验为主。

## XGBoost / SHAP 辅助分析

- 测试集 RMSE=0.1895，MAE=0.1163，R2=0.4983。
- SHAP 计算方式: xgboost native TreeSHAP pred_contribs。
- XGBoost 只用于变量重要性和非线性关系发现，不用于因果识别；因果解释仍以固定效应模型及稳健性检验为主。

## XGBoost / SHAP 辅助分析

- 测试集 RMSE=0.1895，MAE=0.1163，R2=0.4983。
- SHAP 计算方式: xgboost native TreeSHAP pred_contribs。
- XGBoost 只用于变量重要性和非线性关系发现，不用于因果识别；因果解释仍以固定效应模型及稳健性检验为主。

## XGBoost / SHAP 辅助分析

- 测试集 RMSE=0.1895，MAE=0.1163，R2=0.4983。
- SHAP 计算方式: xgboost native TreeSHAP pred_contribs。
- XGBoost 只用于变量重要性和非线性关系发现，不用于因果识别；因果解释仍以固定效应模型及稳健性检验为主。
