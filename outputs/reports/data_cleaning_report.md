# 数据清洗报告

- 样本期: 2011-2023
- CSMAR 表均从第 4 行读取数据，第 1 行作为英文字段名；读取前先扫描第 1 行实际字段宽度，再显式传入 max_col，避免 ws.max_column 异常。

## CSMAR 表处理行数

| 表 | 读取列数 | 原始行数 | Typrep=A 后 | 年末样本后 | 删除重复 | 最终行数 |
|---|---:|---:|---:|---:|---:|---:|
| balance | 7 | 389202 | 206557 | 42972 | 0 | 42972 |
| income | 9 | 389181 | 206554 | 42972 | 0 | 42972 |
| cashflow | 7 | 389043 | 206491 | 42958 | 0 | 42958 |
| headcount | 3 | 41344 |  | 41344 | 0 | 41344 |

## 公司信息表

- 原始行数: 42192
- 读取列数: 8
- 识别结构: annual_listed_company_info
- 合并键: stkcd+year
- 删除重复证券代码: 0
- 最终公司信息行数: 42192
- 因原表缺失而补为空的字段: establish_date, ownership_type, industry_a, industry_b, industry_c

## 合并结果

- 财务、员工与公司信息合并后样本量: 41331
- 城市名称缺失行数: 0
- 省份名称缺失行数: 0
- 数字普惠金融指数匹配层级: {'city': 37119, 'province': 4185, 'missing': 27}

后续变量构造脚本会继续删除核心变量非法值、生成模型变量并进行缩尾。

## 变量构造与异常值处理

- 变量构造前样本量: 41331
- 删除条件统计:
  - revenue missing_or_<=0: 14
  - fixed_assets missing_or_<=0: 12
  - employees missing_or_<=0: 11
  - total_assets missing_or_<=0: 0
  - total_liabilities missing_or_<0: 1
  - digital_index missing: 27
- 删除非法核心样本后样本量: 41268
- 缩尾变量: lnY, lnK, lnL, lnM, Size, Lev, ROA, Growth, Cashflow, Tangibility, RD, RD_observed, lnrd_expense, AdminRatio, InvestCashflow, SA, Turnover, lndigital, lndigital_coverage, lndigital_depth, lndigital_digitization
- RD 改进: `RD` 仍用于基准控制，研发费用缺失按 0 处理，同时加入 `rd_missing`；机制检验新增 `RD_observed`，只使用披露研发费用的样本。
- SA 改进: SA 指数按总资产百万元口径计算，并对 SA 公式中的规模项按文献常见做法封顶，避免直接使用元口径导致指数失真。

## TFP 测算

- tfp_ols 可用样本: 41268
- tfp_ols_m 可用样本: 41267
- 两个 TFP 指标均为带年份和行业固定效应生产函数的残差。
