from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pandas as pd


SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
cfg = importlib.import_module("00_config")


def build_models(df: pd.DataFrame) -> list[dict]:
    models = [
        ("Model 1", "tfp_ols_m", ["lndigital"]),
        ("Model 2", "tfp_ols_m", ["lndigital"] + cfg.CONTROL_VARS),
        ("Model 3", "tfp_ols", ["lndigital"] + cfg.CONTROL_VARS),
        ("Model 4", "tfp_ols_m", ["lndigital_coverage"] + cfg.CONTROL_VARS),
        ("Model 5", "tfp_ols_m", ["lndigital_depth"] + cfg.CONTROL_VARS),
        ("Model 6", "tfp_ols_m", ["lndigital_digitization"] + cfg.CONTROL_VARS),
    ]
    results = []
    for model_name, dep_var, x_vars in models:
        print(f"运行 {model_name}: {dep_var} <- {x_vars[0]}")
        results.append(cfg.run_panel_fe(df, dep_var, x_vars, model_name, cluster_var="stkcd"))
    return results


def write_interpretation(results: list[dict]) -> None:
    model2 = next((res for res in results if res["model"] == "Model 2"), None)
    coef = model2["params"].get("lndigital") if model2 else None
    pvalue = model2["pvalues"].get("lndigital") if model2 else None
    stars = cfg.significance_stars(pvalue)
    coef_text = "无法取得" if coef is None else f"{coef:.4f}{stars}"
    p_text = "" if pvalue is None else f"，p={pvalue:.4f}"
    content = "\n".join(
        [
            "# 模型结果解释",
            "",
            "## 基准固定效应模型",
            "",
            f"- 首选基准模型为 Model 2，因变量为 `tfp_ols_m`，核心解释变量为 `lndigital`，控制企业固定效应和年份固定效应，并在企业层面聚类稳健标准误。",
            f"- Model 2 中 `lndigital` 的估计系数为 {coef_text}{p_text}。",
            "- 当前样本期为 2011-2023。",
            "- 基准控制变量已加入 `rd_missing`，避免把研发费用未披露样本完全等同于真实零研发。",
            "- 表格中的星号含义: *** p<0.01，** p<0.05，* p<0.10。",
            "",
            "## XGBoost / SHAP 说明",
            "",
            "- XGBoost/SHAP 只用于变量重要性和非线性关系发现，不用于因果识别；因果解释仍以固定效应模型、机制检验、异质性分析和稳健性检验为主。",
            "",
            "## 当前数据限制",
            "",
            "- 若原始公司信息表不含 `OWNERSHIPTYPE`，则 `SOE` 无法识别；SOE 分组回归会在 `heterogeneity_results.xlsx` 的 `errors` sheet 中记录为不可估计。",
            "- `Age = year - 上市年份` 在企业固定效应和年份固定效应同时存在时会与固定效应共线，回归中被自动吸收；它仍保留用于 SA 指数构造。",
            "- SA 指数已改为按总资产百万元口径计算，避免用元口径直接进入 SA 公式造成数量级失真。",
            "",
        ]
    )
    cfg.write_markdown(cfg.REPORTS_DIR / "model_interpretation.md", content)


def main() -> None:
    cfg.ensure_project_dirs()
    if not cfg.PANEL_TFP_FILE.exists():
        raise FileNotFoundError(f"请先运行 03_tfp_estimation.py，缺失: {cfg.PANEL_TFP_FILE}")
    df = pd.read_csv(cfg.PANEL_TFP_FILE, dtype={"stkcd": str})

    results = build_models(df)
    ordered_vars = [
        "lndigital",
        "lndigital_coverage",
        "lndigital_depth",
        "lndigital_digitization",
    ] + cfg.CONTROL_VARS
    cfg.save_regression_results(results, ordered_vars, cfg.TABLES_DIR / "baseline_fe_results.xlsx")
    write_interpretation(results)
    print(f"已保存基准回归结果: {cfg.TABLES_DIR / 'baseline_fe_results.xlsx'}")


if __name__ == "__main__":
    main()
