from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pandas as pd


SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
cfg = importlib.import_module("00_config")


def main() -> None:
    cfg.ensure_project_dirs()
    if not cfg.PANEL_TFP_FILE.exists():
        raise FileNotFoundError(f"请先运行 03_tfp_estimation.py，缺失: {cfg.PANEL_TFP_FILE}")
    df = pd.read_csv(cfg.PANEL_TFP_FILE, dtype={"stkcd": str})

    specs = [
        ("M1_SA_no_size_age", "SA", ["lndigital"] + cfg.CONTROL_VARS_SA_MECHANISM),
        ("M1_TFP_SA_no_size_age", "tfp_ols_m", ["lndigital", "SA"] + cfg.CONTROL_VARS_SA_MECHANISM),
        ("M2_RD_observed", "RD_observed", ["lndigital"] + cfg.CONTROL_VARS_WITHOUT_RD),
        ("M2_TFP_RD_missing_adjusted", "tfp_ols_m", ["lndigital", "RD"] + cfg.CONTROL_VARS_WITHOUT_RD),
        ("M3_Turnover", "Turnover", ["lndigital"] + cfg.CONTROL_VARS),
        ("M3_TFP_Turnover", "tfp_ols_m", ["lndigital", "Turnover"] + cfg.CONTROL_VARS),
    ]
    results = []
    for model_name, dep_var, x_vars in specs:
        print(f"运行机制检验 {model_name}")
        results.append(cfg.run_panel_fe(df, dep_var, x_vars, model_name, cluster_var="stkcd"))

    ordered_vars = ["lndigital", "SA", "RD", "RD_observed", "Turnover"] + cfg.CONTROL_VARS
    cfg.save_regression_results(results, ordered_vars, cfg.TABLES_DIR / "mechanism_results.xlsx")
    print(f"已保存机制检验结果: {cfg.TABLES_DIR / 'mechanism_results.xlsx'}")


if __name__ == "__main__":
    main()
