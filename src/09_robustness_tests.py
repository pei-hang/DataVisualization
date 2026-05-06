from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pandas as pd


SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
cfg = importlib.import_module("00_config")


def plot_coefficients(coefs: pd.DataFrame) -> None:
    if coefs.empty:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    coefs = coefs.sort_values("coef")
    plt.figure(figsize=(9, max(5, 0.35 * len(coefs))))
    plt.errorbar(
        coefs["coef"],
        coefs["model"],
        xerr=[coefs["coef"] - coefs["ci_low"], coefs["ci_high"] - coefs["coef"]],
        fmt="o",
        color="#7f3c2d",
        ecolor="#bd9a8e",
        capsize=3,
    )
    plt.axvline(0, color="#555555", linewidth=1, linestyle="--")
    plt.xlabel("Coefficient on reported digital variable")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(cfg.FIGURES_DIR / "robustness_coefficients.png", dpi=cfg.FIG_DPI)
    plt.close()


def main() -> None:
    cfg.ensure_project_dirs()
    if not cfg.PANEL_TFP_FILE.exists():
        raise FileNotFoundError(f"请先运行 03_tfp_estimation.py，缺失: {cfg.PANEL_TFP_FILE}")
    df = pd.read_csv(cfg.PANEL_TFP_FILE, dtype={"stkcd": str})

    specs = [
        ("R1_tfp_ols", df, "tfp_ols", ["lndigital"] + cfg.CONTROL_VARS, "lndigital", "stkcd"),
        ("R2_coverage", df, "tfp_ols_m", ["lndigital_coverage"] + cfg.CONTROL_VARS, "lndigital_coverage", "stkcd"),
        ("R2_depth", df, "tfp_ols_m", ["lndigital_depth"] + cfg.CONTROL_VARS, "lndigital_depth", "stkcd"),
        (
            "R2_digitization",
            df,
            "tfp_ols_m",
            ["lndigital_digitization"] + cfg.CONTROL_VARS,
            "lndigital_digitization",
            "stkcd",
        ),
        ("R3_L1_lndigital", df, "tfp_ols_m", ["L1_lndigital"] + cfg.CONTROL_VARS, "L1_lndigital", "stkcd"),
        ("R4_exclude_2020", df[df["year"] != 2020], "tfp_ols_m", ["lndigital"] + cfg.CONTROL_VARS, "lndigital", "stkcd"),
        (
            "R5_exclude_2020_2021",
            df[~df["year"].isin([2020, 2021])],
            "tfp_ols_m",
            ["lndigital"] + cfg.CONTROL_VARS,
            "lndigital",
            "stkcd",
        ),
        (
            "R6_exclude_key_cities",
            df[~df["city_norm"].isin(["北京", "上海", "深圳", "杭州"])],
            "tfp_ols_m",
            ["lndigital"] + cfg.CONTROL_VARS,
            "lndigital",
            "stkcd",
        ),
        (
            "R7_cluster_province",
            df,
            "tfp_ols_m",
            ["lndigital"] + cfg.CONTROL_VARS,
            "lndigital",
            "province_norm",
        ),
        (
            "R7_cluster_industry",
            df,
            "tfp_ols_m",
            ["lndigital"] + cfg.CONTROL_VARS,
            "lndigital",
            "industry",
        ),
    ]

    results = []
    errors = []
    coef_rows = []
    for model_name, data, dep_var, x_vars, key_var, cluster_var in specs:
        try:
            print(f"运行稳健性检验 {model_name}, n={len(data)}")
            result = cfg.run_panel_fe(data.copy(), dep_var, x_vars, model_name, cluster_var=cluster_var)
            results.append(result)
            key = cfg.key_coefficient_frame([result], key_var)
            if not key.empty:
                key["reported_variable"] = key_var
                coef_rows.append(key)
        except Exception as exc:
            errors.append({"model": model_name, "n": len(data), "error": str(exc)})

    output_path = cfg.TABLES_DIR / "robustness_results.xlsx"
    ordered_vars = [
        "lndigital",
        "lndigital_coverage",
        "lndigital_depth",
        "lndigital_digitization",
        "L1_lndigital",
    ] + cfg.CONTROL_VARS
    if results:
        cfg.save_regression_results(results, ordered_vars, output_path)
        if errors:
            with pd.ExcelWriter(output_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
                pd.DataFrame(errors).to_excel(writer, sheet_name="errors", index=False)
        coefs = pd.concat(coef_rows, ignore_index=True) if coef_rows else pd.DataFrame()
        if not coefs.empty:
            with pd.ExcelWriter(output_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
                coefs.to_excel(writer, sheet_name="key_coefficients", index=False)
            plot_coefficients(coefs)
    else:
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            pd.DataFrame(errors).to_excel(writer, sheet_name="errors", index=False)

    print(f"已保存稳健性检验结果: {output_path}")


if __name__ == "__main__":
    main()
