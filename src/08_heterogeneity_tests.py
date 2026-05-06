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
        color="#1f5f8b",
        ecolor="#8aa9bd",
        capsize=3,
    )
    plt.axvline(0, color="#555555", linewidth=1, linestyle="--")
    plt.xlabel("Coefficient on lndigital")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(cfg.FIGURES_DIR / "heterogeneity_coefficients.png", dpi=cfg.FIG_DPI)
    plt.close()


def add_group_if_available(groups: list[tuple[str, pd.Series]], errors: list[dict], df: pd.DataFrame, column: str, value, name: str) -> None:
    if column not in df.columns or df[column].notna().sum() == 0:
        errors.append({"group": name, "n": 0, "error": f"{column} 缺失或全为空，跳过该分组。"})
        return
    mask = df[column] == value
    groups.append((name, mask))


def run_interaction_tests(df: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    interaction_specs: list[tuple[str, str, list[str]]] = []
    work = df.copy()
    errors: list[dict] = []

    binary_groups = [
        ("Size_high", "Size_high"),
        ("manufacturing", "manufacturing"),
        ("hightech", "hightech"),
    ]
    if "SOE" in work.columns and work["SOE"].notna().sum() > 0:
        binary_groups.append(("SOE", "SOE"))
    elif "SOE" in work.columns:
        errors.append({"group": "INT_SOE", "n": 0, "error": "SOE 缺失或全为空，无法做交互项检验。"})

    for label, column in binary_groups:
        dummy = f"D_{label}"
        interaction = f"lndigital_x_{label}"
        work[dummy] = pd.to_numeric(work[column], errors="coerce")
        work[interaction] = work["lndigital"] * work[dummy]
        interaction_specs.append((f"INT_{label}", interaction, ["lndigital", dummy, interaction] + cfg.CONTROL_VARS))

    for region in ["central", "west", "northeast"]:
        dummy = f"D_region_{region}"
        interaction = f"lndigital_x_region_{region}"
        work[dummy] = (work["region"] == region).astype(int)
        work.loc[work["region"].isna(), dummy] = pd.NA
        work[interaction] = work["lndigital"] * work[dummy]
        interaction_specs.append(
            (f"INT_region_{region}_vs_east", interaction, ["lndigital", dummy, interaction] + cfg.CONTROL_VARS)
        )

    results = []
    for model_name, interaction, x_vars in interaction_specs:
        try:
            print(f"运行异质性交互项检验 {model_name}")
            result = cfg.run_panel_fe(work, "tfp_ols_m", x_vars, model_name, cluster_var="stkcd")
            result["interaction_variable"] = interaction
            results.append(result)
        except Exception as exc:
            errors.append({"group": model_name, "n": len(work), "error": str(exc)})
    return results, errors


def interaction_key_frame(results: list[dict]) -> pd.DataFrame:
    rows = []
    for result in results:
        variable = result.get("interaction_variable")
        if not variable:
            continue
        coef = result.get("params", {}).get(variable)
        se = result.get("std_errors", {}).get(variable)
        pvalue = result.get("pvalues", {}).get(variable)
        rows.append(
            {
                "model": result.get("model"),
                "interaction_variable": variable,
                "coef": coef,
                "std_error": se,
                "p_value": pvalue,
                "nobs": result.get("nobs"),
                "rsquared_within": result.get("rsquared_within"),
                "engine": result.get("engine"),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    cfg.ensure_project_dirs()
    if not cfg.PANEL_TFP_FILE.exists():
        raise FileNotFoundError(f"请先运行 03_tfp_estimation.py，缺失: {cfg.PANEL_TFP_FILE}")
    df = pd.read_csv(cfg.PANEL_TFP_FILE, dtype={"stkcd": str})

    errors = []
    groups: list[tuple[str, pd.Series]] = []
    add_group_if_available(groups, errors, df, "SOE", 1, "SOE=1")
    add_group_if_available(groups, errors, df, "SOE", 0, "SOE=0")
    groups.extend([
        ("Size_high=1", df["Size_high"] == 1),
        ("Size_high=0", df["Size_high"] == 0),
        ("Region_east", df["region"] == "east"),
        ("Region_central", df["region"] == "central"),
        ("Region_west", df["region"] == "west"),
        ("Region_northeast", df["region"] == "northeast"),
        ("Manufacturing=1", df["manufacturing"] == 1),
        ("Manufacturing=0", df["manufacturing"] == 0),
        ("Hightech=1", df["hightech"] == 1),
        ("Hightech=0", df["hightech"] == 0),
    ])

    results = []
    for group_name, mask in groups:
        subset = df.loc[mask].copy()
        try:
            print(f"运行异质性分组 {group_name}, n={len(subset)}")
            results.append(
                cfg.run_panel_fe(
                    subset,
                    "tfp_ols_m",
                    ["lndigital"] + cfg.CONTROL_VARS,
                    group_name,
                    cluster_var="stkcd",
                )
            )
        except Exception as exc:
            errors.append({"group": group_name, "n": len(subset), "error": str(exc)})

    output_path = cfg.TABLES_DIR / "heterogeneity_results.xlsx"
    if results:
        _, long = cfg.save_regression_results(results, ["lndigital"] + cfg.CONTROL_VARS, output_path)
        interaction_results, interaction_errors = run_interaction_tests(df)
        errors.extend(interaction_errors)
        if interaction_results:
            interaction_vars = sorted(
                {
                    var
                    for result in interaction_results
                    for var in result.get("x_vars", [])
                    if var.startswith("lndigital_x_") or var.startswith("D_") or var == "lndigital"
                }
            )
            interaction_order = ["lndigital"] + [var for var in interaction_vars if var != "lndigital"] + cfg.CONTROL_VARS
            interaction_formatted, interaction_long = cfg.regression_results_to_frames(interaction_results, interaction_order)
            with pd.ExcelWriter(output_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
                interaction_formatted.to_excel(writer, sheet_name="interaction")
                interaction_long.to_excel(writer, sheet_name="interaction_long", index=False)
                interaction_key_frame(interaction_results).to_excel(writer, sheet_name="interaction_key", index=False)
        if errors:
            with pd.ExcelWriter(output_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
                pd.DataFrame(errors).to_excel(writer, sheet_name="errors", index=False)
        coefs = cfg.key_coefficient_frame(results, "lndigital")
        plot_coefficients(coefs)
    else:
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            pd.DataFrame(errors).to_excel(writer, sheet_name="errors", index=False)

    print(f"已保存异质性检验结果: {output_path}")


if __name__ == "__main__":
    main()
