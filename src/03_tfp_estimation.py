from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
cfg = importlib.import_module("00_config")


def estimate_tfp_residuals(df: pd.DataFrame, input_vars: list[str], residual_name: str) -> tuple[pd.Series, pd.DataFrame]:
    work = df[["lnY", "year", "industry"] + input_vars].copy()
    work["industry"] = work["industry"].fillna("Unknown").astype(str)
    for col in ["lnY"] + input_vars:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=["lnY"] + input_vars + ["year"]).copy()
    if work.empty:
        raise ValueError(f"{residual_name} 无可用样本，请检查 lnY 和投入变量。")

    dummies = pd.get_dummies(work[["year", "industry"]].astype(str), drop_first=True, dtype=float)
    x = pd.concat([work[input_vars].astype(float), dummies], axis=1)
    x.insert(0, "const", 1.0)
    y = work["lnY"].astype(float)
    beta, *_ = np.linalg.lstsq(x.to_numpy(dtype=float), y.to_numpy(dtype=float), rcond=None)
    fitted = x.to_numpy(dtype=float) @ beta
    residuals = pd.Series(np.nan, index=df.index, name=residual_name, dtype=float)
    residuals.loc[work.index] = y.to_numpy(dtype=float) - fitted

    coef_table = pd.DataFrame(
        {
            "variable": x.columns,
            "coefficient": beta,
            "tfp_version": residual_name,
            "nobs": len(work),
        }
    )
    return residuals, coef_table


def main() -> None:
    cfg.ensure_project_dirs()
    if not cfg.PANEL_MODEL_FILE.exists():
        raise FileNotFoundError(f"请先运行 02_construct_variables.py，缺失: {cfg.PANEL_MODEL_FILE}")
    df = pd.read_csv(cfg.PANEL_MODEL_FILE, dtype={"stkcd": str})
    cfg.require_columns(df, ["lnY", "lnK", "lnL", "lnM", "year", "industry"], "panel_model_data")

    df["tfp_ols"], coef_kl = estimate_tfp_residuals(df, ["lnK", "lnL"], "tfp_ols")
    df["tfp_ols_m"], coef_klm = estimate_tfp_residuals(df, ["lnK", "lnL", "lnM"], "tfp_ols_m")
    df.to_csv(cfg.PANEL_TFP_FILE, index=False, encoding="utf-8-sig")

    coef_table = pd.concat([coef_kl, coef_klm], ignore_index=True)
    with pd.ExcelWriter(cfg.TABLES_DIR / "tfp_estimation_results.xlsx", engine="openpyxl") as writer:
        coef_table.to_excel(writer, sheet_name="coefficients", index=False)

    cfg.append_markdown(
        cfg.REPORTS_DIR / "data_cleaning_report.md",
        "\n".join(
            [
                "",
                "## TFP 测算",
                "",
                f"- tfp_ols 可用样本: {int(df['tfp_ols'].notna().sum())}",
                f"- tfp_ols_m 可用样本: {int(df['tfp_ols_m'].notna().sum())}",
                "- 两个 TFP 指标均为带年份和行业固定效应生产函数的残差。",
                "",
            ]
        ),
    )
    print(f"已保存含 TFP 面板: {cfg.PANEL_TFP_FILE} ({len(df)} rows)")


if __name__ == "__main__":
    main()
