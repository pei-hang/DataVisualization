from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pandas as pd


SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
cfg = importlib.import_module("00_config")


def save_core_figures(df: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 5))
    sns.kdeplot(data=df, x="tfp_ols_m", fill=True, color="#2f6f73")
    plt.xlabel("TFP (OLS with intermediate input)")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(cfg.FIGURES_DIR / "tfp_density.png", dpi=cfg.FIG_DPI)
    plt.close()

    digital_trend = df.groupby("year", as_index=False)["digital_index"].mean()
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=digital_trend, x="year", y="digital_index", marker="o", color="#1f5f8b")
    plt.xlabel("Year")
    plt.ylabel("Mean digital index")
    plt.tight_layout()
    plt.savefig(cfg.FIGURES_DIR / "digital_trend.png", dpi=cfg.FIG_DPI)
    plt.close()

    tfp_trend = df.groupby("year", as_index=False)["tfp_ols_m"].mean()
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=tfp_trend, x="year", y="tfp_ols_m", marker="o", color="#7f3c2d")
    plt.xlabel("Year")
    plt.ylabel("Mean TFP")
    plt.tight_layout()
    plt.savefig(cfg.FIGURES_DIR / "tfp_trend.png", dpi=cfg.FIG_DPI)
    plt.close()

    sample = df[["lndigital", "tfp_ols_m"]].dropna()
    if len(sample) > 8000:
        sample = sample.sample(8000, random_state=cfg.RANDOM_STATE)
    plt.figure(figsize=(8, 5))
    sns.regplot(data=sample, x="lndigital", y="tfp_ols_m", scatter_kws={"s": 8, "alpha": 0.25}, line_kws={"color": "#7f3c2d"})
    plt.xlabel("ln(1 + digital index)")
    plt.ylabel("TFP")
    plt.tight_layout()
    plt.savefig(cfg.FIGURES_DIR / "digital_tfp_scatter.png", dpi=cfg.FIG_DPI)
    plt.close()

    corr_vars = [
        "tfp_ols_m",
        "lndigital",
        "Size",
        "Lev",
        "ROA",
        "Growth",
        "Cashflow",
        "Tangibility",
        "RD",
        "SA",
        "Turnover",
        "Age",
    ]
    corr = df[[col for col in corr_vars if col in df.columns]].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="vlag", center=0, annot=False, square=True, linewidths=0.3)
    plt.tight_layout()
    plt.savefig(cfg.FIGURES_DIR / "correlation_heatmap.png", dpi=cfg.FIG_DPI)
    plt.close()


def plot_saved_coefficients(table_path: Path, figure_name: str, sheet_name: str = "long") -> None:
    if not table_path.exists():
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        if sheet_name == "key_coefficients":
            coefs = pd.read_excel(table_path, sheet_name=sheet_name)
        else:
            long = pd.read_excel(table_path, sheet_name="long")
            coefs = long[long["variable"].eq("lndigital")].copy()
            coefs["ci_low"] = coefs["coef"] - 1.96 * coefs["std_error"]
            coefs["ci_high"] = coefs["coef"] + 1.96 * coefs["std_error"]
    except Exception:
        return
    coefs = coefs.dropna(subset=["coef", "std_error"])
    if coefs.empty:
        return
    coefs = coefs.sort_values("coef")
    plt.figure(figsize=(9, max(5, 0.35 * len(coefs))))
    plt.errorbar(
        coefs["coef"],
        coefs["model"],
        xerr=[coefs["coef"] - coefs["ci_low"], coefs["ci_high"] - coefs["coef"]],
        fmt="o",
        capsize=3,
        color="#3d5f74",
        ecolor="#9aaebb",
    )
    plt.axvline(0, color="#555555", linewidth=1, linestyle="--")
    plt.xlabel("Coefficient")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(cfg.FIGURES_DIR / figure_name, dpi=cfg.FIG_DPI)
    plt.close()


def main() -> None:
    cfg.ensure_project_dirs()
    if not cfg.PANEL_TFP_FILE.exists():
        raise FileNotFoundError(f"请先运行 03_tfp_estimation.py，缺失: {cfg.PANEL_TFP_FILE}")
    df = pd.read_csv(cfg.PANEL_TFP_FILE, dtype={"stkcd": str})
    save_core_figures(df)
    plot_saved_coefficients(cfg.TABLES_DIR / "heterogeneity_results.xlsx", "heterogeneity_coefficients.png")
    plot_saved_coefficients(cfg.TABLES_DIR / "robustness_results.xlsx", "robustness_coefficients.png", "key_coefficients")
    print(f"已保存基础图表到: {cfg.FIGURES_DIR}")


if __name__ == "__main__":
    main()
