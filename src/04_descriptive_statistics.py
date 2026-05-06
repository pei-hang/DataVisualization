from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pandas as pd


SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
cfg = importlib.import_module("00_config")


DESCRIPTIVE_VARS = [
    "tfp_ols_m",
    "tfp_ols",
    "lndigital",
    "lndigital_coverage",
    "lndigital_depth",
    "lndigital_digitization",
    "Size",
    "Lev",
    "ROA",
    "Growth",
    "Cashflow",
    "Tangibility",
    "RD",
    "RD_observed",
    "rd_missing",
    "Age",
    "SA",
    "SA_size_million",
    "Turnover",
    "AdminRatio",
    "InvestCashflow",
]


def main() -> None:
    cfg.ensure_project_dirs()
    input_path = cfg.PANEL_TFP_FILE if cfg.PANEL_TFP_FILE.exists() else cfg.PANEL_MODEL_FILE
    if not input_path.exists():
        raise FileNotFoundError("请先运行 02_construct_variables.py 和 03_tfp_estimation.py。")

    df = pd.read_csv(input_path, dtype={"stkcd": str})
    available = [col for col in DESCRIPTIVE_VARS if col in df.columns]
    missing = [col for col in DESCRIPTIVE_VARS if col not in df.columns]
    if missing:
        print(f"描述统计跳过缺失变量: {missing}")

    desc = df[available].describe(percentiles=[0.25, 0.5, 0.75]).T
    desc = desc.rename(columns={"50%": "median"})
    desc["missing"] = df[available].isna().sum()
    desc = desc[["count", "missing", "mean", "std", "min", "25%", "median", "75%", "max"]]
    corr = df[available].corr()

    with pd.ExcelWriter(cfg.TABLES_DIR / "descriptive_statistics.xlsx", engine="openpyxl") as writer:
        desc.to_excel(writer, sheet_name="descriptive")
    with pd.ExcelWriter(cfg.TABLES_DIR / "correlation_matrix.xlsx", engine="openpyxl") as writer:
        corr.to_excel(writer, sheet_name="correlation")

    print(f"已保存描述统计: {cfg.TABLES_DIR / 'descriptive_statistics.xlsx'}")
    print(f"已保存相关矩阵: {cfg.TABLES_DIR / 'correlation_matrix.xlsx'}")


if __name__ == "__main__":
    main()
