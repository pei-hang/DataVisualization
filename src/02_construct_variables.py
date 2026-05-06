from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
cfg = importlib.import_module("00_config")


def classify_soe(value: object) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value)
    if re.search(r"国有|中央|地方国企|国企|国资", text):
        return 1.0
    if re.search(r"私营|民营|外资|集体", text):
        return 0.0
    return np.nan


def build_industry(df: pd.DataFrame) -> pd.Series:
    candidates = [col for col in ["industry", "industry_d", "industry_c", "industry_b", "industry_a"] if col in df.columns]
    if not candidates:
        raise ValueError("缺失行业字段: 需要 industry 或 industry_d/industry_c/industry_b/industry_a 至少一个。")
    return df[candidates].replace("", np.nan).bfill(axis=1).iloc[:, 0]


def append_variable_report(initial_rows: int, invalid_counts: dict[str, int], final_rows: int) -> None:
    lines = [
        "",
        "## 变量构造与异常值处理",
        "",
        f"- 变量构造前样本量: {initial_rows}",
        "- 删除条件统计:",
    ]
    for name, count in invalid_counts.items():
        lines.append(f"  - {name}: {count}")
    lines.extend(
        [
            f"- 删除非法核心样本后样本量: {final_rows}",
            f"- 缩尾变量: {', '.join(cfg.WINSOR_VARS)}",
            "- RD 改进: `RD` 仍用于基准控制，研发费用缺失按 0 处理，同时加入 `rd_missing`；机制检验新增 `RD_observed`，只使用披露研发费用的样本。",
            "- SA 改进: SA 指数按总资产百万元口径计算，并对 SA 公式中的规模项按文献常见做法封顶，避免直接使用元口径导致指数失真。",
            "",
        ]
    )
    cfg.append_markdown(cfg.REPORTS_DIR / "data_cleaning_report.md", "\n".join(lines))


def main() -> None:
    cfg.ensure_project_dirs()
    if not cfg.PANEL_RAW_FILE.exists():
        raise FileNotFoundError(f"请先运行 01_load_clean_merge.py，缺失: {cfg.PANEL_RAW_FILE}")

    df = pd.read_csv(cfg.PANEL_RAW_FILE, dtype={"stkcd": str})
    required = [
        "stkcd",
        "year",
        "revenue",
        "fixed_assets",
        "employees",
        "operating_cost",
        "cash_paid_goods_services",
        "total_assets",
        "total_liabilities",
        "net_profit",
        "operating_cashflow",
        "investing_cashflow",
        "rd_expense",
        "admin_expense",
        "index_aggregate",
        "coverage_breadth",
        "usage_depth",
        "digitization_level",
        "establish_date",
        "list_date",
        "ownership_type",
        "industry_b",
        "industry_c",
        "industry_d",
        "province_norm",
        "city_norm",
    ]
    cfg.require_columns(df, required, "merged_raw_panel")
    initial_rows = len(df)

    df = cfg.to_numeric(df, cfg.FINANCIAL_NUMERIC_COLUMNS + cfg.DFI_VALUE_COLUMNS)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df = df[df["year"].between(cfg.SAMPLE_START, cfg.SAMPLE_END)].copy()
    df["year"] = df["year"].astype(int)
    df = df.sort_values(["stkcd", "year"]).reset_index(drop=True)

    df["Y"] = df["revenue"]
    df["K"] = df["fixed_assets"]
    df["L"] = df["employees"]
    df["M"] = df["operating_cost"].where(df["operating_cost"] > 0, df["cash_paid_goods_services"])
    df["lnY"] = cfg.safe_log(df["Y"])
    df["lnK"] = cfg.safe_log(df["K"])
    df["lnL"] = cfg.safe_log(df["L"])
    df["lnM"] = cfg.safe_log(df["M"])

    df["digital_index"] = df["index_aggregate"]
    df["digital_coverage"] = df["coverage_breadth"]
    df["digital_depth"] = df["usage_depth"]
    df["digital_digitization"] = df["digitization_level"]
    df["lndigital"] = np.log1p(df["digital_index"])
    df["lndigital_coverage"] = np.log1p(df["digital_coverage"])
    df["lndigital_depth"] = np.log1p(df["digital_depth"])
    df["lndigital_digitization"] = np.log1p(df["digital_digitization"])

    df["Size"] = cfg.safe_log(df["total_assets"])
    df["Lev"] = cfg.safe_divide(df["total_liabilities"], df["total_assets"])
    df["ROA"] = cfg.safe_divide(df["net_profit"], df["total_assets"])
    df["Growth"] = df.groupby("stkcd")["revenue"].pct_change(fill_method=None)
    df["Cashflow"] = cfg.safe_divide(df["operating_cashflow"], df["total_assets"])
    df["Tangibility"] = cfg.safe_divide(df["fixed_assets"], df["total_assets"])

    df["rd_missing"] = df["rd_expense"].isna().astype(int)
    df["rd_expense_for_ratio"] = df["rd_expense"].fillna(0)
    df["RD"] = cfg.safe_divide(df["rd_expense_for_ratio"], df["revenue"])
    df["RD_observed"] = cfg.safe_divide(df["rd_expense"], df["revenue"])
    df["lnrd_expense"] = np.log1p(df["rd_expense"].where(df["rd_expense"] >= 0))
    df["AdminRatio"] = cfg.safe_divide(df["admin_expense"], df["revenue"])
    df["InvestCashflow"] = cfg.safe_divide(df["investing_cashflow"], df["total_assets"])

    df["establish_date_dt"] = cfg.parse_datetime(df["establish_date"])
    df["list_date_dt"] = cfg.parse_datetime(df["list_date"])
    establish_year = df["establish_date_dt"].dt.year
    list_year = df["list_date_dt"].dt.year
    base_year = establish_year.fillna(list_year)
    df["Age"] = df["year"] - base_year
    df.loc[df["Age"] < 0, "Age"] = np.nan

    df["SA_size_million"] = cfg.safe_log(df["total_assets"] / 1_000_000)
    df["SA_size_for_formula"] = df["SA_size_million"].clip(upper=np.log(4500))
    df["SA_age_for_formula"] = df["Age"].clip(upper=37)
    df["SA"] = (
        -0.737 * df["SA_size_for_formula"]
        + 0.043 * (df["SA_size_for_formula"] ** 2)
        - 0.040 * df["SA_age_for_formula"]
    )
    df["Turnover"] = cfg.safe_divide(df["revenue"], df["total_assets"])

    df["SOE"] = df["ownership_type"].map(classify_soe)
    df["industry"] = build_industry(df)
    df["manufacturing"] = df["industry"].astype(str).str.contains("制造", na=False).astype(int)
    hightech_pattern = "信息传输|软件|计算机|医药|专用设备|电子|通信|仪器仪表"
    df["hightech"] = df["industry"].astype(str).str.contains(hightech_pattern, na=False).astype(int)
    df["region"] = df["province_norm"].map(cfg.region_group)

    annual_median = df.groupby("year")["Size"].transform("median")
    df["Size_high"] = np.where(df["Size"] > annual_median, 1, 0)
    df.loc[df["Size"].isna() | annual_median.isna(), "Size_high"] = np.nan

    invalid_conditions = {
        "revenue missing_or_<=0": df["revenue"].isna() | (df["revenue"] <= 0),
        "fixed_assets missing_or_<=0": df["fixed_assets"].isna() | (df["fixed_assets"] <= 0),
        "employees missing_or_<=0": df["employees"].isna() | (df["employees"] <= 0),
        "total_assets missing_or_<=0": df["total_assets"].isna() | (df["total_assets"] <= 0),
        "total_liabilities missing_or_<0": df["total_liabilities"].isna() | (df["total_liabilities"] < 0),
        "digital_index missing": df["digital_index"].isna(),
    }
    invalid_counts = {name: int(mask.sum()) for name, mask in invalid_conditions.items()}
    invalid_mask = pd.concat(invalid_conditions.values(), axis=1).any(axis=1)
    df = df.loc[~invalid_mask].copy()

    df = cfg.winsorize_dataframe(df, cfg.WINSOR_VARS)
    df = df.sort_values(["stkcd", "year"]).reset_index(drop=True)
    df["L1_lndigital"] = df.groupby("stkcd")["lndigital"].shift(1)

    df.to_csv(cfg.PANEL_MODEL_FILE, index=False, encoding="utf-8-sig")
    append_variable_report(initial_rows, invalid_counts, len(df))
    print(f"已保存变量构造后面板: {cfg.PANEL_MODEL_FILE} ({len(df)} rows)")


if __name__ == "__main__":
    main()
