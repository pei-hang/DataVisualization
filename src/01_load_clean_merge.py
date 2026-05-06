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


def clean_csmar_panel_table(name: str) -> tuple[pd.DataFrame, dict[str, int]]:
    meta = cfg.CSMAR_TABLES[name]
    raw = cfg.read_csmar_excel_fixed_cols(meta["path"], meta["sheet"], meta["max_col"], name)
    cfg.require_columns(raw, meta["required"], name)

    report = {"raw_rows": len(raw), "columns_read": len(raw.columns)}
    df = raw.copy()
    if meta.get("keep_typrep_a"):
        df["Typrep"] = df["Typrep"].astype(str).str.strip()
        df = df[df["Typrep"] == "A"].copy()
        report["after_typrep_a"] = len(df)

    df = cfg.keep_year_end_sample(df, meta["date_col"], name)
    report["after_year_end_sample"] = len(df)
    df = df.rename(columns=meta["rename"])
    df["stkcd"] = df["stkcd"].map(cfg.normalize_stkcd)
    df = df.dropna(subset=["stkcd", "year"]).copy()

    value_cols = [target for source, target in meta["rename"].items() if target != "stkcd"]
    keep_cols = ["stkcd", "year"] + value_cols
    df = df[keep_cols].copy()
    df = cfg.to_numeric(df, [col for col in value_cols if col not in {"stkcd"}])

    duplicate_count = int(df.duplicated(["stkcd", "year"]).sum())
    report["duplicates_dropped"] = duplicate_count
    df = df.sort_values(["stkcd", "year"]).drop_duplicates(["stkcd", "year"], keep="last")
    report["final_rows"] = len(df)
    return df, report


def clean_info_table() -> tuple[pd.DataFrame, dict[str, int]]:
    meta = cfg.CSMAR_TABLES["info"]
    raw = cfg.read_csmar_excel_fixed_cols(meta["path"], meta["sheet"], meta["max_col"], "info")

    if "Symbol" in raw.columns:
        required = ["Symbol", "ShortName", "EndDate", "IndustryName", "IndustryCode", "LISTINGDATE", "PROVINCE", "CITY"]
        cfg.require_columns(raw, required, "STK_LISTEDCOINFOANL")
        df = raw.rename(
            columns={
                "Symbol": "stkcd",
                "ShortName": "firm_name",
                "EndDate": "info_date",
                "IndustryName": "industry",
                "IndustryCode": "industry_code",
                "LISTINGDATE": "list_date",
                "PROVINCE": "province",
                "CITY": "city",
            }
        ).copy()
        df = cfg.keep_year_end_sample(df, "info_date", "STK_LISTEDCOINFOANL")
        df["industry_a"] = pd.NA
        df["industry_b"] = pd.NA
        df["industry_c"] = pd.NA
        df["industry_d"] = df["industry"]
        df["establish_date"] = pd.NaT
        df["ownership_type"] = pd.NA
        merge_keys = ["stkcd", "year"]
        schema_name = "annual_listed_company_info"
        filled_missing = ["establish_date", "ownership_type", "industry_a", "industry_b", "industry_c"]
    elif "Stkcd" in raw.columns:
        required = [
            "Stkcd",
            "Stknme",
            "Listdt",
            "Indnme",
            "Nindnme",
            "Nnindnme",
            "IndnmeZX",
            "Estbdt",
            "PROVINCE",
            "CITY",
            "OWNERSHIPTYPE",
        ]
        cfg.require_columns(raw, required, "InfoTable")
        df = raw.rename(
            columns={
                "Stkcd": "stkcd",
                "Stknme": "firm_name",
                "Listdt": "list_date",
                "Indnme": "industry_a",
                "Nindnme": "industry_b",
                "Nnindnme": "industry_c",
                "IndnmeZX": "industry_d",
                "Estbdt": "establish_date",
                "PROVINCE": "province",
                "CITY": "city",
                "OWNERSHIPTYPE": "ownership_type",
            }
        ).copy()
        df["industry"] = df[["industry_d", "industry_c", "industry_b", "industry_a"]].replace("", pd.NA).bfill(axis=1).iloc[:, 0]
        df["industry_code"] = pd.NA
        merge_keys = ["stkcd"]
        schema_name = "static_company_info"
        filled_missing = ["industry_code"]
    else:
        raise ValueError(f"info 无法识别公司信息表结构，当前字段: {list(raw.columns)}")

    df["stkcd"] = df["stkcd"].map(cfg.normalize_stkcd)
    df["list_date"] = cfg.parse_datetime(df["list_date"])
    df["establish_date"] = cfg.parse_datetime(df["establish_date"])
    df["province_norm"] = df["province"].map(cfg.normalize_region_name)
    df["city_norm"] = df["city"].map(cfg.normalize_region_name)

    municipality_mask = df["province_norm"].isin(cfg.MUNICIPALITIES)
    bad_city_mask = df["city_norm"].isna() | df["city_norm"].isin(["市辖区", "县", "辖区"])
    df.loc[municipality_mask & bad_city_mask, "city_norm"] = df.loc[municipality_mask & bad_city_mask, "province_norm"]

    duplicate_count = int(df.duplicated(merge_keys).sum())
    df = df.dropna(subset=merge_keys).sort_values(merge_keys).drop_duplicates(merge_keys, keep="last")
    keep_cols = [
        "stkcd",
        "firm_name",
        "list_date",
        "establish_date",
        "industry_a",
        "industry_b",
        "industry_c",
        "industry_d",
        "industry",
        "industry_code",
        "province",
        "city",
        "province_norm",
        "city_norm",
        "ownership_type",
    ]
    if "year" in df.columns:
        keep_cols.insert(1, "year")
    df = df[keep_cols].copy()
    report = {
        "raw_rows": len(raw),
        "columns_read": len(raw.columns),
        "duplicates_dropped": duplicate_count,
        "final_rows": len(df),
        "schema_name": schema_name,
        "merge_key": "+".join(merge_keys),
        "filled_missing": ", ".join(filled_missing),
    }
    return df, report


def clean_dfi_city() -> pd.DataFrame:
    raw = pd.read_excel(cfg.DFI_FILE, sheet_name="Prefecture_Level_Cities")
    cfg.require_columns(raw, cfg.DFI_CITY_COLUMNS, "PKU-DFIIC Prefecture_Level_Cities")
    raw = raw[cfg.DFI_CITY_COLUMNS].copy()
    raw = raw[raw["year"].between(cfg.SAMPLE_START, cfg.SAMPLE_END)].copy()

    frames = []
    for name_col, code_col, source in [
        ("pref_name_year18", "pref_code_year18", "year18"),
        ("pref_name_year14", "pref_code_year14", "year14"),
    ]:
        tmp = raw[["year", name_col, code_col] + cfg.DFI_VALUE_COLUMNS].copy()
        tmp = tmp.rename(columns={name_col: "city_name", code_col: "city_code"})
        tmp["city_norm"] = tmp["city_name"].map(cfg.normalize_region_name)
        tmp["source_name_version"] = source
        frames.append(tmp)

    city = pd.concat(frames, ignore_index=True)
    city = city.dropna(subset=["year", "city_norm"]).copy()
    city["year"] = city["year"].astype(int)
    city = cfg.to_numeric(city, cfg.DFI_VALUE_COLUMNS)
    city = city.sort_values(["year", "city_norm", "source_name_version"])
    city = city.drop_duplicates(["year", "city_norm"], keep="first")
    return city


def clean_dfi_province() -> pd.DataFrame:
    raw = pd.read_excel(cfg.DFI_FILE, sheet_name="Provinces")
    cfg.require_columns(raw, cfg.DFI_PROVINCE_COLUMNS, "PKU-DFIIC Provinces")
    province = raw[cfg.DFI_PROVINCE_COLUMNS].copy()
    province = province[province["year"].between(cfg.SAMPLE_START, cfg.SAMPLE_END)].copy()
    province["year"] = province["year"].astype(int)
    province["province_norm"] = province["prov_name"].map(cfg.normalize_region_name)
    province = cfg.to_numeric(province, cfg.DFI_VALUE_COLUMNS)
    province = province.dropna(subset=["year", "province_norm"]).copy()
    province = province.drop_duplicates(["year", "province_norm"], keep="first")
    return province


def merge_financial_tables(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    panel = tables["balance"]
    for name in ["income", "cashflow", "headcount"]:
        panel = panel.merge(tables[name], on=["stkcd", "year"], how="inner", validate="one_to_one")
    return panel


def attach_dfi(panel: pd.DataFrame, city_dfi: pd.DataFrame, province_dfi: pd.DataFrame) -> pd.DataFrame:
    city_cols = ["year", "city_norm"] + cfg.DFI_VALUE_COLUMNS
    province_cols = ["year", "province_norm"] + cfg.DFI_VALUE_COLUMNS
    city = city_dfi[city_cols].rename(columns={col: f"{col}_city" for col in cfg.DFI_VALUE_COLUMNS})
    province = province_dfi[province_cols].rename(columns={col: f"{col}_province" for col in cfg.DFI_VALUE_COLUMNS})

    out = panel.merge(city, on=["year", "city_norm"], how="left")
    out = out.merge(province, on=["year", "province_norm"], how="left")

    for col in cfg.DFI_VALUE_COLUMNS:
        out[col] = out[f"{col}_city"].combine_first(out[f"{col}_province"])

    city_hit = out["index_aggregate_city"].notna()
    province_hit = out["index_aggregate_province"].notna()
    out["dfi_match_level"] = np.select(
        [city_hit, (~city_hit) & province_hit],
        ["city", "province"],
        default="missing",
    )
    return out


def write_cleaning_report(
    table_reports: dict[str, dict[str, int]],
    info_report: dict[str, int],
    panel: pd.DataFrame,
) -> None:
    match_counts = panel["dfi_match_level"].value_counts(dropna=False).to_dict()
    missing_city = int(panel["city_norm"].isna().sum())
    missing_province = int(panel["province_norm"].isna().sum())
    content = [
        "# 数据清洗报告",
        "",
        f"- 样本期: {cfg.SAMPLE_START}-{cfg.SAMPLE_END}",
        "- CSMAR 表均从第 4 行读取数据，第 1 行作为英文字段名；读取前先扫描第 1 行实际字段宽度，再显式传入 max_col，避免 ws.max_column 异常。",
        "",
        "## CSMAR 表处理行数",
        "",
        "| 表 | 读取列数 | 原始行数 | Typrep=A 后 | 年末样本后 | 删除重复 | 最终行数 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, report in table_reports.items():
        content.append(
            "| {name} | {cols} | {raw} | {typ} | {annual} | {dup} | {final} |".format(
                name=name,
                cols=report.get("columns_read", ""),
                raw=report.get("raw_rows", ""),
                typ=report.get("after_typrep_a", ""),
                annual=report.get("after_year_end_sample", ""),
                dup=report.get("duplicates_dropped", ""),
                final=report.get("final_rows", ""),
            )
        )
    content.extend(
        [
            "",
            "## 公司信息表",
            "",
            f"- 原始行数: {info_report['raw_rows']}",
            f"- 读取列数: {info_report.get('columns_read', '')}",
            f"- 识别结构: {info_report.get('schema_name', '')}",
            f"- 合并键: {info_report.get('merge_key', '')}",
            f"- 删除重复证券代码: {info_report['duplicates_dropped']}",
            f"- 最终公司信息行数: {info_report['final_rows']}",
            f"- 因原表缺失而补为空的字段: {info_report.get('filled_missing', '')}",
            "",
            "## 合并结果",
            "",
            f"- 财务、员工与公司信息合并后样本量: {len(panel)}",
            f"- 城市名称缺失行数: {missing_city}",
            f"- 省份名称缺失行数: {missing_province}",
            f"- 数字普惠金融指数匹配层级: {match_counts}",
            "",
            "后续变量构造脚本会继续删除核心变量非法值、生成模型变量并进行缩尾。",
            "",
        ]
    )
    cfg.write_markdown(cfg.REPORTS_DIR / "data_cleaning_report.md", "\n".join(content))


def main() -> None:
    cfg.ensure_project_dirs()

    panel_tables: dict[str, pd.DataFrame] = {}
    table_reports: dict[str, dict[str, int]] = {}
    for name in ["balance", "income", "cashflow", "headcount"]:
        panel_tables[name], table_reports[name] = clean_csmar_panel_table(name)

    info, info_report = clean_info_table()
    city_dfi = clean_dfi_city()
    province_dfi = clean_dfi_province()
    city_dfi.to_csv(cfg.CITY_DFI_FILE, index=False, encoding="utf-8-sig")
    province_dfi.to_csv(cfg.PROVINCE_DFI_FILE, index=False, encoding="utf-8-sig")

    financial_panel = merge_financial_tables(panel_tables)
    if "year" in info.columns:
        panel = financial_panel.merge(info, on=["stkcd", "year"], how="left", validate="many_to_one")
    else:
        panel = financial_panel.merge(info, on="stkcd", how="left", validate="many_to_one")
    panel = attach_dfi(panel, city_dfi, province_dfi)
    panel = panel.sort_values(["stkcd", "year"]).reset_index(drop=True)
    panel.to_csv(cfg.PANEL_RAW_FILE, index=False, encoding="utf-8-sig")
    write_cleaning_report(table_reports, info_report, panel)
    print(f"已保存合并后原始面板: {cfg.PANEL_RAW_FILE} ({len(panel)} rows)")


if __name__ == "__main__":
    main()
