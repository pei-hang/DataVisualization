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


def prepare_ml_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    cfg.require_columns(df, ["tfp_ols_m", "year", "industry"] + cfg.ML_BASE_FEATURES, "panel_model_data_with_tfp")
    work = df.copy()
    for col in cfg.ML_BASE_FEATURES + ["tfp_ols_m"]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=["tfp_ols_m", "year"]).copy()

    base = work[cfg.ML_BASE_FEATURES].copy()
    year_dummies = pd.get_dummies(work["year"].astype(int), prefix="year", dtype=float)
    industry_dummies = pd.get_dummies(work["industry"].fillna("Unknown").astype(str), prefix="industry", dtype=float)
    x = pd.concat([base, year_dummies, industry_dummies], axis=1)
    x = x.replace([np.inf, -np.inf], np.nan)
    all_missing_features = x.columns[x.isna().all()].tolist()
    if all_missing_features:
        x = x.drop(columns=all_missing_features)
        x.attrs["dropped_all_missing_features"] = all_missing_features
    y = work["tfp_ols_m"].astype(float)
    years = work["year"].astype(int)
    return x, y, years


def plot_native_shap_summary(shap_values: np.ndarray, x_sample: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    # 解决Mac中文显示
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    # 解决负号显示方块
    plt.rcParams['axes.unicode_minus'] = False

    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[-30:]
    top_idx = top_idx[np.argsort(mean_abs[top_idx])]
    plt.figure(figsize=(9, max(6, 0.28 * len(top_idx))))
    rng = np.random.default_rng(cfg.RANDOM_STATE)
    for y_pos, idx in enumerate(top_idx):
        feature_values = x_sample.iloc[:, idx]
        valid = pd.to_numeric(feature_values, errors="coerce")
        colors = valid.to_numpy(dtype=float)
        jitter = rng.normal(0, 0.08, size=len(x_sample))
        plt.scatter(
            shap_values[:, idx],
            np.full(len(x_sample), y_pos) + jitter,
            c=colors,
            cmap="viridis",
            s=8,
            alpha=0.55,
            linewidths=0,
        )
    plt.yticks(range(len(top_idx)), x_sample.columns[top_idx])
    plt.axvline(0, color="#555555", linewidth=0.8)
    plt.xlabel("SHAP value")
    plt.ylabel("")
    cbar = plt.colorbar()
    cbar.set_label("Feature value")
    plt.tight_layout()
    plt.savefig(output_path, dpi=cfg.FIG_DPI, bbox_inches="tight")
    plt.close()


def plot_native_shap_dependence(shap_values: np.ndarray, x_sample: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt


    # 解决Mac中文显示
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    # 解决负号显示方块
    plt.rcParams['axes.unicode_minus'] = False

    if "lndigital" not in x_sample.columns:
        raise ValueError("SHAP dependence plot 需要特征 lndigital。")
    idx = x_sample.columns.get_loc("lndigital")
    color = x_sample["lndigital_depth"] if "lndigital_depth" in x_sample.columns else x_sample["lndigital"]
    plt.figure(figsize=(8, 5))
    plt.scatter(x_sample["lndigital"], shap_values[:, idx], c=color, cmap="viridis", s=14, alpha=0.65, linewidths=0)
    plt.axhline(0, color="#555555", linewidth=0.8, linestyle="--")
    plt.xlabel("lndigital")
    plt.ylabel("SHAP value for lndigital")
    cbar = plt.colorbar()
    cbar.set_label("lndigital_depth")
    plt.tight_layout()
    plt.savefig(output_path, dpi=cfg.FIG_DPI, bbox_inches="tight")
    plt.close()


def save_shap_plots(model, x_sample: pd.DataFrame) -> str:
    try:
        import shap

        import matplotlib.pyplot as plt

        # 解决Mac中文显示
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
        # 解决负号显示方块
        plt.rcParams['axes.unicode_minus'] = False

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_sample)
        plt.figure()
        shap.summary_plot(shap_values, x_sample, show=False, max_display=30)
        plt.tight_layout()
        plt.savefig(cfg.FIGURES_DIR / "shap_summary.png", dpi=cfg.FIG_DPI, bbox_inches="tight")
        plt.close()

        plt.figure()
        shap.dependence_plot("lndigital", shap_values, x_sample, show=False)
        plt.tight_layout()
        plt.savefig(cfg.FIGURES_DIR / "shap_lndigital_dependence.png", dpi=cfg.FIG_DPI, bbox_inches="tight")
        plt.close()
        return "shap.TreeExplainer"
    except Exception as shap_error:
        print(f"shap 包不可用或执行失败，改用 XGBoost 原生 TreeSHAP。原因: {shap_error}")
        from xgboost import DMatrix

        dmatrix = DMatrix(x_sample, feature_names=list(x_sample.columns))
        contribs = model.get_booster().predict(dmatrix, pred_contribs=True)
        shap_values = contribs[:, :-1]
        plot_native_shap_summary(shap_values, x_sample, cfg.FIGURES_DIR / "shap_summary.png")
        plot_native_shap_dependence(shap_values, x_sample, cfg.FIGURES_DIR / "shap_lndigital_dependence.png")
        return "xgboost native TreeSHAP pred_contribs"


def main() -> None:
    cfg.ensure_project_dirs()
    if not cfg.PANEL_TFP_FILE.exists():
        raise FileNotFoundError(f"请先运行 03_tfp_estimation.py，缺失: {cfg.PANEL_TFP_FILE}")

    import matplotlib


    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 解决Mac中文显示
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    # 解决负号显示方块
    plt.rcParams['axes.unicode_minus'] = False
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    try:
        from xgboost import XGBRegressor
    except Exception as exc:
        status = pd.DataFrame(
            [
                {
                    "status": "not_run",
                    "reason": "无法加载 xgboost；macOS pip 版 XGBoost 通常需要 OpenMP 运行库 libomp.dylib。",
                    "fix": "安装 libomp 后重新运行 python src/06_xgboost_shap.py，例如 brew install libomp。",
                }
            ]
        )
        with pd.ExcelWriter(cfg.TABLES_DIR / "ml_feature_importance.xlsx", engine="openpyxl") as writer:
            status.to_excel(writer, sheet_name="status", index=False)
        for figure_name in ["xgboost_feature_importance.png", "shap_summary.png", "shap_lndigital_dependence.png"]:
            figure_path = cfg.FIGURES_DIR / figure_name
            if figure_path.exists():
                figure_path.unlink()
        cfg.append_markdown(
            cfg.REPORTS_DIR / "model_interpretation.md",
            "\n".join(
                [
                    "",
                    "## XGBoost / SHAP 运行状态",
                    "",
                    "- `06_xgboost_shap.py` 未能加载 XGBoost。macOS pip 版 XGBoost 通常需要 OpenMP 运行库 `libomp.dylib`；安装 `libomp` 后可重新运行该脚本。",
                    "",
                ]
            ),
        )
        raise RuntimeError(
            "无法加载 xgboost。macOS 使用 pip 安装 xgboost 时通常还需要 OpenMP 运行库。"
            "请先安装 libomp，例如: brew install libomp；或使用 conda-forge 环境安装 py-xgboost。"
        ) from exc

    df = pd.read_csv(cfg.PANEL_TFP_FILE, dtype={"stkcd": str})
    x, y, years = prepare_ml_matrix(df)
    dropped_features = x.attrs.get("dropped_all_missing_features", [])
    train_mask = years.between(2016, 2021)
    test_mask = years.between(2022, 2023)
    if train_mask.sum() == 0 or test_mask.sum() == 0:
        raise ValueError("XGBoost 训练集或测试集为空；需要 2016-2021 训练、2022-2023 测试。")

    x_train, y_train = x.loc[train_mask], y.loc[train_mask]
    x_test, y_test = x.loc[test_mask], y.loc[test_mask]

    model = XGBRegressor(
        n_estimators=500,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=cfg.RANDOM_STATE,
        n_jobs=-1,
        reg_lambda=1.0,
    )
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    rmse = float(mean_squared_error(y_test, pred) ** 0.5)
    mae = float(mean_absolute_error(y_test, pred))
    r2 = float(r2_score(y_test, pred)) if len(y_test) > 1 else np.nan

    importance = pd.DataFrame(
        {
            "feature": x.columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    metrics = pd.DataFrame([{"RMSE": rmse, "MAE": mae, "R2": r2, "train_n": len(y_train), "test_n": len(y_test)}])
    feature_metadata = pd.DataFrame({"dropped_all_missing_features": dropped_features})
    with pd.ExcelWriter(cfg.TABLES_DIR / "ml_feature_importance.xlsx", engine="openpyxl") as writer:
        metrics.to_excel(writer, sheet_name="metrics", index=False)
        importance.to_excel(writer, sheet_name="feature_importance", index=False)
        feature_metadata.to_excel(writer, sheet_name="feature_metadata", index=False)

    top = importance.head(30).sort_values("importance")
    plt.figure(figsize=(9, 8))
    plt.barh(top["feature"], top["importance"], color="#2f6f73")
    plt.xlabel("Feature importance")
    plt.tight_layout()
    plt.savefig(cfg.FIGURES_DIR / "xgboost_feature_importance.png", dpi=cfg.FIG_DPI)
    plt.close()

    sample_size = min(2000, len(x_test))
    x_sample = x_test.sample(n=sample_size, random_state=cfg.RANDOM_STATE) if len(x_test) > sample_size else x_test
    shap_engine = save_shap_plots(model, x_sample)

    cfg.append_markdown(
        cfg.REPORTS_DIR / "model_interpretation.md",
        "\n".join(
            [
                "",
                "## XGBoost / SHAP 辅助分析",
                "",
                f"- 测试集 RMSE={rmse:.4f}，MAE={mae:.4f}，R2={cfg.finite_or_blank(r2)}。",
                f"- SHAP 计算方式: {shap_engine}。",
                "- XGBoost 只用于变量重要性和非线性关系发现，不用于因果识别；因果解释仍以固定效应模型及稳健性检验为主。",
                "",
            ]
        ),
    )
    print(f"已保存 XGBoost/SHAP 结果: {cfg.TABLES_DIR / 'ml_feature_importance.xlsx'}")


if __name__ == "__main__":
    main()
