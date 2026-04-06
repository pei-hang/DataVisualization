from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable
from urllib import parse, request

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


CITY = "南昌"
LATITUDE = 28.6829
LONGITUDE = 115.8582
TIMEZONE = "Asia/Shanghai"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

WEATHER_MAP = {
    0: "晴",
    1: "少云",
    2: "多云",
    3: "阴",
    45: "雾",
    48: "雾凇",
    51: "轻毛毛雨",
    53: "毛毛雨",
    55: "强毛毛雨",
    56: "冻毛毛雨",
    57: "强冻毛毛雨",
    61: "小雨",
    63: "中雨",
    65: "大雨",
    66: "冻雨",
    67: "强冻雨",
    71: "小雪",
    73: "中雪",
    75: "大雪",
    77: "米雪",
    80: "阵雨",
    81: "强阵雨",
    82: "暴雨阵雨",
    85: "阵雪",
    86: "强阵雪",
    95: "雷暴",
    96: "雷暴夹小冰雹",
    99: "雷暴夹大冰雹",
}


def configure_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC",
        "Hiragino Sans GB",
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    OUTPUT_DIR.mkdir(exist_ok=True)


def normalize_fields(fields: Iterable[str] | str | None) -> str | None:
    if fields is None:
        return None
    if isinstance(fields, str):
        return fields
    return ",".join(fields)


class OpenMeteoClient:
    def __init__(self, latitude: float, longitude: float, timezone: str) -> None:
        self.latitude = latitude
        self.longitude = longitude
        self.timezone = timezone

    def _request(self, url: str, **params: object) -> dict:
        query = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "timezone": self.timezone,
        }
        query.update({k: v for k, v in params.items() if v is not None})
        encoded = parse.urlencode(query)
        req = request.Request(
            f"{url}?{encoded}",
            headers={"User-Agent": "DV-Experiment/1.0"},
        )
        with request.urlopen(req, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))

    def forecast(
        self,
        *,
        hourly: Iterable[str] | str | None = None,
        daily: Iterable[str] | str | None = None,
        forecast_days: int | None = None,
        past_days: int | None = None,
    ) -> dict:
        return self._request(
            FORECAST_URL,
            hourly=normalize_fields(hourly),
            daily=normalize_fields(daily),
            forecast_days=forecast_days,
            past_days=past_days,
        )

    def archive(
        self,
        *,
        hourly: Iterable[str] | str | None = None,
        daily: Iterable[str] | str | None = None,
        start_date: str,
        end_date: str,
    ) -> dict:
        return self._request(
            ARCHIVE_URL,
            hourly=normalize_fields(hourly),
            daily=normalize_fields(daily),
            start_date=start_date,
            end_date=end_date,
        )


def hourly_frame(payload: dict) -> pd.DataFrame:
    df = pd.DataFrame(payload["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    return df


def daily_frame(payload: dict) -> pd.DataFrame:
    df = pd.DataFrame(payload["daily"])
    df["time"] = pd.to_datetime(df["time"])
    return df


def save_figure(fig: plt.Figure, filename: str) -> Path:
    output = OUTPUT_DIR / filename
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def save_grid(grid: sns.axisgrid.Grid, filename: str) -> Path:
    output = OUTPUT_DIR / filename
    grid.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(grid.fig)
    return output


def direction_to_cardinal(angle: float) -> str:
    if pd.isna(angle):
        return "未知"
    angle = angle % 360
    if angle >= 315 or angle < 45:
        return "北"
    if angle < 135:
        return "东"
    if angle < 225:
        return "南"
    return "西"


def rolling_window_or_fallback(window: int, size: int) -> int:
    return min(window, max(size // 4, 1))


def recent_date_range(days: int) -> tuple[str, str]:
    end_ts = pd.Timestamp.now(tz=TIMEZONE).normalize() - pd.Timedelta(days=1)
    start_ts = end_ts - pd.Timedelta(days=days - 1)
    return str(start_ts.date()), str(end_ts.date())


def plot_dynamic_scatter(client: OpenMeteoClient) -> list[Path]:
    payload = client.forecast(
        hourly=["temperature_2m", "relative_humidity_2m"],
        forecast_days=3,
    )
    df = hourly_frame(payload)
    df["日期"] = df["time"].dt.strftime("%m-%d")
    df["时间段"] = pd.cut(
        df["time"].dt.hour,
        bins=[-1, 5, 11, 17, 23],
        labels=["凌晨", "上午", "下午", "夜间"],
    )

    g = sns.relplot(
        data=df,
        x="temperature_2m",
        y="relative_humidity_2m",
        hue="时间段",
        col="日期",
        kind="scatter",
        palette="viridis",
        alpha=0.75,
        s=60,
        height=4,
        aspect=1,
    )
    g.set_axis_labels("气温 (°C)", "相对湿度 (%)")
    g.fig.subplots_adjust(top=0.82)
    g.fig.suptitle(f"{CITY}未来 3 天逐小时温湿度动态散点图-裴航")
    return [save_grid(g, "01_dynamic_scatter.png")]


def plot_pair_matrix(client: OpenMeteoClient) -> list[Path]:
    payload = client.forecast(
        hourly=["temperature_2m", "relative_humidity_2m", "wind_speed_10m"],
        forecast_days=7,
    )
    df = hourly_frame(payload)[
        ["temperature_2m", "relative_humidity_2m", "wind_speed_10m"]
    ].dropna()
    df = df.rename(
        columns={
            "temperature_2m": "气温",
            "relative_humidity_2m": "湿度",
            "wind_speed_10m": "风速",
        }
    )

    g = sns.PairGrid(df, diag_sharey=False)
    g.map_upper(sns.scatterplot, alpha=0.55, s=28, color="#2a9d8f")
    g.map_lower(sns.kdeplot, fill=True, cmap="Blues", thresh=0.05)
    g.map_diag(sns.histplot, kde=True, color="#e76f51")
    g.fig.subplots_adjust(top=0.92)
    g.fig.suptitle(f"{CITY}未来 7 天气象变量关系矩阵-裴航")
    return [save_grid(g, "02_pair_matrix.png")]


def plot_week_combo(client: OpenMeteoClient) -> list[Path]:
    payload = client.archive(
        hourly=["temperature_2m", "precipitation"],
        start_date="2024-06-24",
        end_date="2024-06-30",
    )
    df = hourly_frame(payload)
    week_df = (
        df.assign(日期=df["time"].dt.floor("D"))
        .groupby("日期", as_index=False)
        .agg(
            temperature_2m=("temperature_2m", "mean"),
            precipitation=("precipitation", lambda x: int((x > 0).sum())),
        )
        .rename(
            columns={
                "temperature_2m": "日均气温",
                "precipitation": "降雨小时数",
            }
        )
    )

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=False,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    sns.lineplot(
        data=week_df,
        x="日期",
        y="日均气温",
        marker="o",
        linewidth=2.5,
        color="#e76f51",
        ax=ax1,
    )
    ax1.set_title("2024-06-24 至 2024-06-30 温度趋势")
    ax1.set_xlabel("")
    ax1.set_ylabel("日均气温 (°C)")

    heatmap_data = pd.DataFrame(
        [week_df["降雨小时数"].to_list()],
        index=["降雨小时数"],
        columns=week_df["日期"].dt.strftime("%m-%d"),
    )
    sns.heatmap(
        heatmap_data,
        cmap="YlGnBu",
        annot=True,
        fmt=".0f",
        linewidths=0.5,
        cbar_kws={"label": "小时"},
        ax=ax2,
    )
    ax2.set_xlabel("日期")
    ax2.set_ylabel("")

    fig.suptitle(f"{CITY}周数据热力-折线组合图-裴航", y=0.98)
    return [save_figure(fig, "03_week_combo.png")]


def plot_weather_boxen(client: OpenMeteoClient) -> list[Path]:
    payload = client.archive(
        daily=["weather_code", "temperature_2m_max"],
        start_date="2024-01-01",
        end_date="2024-03-31",
    )
    df = daily_frame(payload)
    df["天气类型"] = df["weather_code"].map(WEATHER_MAP)
    df = df[df["天气类型"].notna()].copy()
    order = df["天气类型"].value_counts().index.tolist()

    fig, ax1 = plt.subplots(figsize=(14, 7))
    sns.countplot(
        data=df,
        x="天气类型",
        order=order,
        color="#a8dadc",
        alpha=0.85,
        ax=ax1,
    )
    ax1.set_ylabel("出现天数")
    ax1.set_xlabel("天气类型")
    ax1.set_title("2024 年第一季度天气类型频次与温度极值分布-裴航")

    ax2 = ax1.twinx()
    ax2.patch.set_alpha(0)
    sns.boxenplot(
        data=df,
        x="天气类型",
        y="temperature_2m_max",
        order=order,
        color="#e76f51",
        saturation=0.65,
        linewidth=1.2,
        ax=ax2,
    )
    ax2.set_ylabel("日最高气温 (°C)")

    for axis in (ax1, ax2):
        axis.tick_params(axis="x", rotation=25)

    return [save_figure(fig, "04_weather_boxen.png")]


def plot_wind_facets(client: OpenMeteoClient) -> list[Path]:
    payload = client.forecast(
        hourly=["wind_direction_10m", "wind_speed_10m"],
        forecast_days=7,
    )
    df = hourly_frame(payload).dropna()
    df["风向"] = df["wind_direction_10m"].apply(direction_to_cardinal)
    speed_order = ["微风", "和风", "强风", "烈风", "暴风"]
    df["风速等级"] = pd.cut(
        df["wind_speed_10m"],
        bins=[-np.inf, 3, 6, 9, 12, np.inf],
        labels=speed_order,
    )
    direction_order = ["北", "东", "南", "西"]

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.scatterplot(
        data=df,
        x="风向",
        y="wind_speed_10m",
        hue="风速等级",
        hue_order=speed_order,
        palette="crest",
        alpha=0.75,
        s=60,
        ax=ax,
    )
    ax.set_title(f"{CITY}未来一周风向-风速基础散点图-裴航")
    ax.set_xlabel("风向")
    ax.set_ylabel("风速 (m/s)")
    scatter_path = save_figure(fig, "05_wind_scatter.png")

    g = sns.FacetGrid(
        df,
        col="风速等级",
        col_wrap=3,
        col_order=speed_order,
        height=3.8,
        sharey=True,
    )
    g.map_dataframe(
        sns.stripplot,
        x="风向",
        y="wind_speed_10m",
        order=direction_order,
        jitter=0.25,
        alpha=0.7,
        color="#457b9d",
    )
    g.set_axis_labels("风向", "风速 (m/s)")
    g.fig.subplots_adjust(top=0.88)
    g.fig.suptitle(f"{CITY}未来一周分类网格组合图-裴航")
    facet_path = save_grid(g, "06_wind_facet_grid.png")
    return [scatter_path, facet_path]


def plot_regression_grid(client: OpenMeteoClient) -> list[Path]:
    start_date, end_date = recent_date_range(30)
    payload = client.archive(
        hourly=[
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
            "cloud_cover",
        ],
        start_date=start_date,
        end_date=end_date,
    )
    df = hourly_frame(payload).dropna()
    label_map = {
        "relative_humidity_2m": "相对湿度",
        "wind_speed_10m": "风速",
        "cloud_cover": "云量",
    }
    long_df = df.melt(
        id_vars=["temperature_2m"],
        value_vars=list(label_map),
        var_name="变量名",
        value_name="观测值",
    )
    long_df["变量名"] = long_df["变量名"].map(label_map)

    g = sns.lmplot(
        data=long_df,
        x="观测值",
        y="temperature_2m",
        col="变量名",
        col_wrap=2,
        height=4.2,
        scatter_kws={"alpha": 0.35, "s": 18, "color": "#1d3557"},
        line_kws={"color": "#d62828", "linewidth": 2},
        facet_kws={"sharex": False, "sharey": True},
    )
    g.set_axis_labels("观测值", "气温 (°C)")
    g.fig.subplots_adjust(top=0.88)
    g.fig.suptitle(f"{CITY}近 30 天多维度回归网格-裴航")
    lmplot_path = save_grid(g, "07_regression_grid.png")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()
    for ax, column in zip(axes, label_map):
        sns.residplot(
            data=df,
            x=column,
            y="temperature_2m",
            scatter_kws={"alpha": 0.35, "s": 18, "color": "#457b9d"},
            line_kws={"color": "#d62828", "linewidth": 2},
            ax=ax,
        )
        ax.set_title(f"{label_map[column]}残差视图")
        ax.set_xlabel(label_map[column])
        ax.set_ylabel("气温残差")
    axes[-1].set_visible(False)
    fig.suptitle(f"{CITY}近 30 天回归残差分析-裴航", y=0.98)
    resid_path = save_figure(fig, "08_regression_residuals.png")
    return [lmplot_path, resid_path]


def plot_time_series_regression(client: OpenMeteoClient) -> list[Path]:
    start_date, end_date = recent_date_range(30)
    payload = client.archive(
        hourly=["temperature_2m"],
        start_date=start_date,
        end_date=end_date,
    )
    df = hourly_frame(payload).dropna().reset_index(drop=True)
    window = rolling_window_or_fallback(7 * 24, len(df))
    df["7d_MA"] = df["temperature_2m"].rolling(
        window=window,
        min_periods=min(24, window),
    ).mean()
    df["hour_index"] = np.arange(len(df))

    slope, intercept = np.polyfit(df["hour_index"], df["temperature_2m"], 1)
    df["trend"] = slope * df["hour_index"] + intercept

    residual = df["temperature_2m"] - df["7d_MA"].fillna(df["temperature_2m"].mean())
    anomalies = df.loc[residual.abs().nlargest(3).index].sort_values("hour_index")

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.lineplot(
        data=df,
        x="hour_index",
        y="temperature_2m",
        linewidth=1.2,
        alpha=0.4,
        color="#457b9d",
        label="逐小时气温",
        ax=ax,
    )
    sns.lineplot(
        data=df,
        x="hour_index",
        y="7d_MA",
        linewidth=2.6,
        color="#2a9d8f",
        label="7 日移动平均",
        ax=ax,
    )
    sns.regplot(
        data=df,
        x="hour_index",
        y="temperature_2m",
        scatter=False,
        ci=95,
        line_kws={"color": "#d62828", "linewidth": 2},
        ax=ax,
    )

    ax.scatter(
        anomalies["hour_index"],
        anomalies["temperature_2m"],
        color="#d62828",
        s=50,
        zorder=5,
        label="异常点",
    )
    for row in anomalies.itertuples():
        ax.annotate(
            row.time.strftime("%m-%d %H:%M"),
            (row.hour_index, row.temperature_2m),
            textcoords="offset points",
            xytext=(5, 8),
            fontsize=9,
        )

    tick_positions = df.loc[df["time"].dt.hour == 0, "hour_index"]
    tick_labels = df.loc[df["time"].dt.hour == 0, "time"].dt.strftime("%m-%d")
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=30)
    ax.set_xlabel("日期")
    ax.set_ylabel("气温 (°C)")
    ax.set_title(f"{CITY}近 30 天气温时间序列回归分析-裴航")
    ax.text(
        0.02,
        0.95,
        f"回归方程: y = {slope:.4f}x + {intercept:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
    )
    return [save_figure(fig, "09_time_series_regression.png")]


def plot_pairgrid_network(client: OpenMeteoClient) -> list[Path]:
    start_date, end_date = recent_date_range(14)
    payload = client.archive(
        hourly=["apparent_temperature", "relative_humidity_2m", "wind_speed_10m"],
        start_date=start_date,
        end_date=end_date,
    )
    df = hourly_frame(payload)[
        ["apparent_temperature", "relative_humidity_2m", "wind_speed_10m"]
    ].dropna()
    df = df.rename(
        columns={
            "apparent_temperature": "体感温度",
            "relative_humidity_2m": "相对湿度",
            "wind_speed_10m": "风速",
        }
    )

    g = sns.PairGrid(df, diag_sharey=False)
    g.map_upper(sns.scatterplot, alpha=0.45, s=20, color="#264653")
    g.map_diag(sns.histplot, kde=True, color="#f4a261")
    g.map_lower(sns.kdeplot, fill=True, thresh=0.05, cmap="YlOrBr")
    g.fig.subplots_adjust(top=0.92)
    g.fig.suptitle(f"{CITY}综合关系网络图-裴航")
    return [save_grid(g, "10_pairgrid_network.png")]


def main() -> None:
    configure_style()
    client = OpenMeteoClient(LATITUDE, LONGITUDE, TIMEZONE)

    tasks = [
        ("1.1 动态双变量散点图", plot_dynamic_scatter),
        ("1.2 矩阵网格对比图", plot_pair_matrix),
        ("1.3 热力-折线组合图", plot_week_combo),
        ("2.1 增强箱线图", plot_weather_boxen),
        ("2.2 分类网格组合图", plot_wind_facets),
        ("3.1 多维度回归网格", plot_regression_grid),
        ("3.2 时间序列回归", plot_time_series_regression),
        ("3.3 综合关系网络图", plot_pairgrid_network),
    ]

    generated: list[Path] = []
    print(f"开始生成 {CITY}天气数据可视化，输出目录: {OUTPUT_DIR}")
    for name, task in tasks:
        try:
            outputs = task(client)
            generated.extend(outputs)
            print(f"[完成] {name}")
        except Exception as exc:
            print(f"[失败] {name}: {exc}")

    if generated:
        print("\n已生成文件：")
        for path in generated:
            print(f"- {path}")


if __name__ == "__main__":
    main()
