from __future__ import annotations

import json
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any, Iterable
from urllib import parse, request

import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import Bar, Geo, HeatMap, Line, Radar, Timeline
from pyecharts.commons.utils import JsCode
from pyecharts.globals import GeoType, ThemeType
from pyecharts.render.engine import RenderEngine


TIMEZONE = "Asia/Shanghai"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs" / "experiment6"


@dataclass(frozen=True)
class CityMeta:
    name: str
    lat: float
    lon: float
    label_pos: str


@dataclass
class ChartBundle:
    chart: Any
    height: str
    column_class: str = "wide"


CITIES = [
    CityMeta("南昌", 28.6829, 115.8582, "left"),
    CityMeta("长沙", 28.2282, 112.9388, "bottom"),
    CityMeta("武汉", 30.5928, 114.3055, "top"),
    CityMeta("南京", 32.0603, 118.7969, "bottom"),
    CityMeta("上海", 31.2304, 121.4737, "right"),
]
CITY_MAP = {city.name: city for city in CITIES}

RADAR_METRICS = ["平均最高温", "总降水量", "平均风速", "高温日占比"]
RADAR_COLORS = ["#d94841", "#2a9d8f", "#457b9d", "#f4a261", "#7b2cbf"]

PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{title}</title>
  {head_scripts}
  <style>
    * {{
      box-sizing: border-box;
    }}
    body {{
      margin: 0;
      font-family: "PingFang SC", "Microsoft YaHei", sans-serif;
      background: linear-gradient(180deg, #eff5ff 0%, #ffffff 100%);
      color: #1f2937;
    }}
    .page {{
      max-width: 1460px;
      margin: 0 auto;
      padding: 24px 18px 36px;
    }}
    .title {{
      margin: 0 0 10px;
      font-size: 28px;
      font-weight: 700;
      letter-spacing: 0.5px;
    }}
    .subtitle {{
      margin: 0 0 20px;
      color: #475569;
      line-height: 1.7;
      font-size: 15px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(12, minmax(0, 1fr));
      gap: 18px;
    }}
    .card {{
      grid-column: span 12;
      background: rgba(255, 255, 255, 0.96);
      border: 1px solid rgba(148, 163, 184, 0.18);
      border-radius: 18px;
      box-shadow: 0 18px 50px rgba(148, 163, 184, 0.16);
      padding: 10px;
    }}
    .card.half {{
      grid-column: span 6;
    }}
    .chart-box {{
      width: 100%;
    }}
    @media (max-width: 960px) {{
      .card.half {{
        grid-column: span 12;
      }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <h1 class="title">{title}</h1>
    <p class="subtitle">{subtitle}</p>
    <section class="grid">
      {body}
    </section>
  </main>
  {chart_scripts}
  {extra_scripts}
</body>
</html>
"""


def normalize_fields(fields: Iterable[str] | str | None) -> str | None:
    if fields is None:
        return None
    if isinstance(fields, str):
        return fields
    return ",".join(fields)


def unique_keep_order(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


class OpenMeteoClient:
    def __init__(self, latitude: float, longitude: float, timezone: str = TIMEZONE) -> None:
        self.latitude = latitude
        self.longitude = longitude
        self.timezone = timezone

    def _request(self, url: str, **params: object) -> dict:
        query = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "timezone": self.timezone,
        }
        query.update({key: value for key, value in params.items() if value is not None})
        req = request.Request(
            f"{url}?{parse.urlencode(query)}",
            headers={"User-Agent": "DV-Experiment6/1.0"},
        )
        with request.urlopen(req, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))

    def forecast(
        self,
        *,
        hourly: Iterable[str] | str | None = None,
        daily: Iterable[str] | str | None = None,
        current: Iterable[str] | str | None = None,
        forecast_days: int | None = None,
        past_days: int | None = None,
    ) -> dict:
        return self._request(
            FORECAST_URL,
            hourly=normalize_fields(hourly),
            daily=normalize_fields(daily),
            current=normalize_fields(current),
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
    frame = pd.DataFrame(payload["hourly"])
    frame["time"] = pd.to_datetime(frame["time"])
    return frame


def daily_frame(payload: dict) -> pd.DataFrame:
    frame = pd.DataFrame(payload["daily"])
    frame["time"] = pd.to_datetime(frame["time"])
    return frame


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def city_client(city_name: str) -> OpenMeteoClient:
    city = CITY_MAP[city_name]
    return OpenMeteoClient(city.lat, city.lon)


def compute_z_scores(values: dict[str, float]) -> dict[str, float]:
    series = pd.Series(values, dtype="float64")
    std = float(series.std(ddof=0))
    if std == 0:
        return {key: 0.0 for key in values}
    mean_value = float(series.mean())
    return {key: round((value - mean_value) / std, 3) for key, value in values.items()}


def fetch_recent_week_hourly(city_name: str) -> pd.DataFrame:
    payload = city_client(city_name).forecast(
        hourly=["temperature_2m", "precipitation"],
        forecast_days=1,
        past_days=7,
    )
    frame = hourly_frame(payload)[["time", "temperature_2m", "precipitation"]].dropna()
    return frame.tail(24 * 7).reset_index(drop=True)


def fetch_august_city_data(city_name: str) -> pd.DataFrame:
    payload = city_client(city_name).archive(
        daily=["temperature_2m_max", "precipitation_sum"],
        hourly=["wind_speed_10m"],
        start_date="2025-08-01",
        end_date="2025-08-31",
    )
    daily_df = daily_frame(payload)[["time", "temperature_2m_max", "precipitation_sum"]]
    hourly_df = hourly_frame(payload)[["time", "wind_speed_10m"]]

    wind_daily = (
        hourly_df.assign(date=hourly_df["time"].dt.floor("D"))
        .groupby("date", as_index=False)["wind_speed_10m"]
        .mean()
        .rename(columns={"date": "time", "wind_speed_10m": "wind_speed_10m_mean"})
    )
    merged = daily_df.merge(wind_daily, on="time", how="left")
    merged["precipitation_sum"] = merged["precipitation_sum"].fillna(0.0)
    merged["wind_speed_10m_mean"] = merged["wind_speed_10m_mean"].fillna(
        merged["wind_speed_10m_mean"].mean()
    )
    merged["hot_day"] = (merged["temperature_2m_max"] > 30).astype(int)
    merged["city"] = city_name
    return merged


def fetch_realtime_and_forecast(city_name: str) -> tuple[dict[str, float], pd.DataFrame]:
    payload = city_client(city_name).forecast(
        current=["temperature_2m", "wind_speed_10m"],
        hourly=["temperature_2m"],
        forecast_days=2,
    )
    current_time = pd.to_datetime(payload["current"]["time"]).floor("h")
    current_data = {
        "temperature_2m": float(payload["current"]["temperature_2m"]),
        "wind_speed_10m": float(payload["current"]["wind_speed_10m"]),
    }
    hourly_df = hourly_frame(payload)[["time", "temperature_2m"]].dropna()
    future_df = hourly_df[hourly_df["time"] >= current_time].head(24).copy()
    if len(future_df) < 24:
        future_df = hourly_df.head(24).copy()
    return current_data, future_df.reset_index(drop=True)


def build_hourly_combo_chart(df: pd.DataFrame) -> ChartBundle:
    labels = df["time"].dt.strftime("%m-%d %H:%M").tolist()
    bar = (
        Bar(init_opts=opts.InitOpts(theme=ThemeType.LIGHT, chart_id="hourly_combo_chart"))
        .add_xaxis(labels)
        .add_yaxis(
            "每小时降水量 (mm)",
            df["precipitation"].round(2).tolist(),
            yaxis_index=1,
            color="#4ea8de",
            category_gap="55%",
            label_opts=opts.LabelOpts(is_show=False),
        )
        .extend_axis(
            yaxis=opts.AxisOpts(
                name="降水量 (mm)",
                min_=0,
                position="right",
                splitline_opts=opts.SplitLineOpts(is_show=False),
            )
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="南昌过去7天逐小时温度与降水变化"),
            tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
            legend_opts=opts.LegendOpts(pos_top="7%"),
            xaxis_opts=opts.AxisOpts(
                name="时间",
                axislabel_opts=opts.LabelOpts(rotate=38, interval=7),
            ),
            yaxis_opts=opts.AxisOpts(name="温度 (°C)"),
            datazoom_opts=[
                opts.DataZoomOpts(type_="inside", range_start=0, range_end=45),
                opts.DataZoomOpts(
                    type_="slider",
                    range_start=0,
                    range_end=45,
                    pos_bottom="0%",
                ),
            ],
        )
    )
    line = (
        Line()
        .add_xaxis(labels)
        .add_yaxis(
            "每小时温度 (°C)",
            df["temperature_2m"].round(2).tolist(),
            is_smooth=True,
            color="#e76f51",
            symbol="circle",
            symbol_size=6,
            linestyle_opts=opts.LineStyleOpts(width=3),
            label_opts=opts.LabelOpts(is_show=False),
        )
    )
    bar.overlap(line)
    return ChartBundle(bar, height="600px")


def build_august_analysis_charts(
    august_frames: dict[str, pd.DataFrame],
) -> tuple[ChartBundle, ChartBundle, str]:
    summary = pd.DataFrame(
        [
            {
                "城市": city_name,
                "平均最高温": float(frame["temperature_2m_max"].mean()),
                "总降水量": float(frame["precipitation_sum"].sum()),
                "平均风速": float(frame["wind_speed_10m_mean"].mean()),
                "高温日占比": float(frame["hot_day"].mean() * 100),
            }
            for city_name, frame in august_frames.items()
        ]
    )

    zscore_map = {
        metric: compute_z_scores(dict(zip(summary["城市"], summary[metric])))
        for metric in RADAR_METRICS
    }

    radar = Radar(
        init_opts=opts.InitOpts(
            theme=ThemeType.LIGHT,
            chart_id="city_radar_chart",
        )
    )
    radar.add_schema(
        schema=[
            opts.RadarIndicatorItem(name=metric, max_=2.5, min_=-2.5)
            for metric in RADAR_METRICS
        ],
        splitarea_opt=opts.SplitAreaOpts(
            is_show=True,
            areastyle_opts=opts.AreaStyleOpts(opacity=0.08),
        ),
        shape="circle",
        textstyle_opts=opts.TextStyleOpts(font_size=13),
    )

    for index, city_name in enumerate(summary["城市"].tolist()):
        radar.add(
            city_name,
            [[zscore_map[metric][city_name] for metric in RADAR_METRICS]],
            color=RADAR_COLORS[index],
            areastyle_opts=opts.AreaStyleOpts(opacity=0.12),
            linestyle_opts=opts.LineStyleOpts(width=2.2),
            label_opts=opts.LabelOpts(is_show=False),
        )

    radar.set_global_opts(
        title_opts=opts.TitleOpts(
            title="五城市 2025 年 8 月综合气象特征雷达图",
            pos_left="center",
        ),
        legend_opts=opts.LegendOpts(
            pos_top="6%",
            pos_left="center",
        ),
    )

    heatmap_labels = ["最高温", "降水量", "平均风速", "高温日"]
    corr_map: dict[str, list[list[float]]] = {}

    for city_name, frame in august_frames.items():
        corr_df = frame[
            ["temperature_2m_max", "precipitation_sum", "wind_speed_10m_mean", "hot_day"]
        ].copy()

        corr_df["hot_day"] = corr_df["hot_day"].astype(int)

        corr_df = corr_df.rename(
            columns={
                "temperature_2m_max": heatmap_labels[0],
                "precipitation_sum": heatmap_labels[1],
                "wind_speed_10m_mean": heatmap_labels[2],
                "hot_day": heatmap_labels[3],
            }
        )

        corr = (
            corr_df[heatmap_labels]
            .corr()
            .reindex(index=heatmap_labels, columns=heatmap_labels)
            .fillna(0.0)
            .round(3)
        )

        pairs: list[list[float]] = []
        for row_index in range(len(heatmap_labels)):
            for col_index in range(len(heatmap_labels)):
                pairs.append([col_index, row_index, float(corr.iloc[row_index, col_index])])
        corr_map[city_name] = pairs

    initial_city = CITIES[0].name

    heatmap = (
        HeatMap(
            init_opts=opts.InitOpts(
                theme=ThemeType.LIGHT,
                chart_id="city_heatmap_chart",
            )
        )
        .add_xaxis(heatmap_labels)
        .add_yaxis(
            "相关系数",
            heatmap_labels,
            corr_map[initial_city],
            label_opts=opts.LabelOpts(
                is_show=True,
                position="inside",
                formatter=JsCode("function(params){ return params.value[2].toFixed(2); }"),
                font_size=12,
            ),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title=f"{initial_city} 2025 年 8 月气象指标相关性热力图",
                pos_left="center",
            ),
            legend_opts=opts.LegendOpts(is_show=False),
            visualmap_opts=opts.VisualMapOpts(
                min_=-1,
                max_=1,
                range_color=["#1d3557", "#74c0fc", "#f8f9fa", "#ffd166", "#d62828"],
                pos_left="3%",
                pos_bottom="8%",
            ),
            xaxis_opts=opts.AxisOpts(position="top"),
            yaxis_opts=opts.AxisOpts(),
        )
    )

    linkage_script = f"""
    <script>
      (function () {{
        var correlationByCity = {json.dumps(corr_map, ensure_ascii=False)};

        function switchHeatmap(cityName) {{
          if (!correlationByCity[cityName]) {{
            return;
          }}
          chart_city_heatmap_chart.setOption({{
            title: {{ text: cityName + " 2025 年 8 月气象指标相关性热力图" }},
            series: [{{
              data: correlationByCity[cityName],
              label: {{
                show: true,
                position: "inside",
                formatter: function(params) {{
                  return Number(params.value[2]).toFixed(2);
                }}
              }}
            }}]
          }});
        }}

        chart_city_radar_chart.on("click", function (params) {{
          if (params.seriesName) {{
            switchHeatmap(params.seriesName);
          }}
        }});

        chart_city_radar_chart.on("legendselectchanged", function (params) {{
          if (params.name && params.selected[params.name]) {{
            switchHeatmap(params.name);
            return;
          }}
          Object.keys(params.selected).some(function (cityName) {{
            if (params.selected[cityName]) {{
              switchHeatmap(cityName);
              return true;
            }}
            return false;
          }});
        }});
      }})();
    </script>
    """

    return (
        ChartBundle(radar, height="560px", column_class="half"),
        ChartBundle(heatmap, height="560px", column_class="half"),
        linkage_script,
    )


def build_geo_chart(realtime_data: dict[str, dict[str, float]]) -> ChartBundle:
    geo = Geo(init_opts=opts.InitOpts(theme=ThemeType.WALDEN, chart_id="city_geo_chart"))

    for city in CITIES:
        geo.add_coordinate(city.name, city.lon, city.lat)

    geo.add_schema(
        maptype="china",
        itemstyle_opts=opts.ItemStyleOpts(color="#f2f7ff", border_color="#a9c4de", border_width=1),
        emphasis_itemstyle_opts=opts.ItemStyleOpts(color="#dcecff"),
    )

    # 只把温度放进 value，风速通过名字映射读取，避免 Geo 对多维值的嵌套差异。
    for city in CITIES:
        temp = round(realtime_data[city.name]["temperature_2m"], 1)
        wind = round(realtime_data[city.name]["wind_speed_10m"], 1)
        symbol_size = max(12, min(26, 8 + wind * 1.15))
        geo.add(
            city.name,
            [
                opts.GeoItem(
                    name=city.name,
                    longitude=city.lon,
                    latitude=city.lat,
                    value=temp,
                )
            ],
            type_=GeoType.SCATTER,
            symbol="circle",
            symbol_size=symbol_size,
            label_opts=opts.LabelOpts(
                is_show=True,
                position=city.label_pos,
                color="#1f2937",
                font_size=12,
                formatter=f"{city.name}\n{temp:.1f}°C",
            ),
            itemstyle_opts=opts.ItemStyleOpts(border_color="#ffffff", border_width=1.5, opacity=0.9),
            tooltip_opts=opts.TooltipOpts(
                formatter=f"{city.name}<br/>温度: {temp:.1f}°C<br/>风速: {wind:.1f} m/s"
            ),
        )

    geo.set_global_opts(
        title_opts=opts.TitleOpts(
            title="五城市实时天气地理分布",
            subtitle="点大小表示风速，颜色深浅表示当前温度",
        ),
        legend_opts=opts.LegendOpts(is_show=False),
        visualmap_opts=opts.VisualMapOpts(
            min_=min(values["temperature_2m"] for values in realtime_data.values()) - 1,
            max_=max(values["temperature_2m"] for values in realtime_data.values()) + 1,
            dimension=2,
            range_text=["高温", "低温"],
            range_color=["#74c0fc", "#a5d8ff", "#ffe066", "#ffa94d", "#f77f00"],
            pos_left="3%",
            pos_bottom="8%",
        ),
    )
    return ChartBundle(geo, height="640px")


def build_timeline_chart(forecast_frames: dict[str, pd.DataFrame]) -> ChartBundle:
    timeline = Timeline(
        init_opts=opts.InitOpts(
            theme=ThemeType.ROMA,
            chart_id="city_timeline_chart",
            width="1200px",
            height="620px",
        )
    )

    for city in CITIES:
        frame = forecast_frames[city.name]
        labels = frame["time"].dt.strftime("%m-%d %H:%M").tolist()

        line = (
            Line()
            .add_xaxis(labels)
            .add_yaxis(
                f"{city.name}未来24小时温度",
                frame["temperature_2m"].round(2).tolist(),
                is_smooth=True,
                symbol="circle",
                symbol_size=7,
                linestyle_opts=opts.LineStyleOpts(width=3),
                areastyle_opts=opts.AreaStyleOpts(opacity=0.15),
                label_opts=opts.LabelOpts(is_show=False),
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(
                    title=f"{city.name}未来24小时逐小时温度预测",
                    pos_top="3%",
                    pos_left="center",   # 标题居中
                ),
                legend_opts=opts.LegendOpts(
                    pos_top="7%",        # 放到标题下面
                    pos_left="center",   # 图例也居中
                ),
                tooltip_opts=opts.TooltipOpts(trigger="axis"),
                xaxis_opts=opts.AxisOpts(
                    name="时间",
                    axislabel_opts=opts.LabelOpts(rotate=35, interval=2),
                ),
                yaxis_opts=opts.AxisOpts(name="温度 (°C)"),
            )
        )
        timeline.add(line, city.name)

    timeline.add_schema(
        orient="vertical",
        is_auto_play=False,
        play_interval=2400,
        pos_left="3%",          # 左侧 timeline
        pos_top="18%",          # 避开标题和图例
        pos_bottom="12%",
        width="4%",             # 左侧栏不要太宽
        label_opts=opts.LabelOpts(color="#334155"),
    )

    return ChartBundle(timeline, height="620px")


def prepare_chart(bundle: ChartBundle) -> None:
    bundle.chart._prepare_render()
    RenderEngine.generate_js_link(bundle.chart)


def script_tags_from_bundles(bundles: Iterable[ChartBundle]) -> str:
    urls: list[str] = []
    for bundle in bundles:
        urls.extend(bundle.chart.dependencies)
    return "\n  ".join(
        f'<script type="text/javascript" src="{url}"></script>'
        for url in unique_keep_order(urls)
    )


def chart_card_html(bundle: ChartBundle) -> str:
    return (
        f'<section class="card {bundle.column_class}">'
        f'<div id="{bundle.chart.chart_id}" class="chart-box" style="height:{bundle.height};"></div>'
        f"</section>"
    )


def chart_init_script(bundle: ChartBundle) -> str:
    chart = bundle.chart
    return f"""
    <script>
      var chart_{chart.chart_id} = echarts.init(
        document.getElementById('{chart.chart_id}'),
        {json.dumps(chart.theme)},
        {{ renderer: {json.dumps(chart.renderer)}, locale: {json.dumps(chart.locale)} }}
      );
      var option_{chart.chart_id} = {chart.json_contents};
      chart_{chart.chart_id}.setOption(option_{chart.chart_id});
      window.addEventListener('resize', function () {{
        chart_{chart.chart_id}.resize();
      }});
    </script>
    """


def render_html_page(
    *,
    filename: str,
    title: str,
    subtitle: str,
    bundles: list[ChartBundle],
    extra_scripts: str = "",
) -> Path:
    for bundle in bundles:
        prepare_chart(bundle)

    html = PAGE_TEMPLATE.format(
        title=escape(title),
        subtitle=escape(subtitle),
        head_scripts=script_tags_from_bundles(bundles),
        body="\n      ".join(chart_card_html(bundle) for bundle in bundles),
        chart_scripts="\n  ".join(chart_init_script(bundle) for bundle in bundles),
        extra_scripts=extra_scripts,
    )
    output_path = OUTPUT_DIR / filename
    output_path.write_text(html, encoding="utf-8")
    return output_path


def main() -> None:
    ensure_output_dir()

    week_df = fetch_recent_week_hourly("南昌")
    august_frames = {city.name: fetch_august_city_data(city.name) for city in CITIES}

    realtime_data: dict[str, dict[str, float]] = {}
    forecast_frames: dict[str, pd.DataFrame] = {}
    for city in CITIES:
        current_data, forecast_df = fetch_realtime_and_forecast(city.name)
        realtime_data[city.name] = current_data
        forecast_frames[city.name] = forecast_df

    combo_bundle = build_hourly_combo_chart(week_df)
    radar_bundle, heatmap_bundle, linkage_script = build_august_analysis_charts(august_frames)
    geo_bundle = build_geo_chart(realtime_data)
    timeline_bundle = build_timeline_chart(forecast_frames)

    output_paths = [
        render_html_page(
            filename="01_hourly_combo.html",
            title="实验六：过去7天逐小时温度与降水双轴图",
            subtitle="南昌过去7天逐小时 temperature_2m 与 precipitation 数据组合展示。折线表示温度趋势，柱状表示降水变化，时间轴精确到小时。",
            bundles=[combo_bundle],
        ),
        render_html_page(
            filename="02_radar_heatmap_linkage.html",
            title="实验六：多维对比与关联分析",
            subtitle="雷达图对比五城市 2025 年 8 月综合气象特征的 Z-score；点击某个城市后，右侧热力图切换为该城市的指标相关性。",
            bundles=[radar_bundle, heatmap_bundle],
            extra_scripts=linkage_script,
        ),
        render_html_page(
            filename="03_geo_scatter.html",
            title="实验六：五城市实时天气地理散点图",
            subtitle="颜色映射当前温度，散点大小映射实时风速；标签直接显示城市名称与实时温度，并通过不同标签方位减少重叠。",
            bundles=[geo_bundle],
        ),
        render_html_page(
            filename="04_timeline_forecast.html",
            title="实验六：未来24小时温度预测时间轴图",
            subtitle="通过 Timeline 在南昌、长沙、武汉、南京、上海之间切换，查看各城市未来24小时逐小时温度预测。",
            bundles=[timeline_bundle],
        ),
    ]

    print("实验六图表已生成：")
    for path in output_paths:
        print(path)


if __name__ == "__main__":
    main()
