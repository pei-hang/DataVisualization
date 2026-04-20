from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib import error, parse, request

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder
from plotly.io import to_html


TIMEZONE = "Asia/Shanghai"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs" / "experiment7"
PLOTLY_JS = "https://cdn.plot.ly/plotly-3.5.0.min.js"


@dataclass(frozen=True)
class City:
    name: str
    lat: float
    lon: float


NANCHANG = City("南昌", 28.6829, 115.8582)

THREE_CITIES = [
    NANCHANG,
    City("上海", 31.2304, 121.4737),
    City("广州", 23.1291, 113.2644),
]

CAPITAL_CITIES = [
    City("北京", 39.9042, 116.4074),
    City("天津", 39.3434, 117.3616),
    City("上海", 31.2304, 121.4737),
    City("重庆", 29.5630, 106.5516),
    City("哈尔滨", 45.8038, 126.5349),
    City("长春", 43.8171, 125.3235),
    City("沈阳", 41.8057, 123.4315),
    City("呼和浩特", 40.8426, 111.7492),
    City("石家庄", 38.0428, 114.5149),
    City("乌鲁木齐", 43.8256, 87.6168),
    City("兰州", 36.0611, 103.8343),
    City("西宁", 36.6171, 101.7782),
    City("西安", 34.3416, 108.9398),
    City("银川", 38.4872, 106.2309),
    City("郑州", 34.7466, 113.6254),
    City("济南", 36.6512, 117.1201),
    City("太原", 37.8706, 112.5489),
    City("合肥", 31.8206, 117.2272),
    City("武汉", 30.5928, 114.3055),
    City("长沙", 28.2282, 112.9388),
    City("南京", 32.0603, 118.7969),
    City("成都", 30.5728, 104.0668),
    City("贵阳", 26.6470, 106.6302),
    City("昆明", 25.0389, 102.7183),
    City("南宁", 22.8170, 108.3669),
    City("拉萨", 29.6520, 91.1721),
    City("杭州", 30.2741, 120.1551),
    City("南昌", 28.6829, 115.8582),
    City("广州", 23.1291, 113.2644),
    City("福州", 26.0745, 119.2965),
    City("海口", 20.0440, 110.1999),
    City("台北", 25.0330, 121.5654),
]

WEATHER_LABELS = {
    "sunny": "晴",
    "cloudy": "多云/阴",
    "fog": "雾",
    "rain": "雨",
    "snow": "雪",
    "thunder": "雷暴",
    "other": "其他",
}

WEATHER_COLORS = {
    "晴": "#f7b801",
    "多云/阴": "#8ecae6",
    "雾": "#adb5bd",
    "雨": "#277da1",
    "雪": "#caf0f8",
    "雷暴": "#7b2cbf",
    "其他": "#90be6d",
}


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_fields(fields: Iterable[str] | str | None) -> str | None:
    if fields is None:
        return None
    if isinstance(fields, str):
        return fields
    return ",".join(fields)


class OpenMeteoClient:
    def __init__(self, city: City) -> None:
        self.city = city

    def forecast(
        self,
        *,
        hourly: Iterable[str] | str,
        forecast_days: int | None = None,
        past_days: int | None = None,
    ) -> dict:
        return request_forecast(
            [self.city],
            hourly=hourly,
            forecast_days=forecast_days,
            past_days=past_days,
        )


def request_json(url: str, params: dict[str, object], *, retries: int = 3, timeout: int = 45) -> dict | list:
    encoded = parse.urlencode({key: value for key, value in params.items() if value is not None})
    req = request.Request(
        f"{url}?{encoded}",
        headers={"User-Agent": "DV-Experiment7/1.0"},
    )
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with request.urlopen(req, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (TimeoutError, error.URLError) as exc:
            last_error = exc
            if attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
    assert last_error is not None
    raise last_error


def request_forecast(
    cities: list[City],
    *,
    hourly: Iterable[str] | str,
    forecast_days: int | None = None,
    past_days: int | None = None,
) -> dict | list:
    query = {
        "latitude": ",".join(str(city.lat) for city in cities),
        "longitude": ",".join(str(city.lon) for city in cities),
        "timezone": TIMEZONE,
        "hourly": normalize_fields(hourly),
        "forecast_days": forecast_days,
        "past_days": past_days,
    }
    return request_json(FORECAST_URL, query)


def hourly_frame(payload: dict, city: City) -> pd.DataFrame:
    frame = pd.DataFrame(payload["hourly"])
    frame["time"] = pd.to_datetime(frame["time"])
    frame["city"] = city.name
    frame["lat"] = city.lat
    frame["lon"] = city.lon
    return frame


def current_hour() -> pd.Timestamp:
    return pd.Timestamp.now(tz=TIMEZONE).tz_localize(None).floor("h")


def next_24_hours(frame: pd.DataFrame) -> pd.DataFrame:
    now_hour = current_hour()
    future = frame[frame["time"] >= now_hour].head(24).copy()
    if len(future) < 24:
        future = frame.head(24).copy()
    future["time_label"] = future["time"].dt.strftime("%H:%M")
    return future.reset_index(drop=True)


def recent_24_hours(frame: pd.DataFrame) -> pd.DataFrame:
    now_hour = current_hour()
    recent = frame[frame["time"] <= now_hour].tail(24).copy()
    if len(recent) < 24:
        recent = frame.tail(24).copy()
    recent["time_label"] = recent["time"].dt.strftime("%H:%M")
    return recent.reset_index(drop=True)


def weather_category(code: int | float) -> str:
    if pd.isna(code):
        return WEATHER_LABELS["other"]
    code = int(code)
    if code == 0:
        return WEATHER_LABELS["sunny"]
    if code in {1, 2, 3}:
        return WEATHER_LABELS["cloudy"]
    if code in {45, 48}:
        return WEATHER_LABELS["fog"]
    if code in {51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82}:
        return WEATHER_LABELS["rain"]
    if code in {71, 73, 75, 77, 85, 86}:
        return WEATHER_LABELS["snow"]
    if code in {95, 96, 99}:
        return WEATHER_LABELS["thunder"]
    return WEATHER_LABELS["other"]


def add_temperature_size(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    min_temp = float(data["temperature_2m"].min())
    data["bubble_size"] = (data["temperature_2m"] - min_temp + 8).clip(lower=5)
    return data


def save_plotly_html(fig: go.Figure, filename: str, title: str) -> Path:
    fig.update_layout(
        font={"family": "PingFang SC, Microsoft YaHei, sans-serif"},
        margin={"l": 50, "r": 40, "t": 70, "b": 50},
        title={"text": title, "x": 0.5},
    )
    path = OUTPUT_DIR / filename
    fig.write_html(path, include_plotlyjs="cdn", full_html=True)
    return path


def fetch_nanchang_future() -> pd.DataFrame:
    payload = OpenMeteoClient(NANCHANG).forecast(
        hourly=["temperature_2m", "relative_humidity_2m"],
        forecast_days=2,
    )
    return next_24_hours(hourly_frame(payload, NANCHANG))


def fetch_three_city_recent() -> pd.DataFrame:
    payloads = request_forecast(
        THREE_CITIES,
        hourly=[
            "temperature_2m",
            "precipitation_probability",
            "wind_speed_10m",
            "wind_direction_10m",
        ],
        forecast_days=1,
        past_days=1,
    )
    if isinstance(payloads, dict):
        payloads = [payloads]
    frames = [
        recent_24_hours(hourly_frame(payload, city))
        for payload, city in zip(payloads, THREE_CITIES, strict=False)
    ]
    return pd.concat(frames, ignore_index=True)


def fetch_capital_future() -> pd.DataFrame:
    payloads = request_forecast(
        CAPITAL_CITIES,
        hourly=["temperature_2m", "weather_code"],
        forecast_days=2,
    )
    if isinstance(payloads, dict):
        payloads = [payloads]
    frames = [
        next_24_hours(hourly_frame(payload, city))
        for payload, city in zip(payloads, CAPITAL_CITIES, strict=False)
    ]
    data = pd.concat(frames, ignore_index=True)
    data["weathercode"] = data["weather_code"]
    data["weather"] = data["weathercode"].apply(weather_category)
    data["time_label"] = data["time"].dt.strftime("%m-%d %H:%M")
    return add_temperature_size(data)


def plot_temperature_line(frame: pd.DataFrame) -> Path:
    fig = px.line(
        frame,
        x="time_label",
        y="temperature_2m",
        markers=True,
        labels={"time_label": "时间", "temperature_2m": "温度（℃）"},
        title="24小时温度变化（南昌）",
    )
    fig.update_layout(plot_bgcolor="white")
    fig.update_xaxes(showgrid=True, title="时间", gridcolor="#e0e0e0", linecolor="#cccccc")
    fig.update_yaxes(showgrid=True, title="温度（℃）", gridcolor="#e0e0e0", linecolor="#cccccc")
    fig.update_traces(line={"width": 3, "color": "#e76f51"}, marker={"size": 7})
    return save_plotly_html(fig, "01_temperature_line.html", "24小时温度变化（南昌）")


def plot_temperature_humidity_scatter(frame: pd.DataFrame) -> Path:
    fig = px.scatter(
        frame,
        x="temperature_2m",
        y="relative_humidity_2m",
        color="temperature_2m",
        color_continuous_scale="Turbo",
        trendline="ols",
        labels={
            "temperature_2m": "温度（℃）",
            "relative_humidity_2m": "相对湿度（%）",
        },
        title="南昌未来24小时温度与湿度关系",
    )
    fig.update_layout(plot_bgcolor="white")
    fig.update_xaxes(showgrid=True,gridcolor="#e0e0e0", linecolor="#cccccc")
    fig.update_yaxes(showgrid=True,gridcolor="#e0e0e0", linecolor="#cccccc")
    fig.update_traces(marker=dict(size=12))  # 这里改大小
    return save_plotly_html(fig, "02_temperature_humidity_scatter.html", "南昌未来24小时温湿度散点图")


def build_city_temperature_bar(frame: pd.DataFrame) -> go.Figure:
    summary = (
        frame.groupby("city", as_index=False)
        .agg(
            max_temp=("temperature_2m", "max"),
            min_temp=("temperature_2m", "min"),
            avg_temp=("temperature_2m", "mean"),
        )
        .assign(temp_range=lambda df: df["max_temp"] - df["min_temp"])
    )
    fig = go.Figure(
        go.Bar(
            x=summary["city"],
            y=summary["max_temp"].round(2),
            width=0.4,  # 👈 控制柱子粗细，越小越细
            error_y={"type": "data", "array": summary["temp_range"].round(2), "visible": True},
            marker_color=["#e76f51", "#457b9d", "#2a9d8f"],
            text=summary["max_temp"].round(1).astype(str) + "℃",
            textposition="outside",
            hovertemplate=(
                "城市=%{x}<br>最高温=%{y:.1f}℃"
                "<br>昼夜温差=%{error_y.array:.1f}℃<extra></extra>"
            ),
        )
    )
    fig.update_layout(title="三城市最近24小时最高温度对比（误差线=昼夜温差）", height=700)
    fig.update_xaxes(title="城市")
    fig.update_yaxes(title="最高温度（℃）", showgrid=True)
    return fig


def wind_sector(angle: float) -> str:
    labels = ["北", "东北", "东", "东南", "南", "西南", "西", "西北"]
    index = int(((angle % 360) + 22.5) // 45) % 8
    return labels[index]


def build_wind_rose(frame: pd.DataFrame) -> go.Figure:
    data = frame.dropna(subset=["wind_direction_10m", "wind_speed_10m"]).copy()
    data["风向"] = data["wind_direction_10m"].apply(wind_sector)
    sector_order = ["北", "东北", "东", "东南", "南", "西南", "西", "西北"]
    rose = (
        data.groupby(["city", "风向"], as_index=False)
        .agg(avg_speed=("wind_speed_10m", "mean"), count=("wind_speed_10m", "size"))
    )
    fig = go.Figure()
    for index, city in enumerate([city.name for city in THREE_CITIES]):
        city_data = rose[rose["city"] == city].set_index("风向").reindex(sector_order).fillna(0)
        fig.add_trace(
            go.Barpolar(
                r=city_data["avg_speed"].round(2),
                theta=sector_order,
                name=city,
                visible=index == 0,
                marker_color=["#457b9d", "#2a9d8f", "#e76f51"][index],
                customdata=city_data["count"].astype(int),
                hovertemplate="风向=%{theta}<br>平均风速=%{r:.1f} m/s<br>小时数=%{customdata}<extra></extra>",
            )
        )
    fig.update_layout(
        title="三城市最近24小时风速玫瑰图",
        polar={"radialaxis": {"title": "平均风速 (m/s)", "showgrid": True}},
        updatemenus=[
            {
                "buttons": [
                    {
                        "label": city.name,
                        "method": "update",
                        "args": [
                            {"visible": [i == index for i in range(len(THREE_CITIES))]},
                            {"title": f"{city.name}最近24小时风速玫瑰图"},
                        ],
                    }
                    for index, city in enumerate(THREE_CITIES)
                ],
                "direction": "down",
                "x": 0.02,
                "y": 1.15,
            }
        ],
    )
    return fig


def build_city_dropdown_chart(frame: pd.DataFrame) -> go.Figure:
    metrics = [
        ("temperature_2m", "温度（℃）", "#e76f51"),
        ("precipitation_probability", "降水概率（%）", "#4ea8de"),
        ("wind_speed_10m", "风速（m/s）", "#2a9d8f"),
    ]
    fig = go.Figure()
    cities = [city.name for city in THREE_CITIES]
    for city_index, city in enumerate(cities):
        city_data = frame[frame["city"] == city]
        for metric, label, color in metrics:
            fig.add_trace(
                go.Scatter(
                    x=city_data["time_label"],
                    y=city_data[metric],
                    mode="lines+markers",
                    name=label,
                    legendgroup=city,
                    visible=city_index == 0,
                    line={"color": color, "width": 2.6},
                    hovertemplate=f"{city}<br>时间=%{{x}}<br>{label}=%{{y:.1f}}<extra></extra>",
                )
            )
    trace_count = len(metrics)
    buttons = []
    for city_index, city in enumerate(cities):
        visible = [False] * (len(cities) * trace_count)
        for offset in range(trace_count):
            visible[city_index * trace_count + offset] = True
        buttons.append(
            {
                "label": city,
                "method": "update",
                "args": [
                    {"visible": visible},
                    {"title": f"{city}最近24小时气象指标"},
                ],
            }
        )
    fig.update_layout(
        title=f"{cities[0]}最近24小时气象指标",
        updatemenus=[{"buttons": buttons, "direction": "down", "x": 0.02, "y": 1.18}],
    )
    fig.update_xaxes(title="时间", showgrid=True)
    fig.update_yaxes(title="指标值", showgrid=True)
    return fig


def write_dashboard(frame: pd.DataFrame) -> Path:
    bar_html = to_html(build_city_temperature_bar(frame), full_html=False, include_plotlyjs=False)
    rose_html = to_html(build_wind_rose(frame), full_html=False, include_plotlyjs=False)
    dropdown_html = to_html(build_city_dropdown_chart(frame), full_html=False, include_plotlyjs=False)
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <title>三城市交互气象仪表盘</title>
  <script src="{PLOTLY_JS}"></script>
  <style>
    body {{ margin: 0; font-family: "PingFang SC", "Microsoft YaHei", sans-serif; background: #f5f8ff; color: #1f2937; }}
    main {{ max-width: 1380px; margin: 0 auto; padding: 24px 18px 40px; }}
    h1 {{ margin: 0 0 16px; }}
    .tabs {{ display: flex; gap: 10px; margin-bottom: 16px; flex-wrap: wrap; }}
    .tabs button {{ border: 0; border-radius: 999px; padding: 10px 18px; cursor: pointer; background: #dbeafe; color: #1e3a8a; font-weight: 700; }}
    .tabs button.active {{ background: #1d4ed8; color: white; }}
    .panel {{ display: none; background: white; border-radius: 18px; box-shadow: 0 18px 50px rgba(148, 163, 184, 0.18); padding: 10px; }}
    .panel.active {{ display: block; }}
  </style>
</head>
<body>
  <main>
    <h1>南昌 / 上海 / 广州最近24小时交互气象仪表盘</h1>
    <div class="tabs">
      <button class="active" onclick="showTab('bar', this)">最高温对比</button>
      <button onclick="showTab('rose', this)">风速玫瑰图</button>
      <button onclick="showTab('dropdown', this)">城市指标切换</button>
    </div>
    <section id="bar" class="panel active">{bar_html}</section>
    <section id="rose" class="panel">{rose_html}</section>
    <section id="dropdown" class="panel">{dropdown_html}</section>
  </main>
  <script>
    function showTab(id, button) {{
      document.querySelectorAll('.panel').forEach(function(panel) {{ panel.classList.remove('active'); }});
      document.querySelectorAll('.tabs button').forEach(function(tab) {{ tab.classList.remove('active'); }});
      document.getElementById(id).classList.add('active');
      button.classList.add('active');
      setTimeout(function() {{
        document.querySelectorAll('#' + id + ' .js-plotly-plot').forEach(function(plot) {{ Plotly.Plots.resize(plot); }});
      }}, 50);
    }}
  </script>
</body>
</html>
"""
    path = OUTPUT_DIR / "03_three_city_dashboard.html"
    path.write_text(html, encoding="utf-8")
    return path


def plot_current_national_map(frame: pd.DataFrame) -> Path:
    current = frame.sort_values("time").groupby("city", as_index=False).first()
    fig = px.scatter_geo(
        current,
        lon="lon",
        lat="lat",
        size="bubble_size",
        color="weather",
        color_discrete_map=WEATHER_COLORS,
        hover_name="city",
        hover_data={"temperature_2m": ":.1f", "weathercode": True, "bubble_size": False, "lat": False, "lon": False},
        labels={"weather": "天气现象", "temperature_2m": "温度（℃）", "weathercode": "天气代码"},
        title="全国32个主要城市当前天气散点地图",
    )
    fig.update_geos(
        projection_type="mercator",
        lataxis_range=[15, 55],
        lonaxis_range=[72, 136],
        showcountries=True,
        countrycolor="#94a3b8",
        showland=True,
        landcolor="#eef6ff",
    )
    return save_plotly_html(fig, "04_national_current_map.html", "全国32个主要城市当前天气散点地图")


def plot_temperature_animation(frame: pd.DataFrame) -> Path:
    fig = px.scatter_geo(
        frame,
        lon="lon",
        lat="lat",
        size="bubble_size",
        color="temperature_2m",
        color_continuous_scale="RdYlBu_r",
        animation_frame="time_label",
        hover_name="city",
        hover_data={"temperature_2m": ":.1f", "weather": True, "bubble_size": False, "lat": False, "lon": False},
        labels={"temperature_2m": "温度（℃）", "weather": "天气现象"},
        title="全国未来24小时温度场变化动画",
        range_color=[frame["temperature_2m"].min(), frame["temperature_2m"].max()],
    )
    fig.update_geos(
        projection_type="mercator",
        lataxis_range=[15, 55],
        lonaxis_range=[72, 136],
        showcountries=True,
        countrycolor="#94a3b8",
        showland=True,
        landcolor="#eef6ff",
    )
    return save_plotly_html(fig, "05_national_temperature_animation.html", "全国未来24小时温度场变化动画")

def write_linked_view(frame: pd.DataFrame) -> Path:
    current = frame.sort_values("time").groupby("city", as_index=False).first()
    city_order = current["city"].tolist()
    first_city = city_order[0]
    curves = {
        city: {
            "time": city_frame["time_label"].tolist(),
            "temp": city_frame["temperature_2m"].round(2).tolist(),
        }
        for city, city_frame in frame.groupby("city", sort=False)
    }

    map_fig = go.Figure(
        go.Scattergeo(
            lon=current["lon"],
            lat=current["lat"],
            mode="markers+text",
            text=current["city"],
            textposition="top center",
            customdata=current["city"],
            marker={
                "size": current["bubble_size"],
                "color": current["temperature_2m"],
                "colorscale": "RdYlBu_r",
                "colorbar": {"title": "温度（℃）"},
                "line": {"width": 1, "color": "white"},
            },
            hovertemplate="%{customdata}<br>温度=%{marker.color:.1f}℃<extra></extra>",
        )
    )
    map_fig.update_layout(
        title="点击城市查看对应温度曲线",
        # 👇 缩边距，让地图撑满
        margin=dict(l=10, r=10, t=50, b=10)
    )
    map_fig.update_geos(
        projection_type="mercator",
        lataxis_range=[15, 55],
        lonaxis_range=[72, 136],
        showcountries=True,
        countrycolor="#94a3b8",
        showland=True,
        landcolor="#eef6ff",
    )

    first_curve = curves[first_city]
    line_fig = go.Figure(
        go.Scatter(
            x=first_curve["time"],
            y=first_curve["temp"],
            mode="lines+markers",
            line={"width": 3, "color": "#e76f51"},
            name=first_city,
        )
    )
    line_fig.update_layout(
        title=f"{first_city}未来24小时温度曲线",
        # 👇 缩边距，让曲线撑满
        margin=dict(l=40, r=20, t=60, b=50)
    )
    line_fig.update_xaxes(title="时间", showgrid=True)
    line_fig.update_yaxes(title="温度（℃）", showgrid=True)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <title>全国天气地图-温度曲线关联视图</title>
  <script src="{PLOTLY_JS}"></script>
  <style>
    body {{ margin: 0; font-family: "PingFang SC", "Microsoft YaHei", sans-serif; background: #f5f8ff; color: #1f2937; }}
    main {{ max-width: 1480px; margin: 0 auto; padding: 22px 18px 34px; }}
    h1 {{ margin: 0 0 14px; }}
    .layout {{ display: grid; grid-template-columns: 1fr; gap: 20px; }}
    /* 👇 减小卡片内边距，让图更大 */
    .card {{ background: white; border-radius: 18px; box-shadow: 0 18px 50px rgba(148, 163, 184, 0.18); padding: 4px; }}
    /* 👇 给地图固定高度，更大气 */
    #linked-map {{ height: 420px; }}
    #linked-line {{ height: 380px; }}
  </style>
</head>
<body>
  <main>
    <h1>全国天气地图与城市温度曲线关联视图</h1>
    <section class="layout">
      <div class="card"><div id="linked-map"></div></div>
      <div class="card"><div id="linked-line"></div></div>
    </section>
  </main>
  <script>
    const mapSpec = {json.dumps(map_fig.to_plotly_json(), ensure_ascii=False, cls=PlotlyJSONEncoder)};
    const lineSpec = {json.dumps(line_fig.to_plotly_json(), ensure_ascii=False, cls=PlotlyJSONEncoder)};
    const curves = {json.dumps(curves, ensure_ascii=False)};
    Plotly.newPlot('linked-map', mapSpec.data, mapSpec.layout, {{responsive: true}});
    Plotly.newPlot('linked-line', lineSpec.data, lineSpec.layout, {{responsive: true}});
    document.getElementById('linked-map').on('plotly_click', function(event) {{
      const city = event.points[0].customdata;
      const curve = curves[city];
      if (!curve) return;
      const data = [{{
        x: curve.time,
        y: curve.temp,
        mode: 'lines+markers',
        name: city,
        line: {{width: 3, color: '#e76f51'}},
        hovertemplate: city + '<br>时间=%{{x}}<br>温度=%{{y:.1f}}℃<extra></extra>'
      }}];
      const layout = {{
        title: city + '未来24小时温度曲线',
        xaxis: {{title: '时间', showgrid: true}},
        yaxis: {{title: '温度（℃）', showgrid: true}},
        margin: {{l: 40, r: 20, t: 60, b: 50}},
        font: {{family: 'PingFang SC, Microsoft YaHei, sans-serif'}}
      }};
      Plotly.react('linked-line', data, layout, {{responsive: true}});
    }});
  </script>
</body>
</html>
"""
    path = OUTPUT_DIR / "06_linked_map_temperature_curve.html"
    path.write_text(html, encoding="utf-8")
    return path


def main() -> None:
    ensure_output_dir()

    nanchang_future = fetch_nanchang_future()
    three_city_recent = fetch_three_city_recent()
    capital_future = fetch_capital_future()

    outputs = [
        plot_temperature_line(nanchang_future),
        plot_temperature_humidity_scatter(nanchang_future),
        write_dashboard(three_city_recent),
        plot_current_national_map(capital_future),
        plot_temperature_animation(capital_future),
        write_linked_view(capital_future),
    ]

    print("实验七图表已生成：")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
