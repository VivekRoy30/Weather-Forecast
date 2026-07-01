from __future__ import annotations

import json
import math
import statistics
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


ROOT = Path(__file__).parent
OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_GEOCODE = "https://geocoding-api.open-meteo.com/v1/search"


def fetch_json(url: str, params: dict[str, Any], timeout: int = 12) -> dict[str, Any]:
    query = urllib.parse.urlencode(params, doseq=True)
    request = urllib.request.Request(
        f"{url}?{query}",
        headers={"User-Agent": "python-ml-weather-demo/1.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def geocode(place: str) -> dict[str, Any]:
    data = fetch_json(
        OPEN_METEO_GEOCODE,
        {
            "name": place,
            "count": 1,
            "language": "en",
            "format": "json",
        },
    )
    results = data.get("results") or []
    if not results:
        raise ValueError(f"No location found for {place!r}")

    match = results[0]
    return {
        "name": match.get("name", place),
        "country": match.get("country", ""),
        "admin1": match.get("admin1", ""),
        "latitude": match["latitude"],
        "longitude": match["longitude"],
        "timezone": match.get("timezone", "auto"),
    }


def get_weather(location: dict[str, Any]) -> dict[str, Any]:
    return fetch_json(
        OPEN_METEO_FORECAST,
        {
            "latitude": location["latitude"],
            "longitude": location["longitude"],
            "timezone": "auto",
            "past_days": 7,
            "forecast_days": 3,
            "current": [
                "temperature_2m",
                "relative_humidity_2m",
                "apparent_temperature",
                "precipitation",
                "weather_code",
                "wind_speed_10m",
            ],
            "hourly": [
                "temperature_2m",
                "relative_humidity_2m",
                "precipitation_probability",
                "pressure_msl",
                "wind_speed_10m",
                "cloud_cover",
            ],
            "daily": [
                "temperature_2m_max",
                "temperature_2m_min",
                "precipitation_probability_max",
                "weather_code",
            ],
        },
    )


def parse_hour(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=None)


def weather_label(code: int | None) -> str:
    labels = {
        0: "Clear",
        1: "Mostly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Fog",
        48: "Rime fog",
        51: "Light drizzle",
        53: "Drizzle",
        55: "Heavy drizzle",
        61: "Light rain",
        63: "Rain",
        65: "Heavy rain",
        71: "Light snow",
        73: "Snow",
        75: "Heavy snow",
        80: "Light showers",
        81: "Showers",
        82: "Heavy showers",
        95: "Thunderstorm",
        96: "Thunderstorm hail",
        99: "Heavy thunderstorm hail",
    }
    return labels.get(code, "Mixed conditions")


def normalize_columns(rows: list[list[float]]) -> tuple[list[list[float]], list[float], list[float]]:
    columns = list(zip(*rows))
    means = [statistics.fmean(column) for column in columns]
    scales = [statistics.pstdev(column) or 1.0 for column in columns]
    normalized = [
        [(value - means[index]) / scales[index] for index, value in enumerate(row)]
        for row in rows
    ]
    return normalized, means, scales


def solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    size = len(vector)
    augmented = [matrix[row][:] + [vector[row]] for row in range(size)]

    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]

        divisor = augmented[column][column]
        if abs(divisor) < 1e-12:
            continue

        for item in range(column, size + 1):
            augmented[column][item] /= divisor

        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            for item in range(column, size + 1):
                augmented[row][item] -= factor * augmented[column][item]

    return [augmented[row][size] for row in range(size)]


def train_ridge_regression(features: list[list[float]], targets: list[float]) -> dict[str, Any]:
    normalized, means, scales = normalize_columns(features)
    design = [[1.0] + row for row in normalized]
    width = len(design[0])
    penalty = 0.35

    xtx = [[0.0 for _ in range(width)] for _ in range(width)]
    xty = [0.0 for _ in range(width)]

    for row, target in zip(design, targets):
        for left in range(width):
            xty[left] += row[left] * target
            for right in range(width):
                xtx[left][right] += row[left] * row[right]

    for index in range(1, width):
        xtx[index][index] += penalty

    return {
        "weights": solve_linear_system(xtx, xty),
        "means": means,
        "scales": scales,
    }


def predict(model: dict[str, Any], feature: list[float]) -> float:
    normalized = [
        (value - model["means"][index]) / model["scales"][index]
        for index, value in enumerate(feature)
    ]
    row = [1.0] + normalized
    return sum(weight * value for weight, value in zip(model["weights"], row))


def feature_row(dt: datetime, hour_index: int, humidity: float, pressure: float, wind: float, cloud: float) -> list[float]:
    hour_angle = 2 * math.pi * dt.hour / 24
    return [
        hour_index,
        math.sin(hour_angle),
        math.cos(hour_angle),
        humidity,
        pressure,
        wind,
        cloud,
    ]


def build_ml_forecast(weather: dict[str, Any]) -> dict[str, Any]:
    hourly = weather["hourly"]
    times = [parse_hour(value) for value in hourly["time"]]
    temperatures = hourly["temperature_2m"]
    humidities = hourly["relative_humidity_2m"]
    pressures = hourly["pressure_msl"]
    winds = hourly["wind_speed_10m"]
    clouds = hourly["cloud_cover"]
    now = parse_hour(weather["current"]["time"])

    training_features: list[list[float]] = []
    training_targets: list[float] = []
    forecast_features: list[tuple[datetime, list[float], float | None, int]] = []

    for index, dt in enumerate(times):
        if None in (temperatures[index], humidities[index], pressures[index], winds[index], clouds[index]):
            continue

        feature = feature_row(
            dt,
            index,
            float(humidities[index]),
            float(pressures[index]),
            float(winds[index]),
            float(clouds[index]),
        )

        if dt <= now and temperatures[index] is not None:
            training_features.append(feature)
            training_targets.append(float(temperatures[index]))
        elif dt > now:
            forecast_features.append((dt, feature, temperatures[index], index))

    if len(training_features) < 24 or not forecast_features:
        raise ValueError("Not enough hourly data to train a forecast model.")

    model = train_ridge_regression(training_features, training_targets)
    residuals = [
        abs(actual - predict(model, feature))
        for feature, actual in zip(training_features, training_targets)
    ]
    mean_error = statistics.fmean(residuals[-48:]) if residuals else 0.0

    points = []
    for dt, feature, api_temperature, index in forecast_features[:24]:
        ml_temperature = predict(model, feature)
        points.append(
            {
                "time": dt.strftime("%H:%M"),
                "date": dt.strftime("%b %d"),
                "temperature": round(ml_temperature, 1),
                "apiTemperature": round(float(api_temperature), 1) if api_temperature is not None else None,
                "humidity": hourly["relative_humidity_2m"][index],
                "rain": hourly["precipitation_probability"][index],
                "wind": hourly["wind_speed_10m"][index],
                "cloud": hourly["cloud_cover"][index],
            }
        )

    return {
        "points": points,
        "meanError": round(mean_error, 2),
        "trainingSamples": len(training_features),
        "model": "Pure Python ridge regression",
    }


def response_payload(place: str) -> dict[str, Any]:
    location = geocode(place)
    weather = get_weather(location)
    current = weather["current"]
    ml = build_ml_forecast(weather)
    current_day = current["time"].split("T", 1)[0]
    daily = []
    for index, date in enumerate(weather["daily"]["time"]):
        if date < current_day:
            continue
        daily.append(
            {
                "date": date,
                "max": weather["daily"]["temperature_2m_max"][index],
                "min": weather["daily"]["temperature_2m_min"][index],
                "rain": weather["daily"]["precipitation_probability_max"][index],
                "label": weather_label(weather["daily"]["weather_code"][index]),
            }
        )
        if len(daily) == 3:
            break

    return {
        "location": location,
        "updatedAt": datetime.now(timezone.utc).isoformat(),
        "current": {
            "temperature": current.get("temperature_2m"),
            "feelsLike": current.get("apparent_temperature"),
            "humidity": current.get("relative_humidity_2m"),
            "precipitation": current.get("precipitation"),
            "wind": current.get("wind_speed_10m"),
            "code": current.get("weather_code"),
            "label": weather_label(current.get("weather_code")),
        },
        "daily": daily,
        "ml": ml,
    }


class WeatherHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/api/weather":
            return super().do_GET()

        params = urllib.parse.parse_qs(parsed.query)
        place = (params.get("city") or ["New York"])[0].strip() or "New York"

        try:
            payload = response_payload(place)
            self.send_json(payload)
        except Exception as error:  # The browser gets a useful message instead of a blank page.
            self.send_json({"error": str(error)}, status=502)

    def send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def main() -> None:
    port = 8000
    server = ThreadingHTTPServer(("127.0.0.1", port), WeatherHandler)
    print(f"Weather ML app running at http://127.0.0.1:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
