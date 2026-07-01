# Real-Time Weather Forecasting Webpage

This project is a dependency-free Python weather app. It uses Open-Meteo for live weather data and trains a small pure-Python ridge regression model on recent hourly observations to generate the next 24 hours of temperature predictions.

## Run

```bash
python3 app.py
```

Open:

```text
http://127.0.0.1:8000
```

## How It Works

- `app.py` serves the webpage and exposes `/api/weather`.
- City names are geocoded with Open-Meteo.
- Recent hourly weather is used as training data.
- The model learns from hour-of-day, humidity, pressure, wind, cloud cover, and time index.
- `index.html`, `style.css`, and `script.js` render the real-time dashboard and forecast chart.
