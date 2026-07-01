const form = document.querySelector("#weather-form");
const input = document.querySelector("#city-input");
const toast = document.querySelector("#toast");
const chart = document.querySelector("#forecast-chart");
const ctx = chart.getContext("2d");
let lastWeatherPoints = [];

const fields = {
  location: document.querySelector("#location-name"),
  condition: document.querySelector("#condition-label"),
  temperature: document.querySelector("#current-temp"),
  feelsLike: document.querySelector("#feels-like"),
  humidity: document.querySelector("#humidity"),
  wind: document.querySelector("#wind"),
  model: document.querySelector("#model-name"),
  samples: document.querySelector("#training-samples"),
  error: document.querySelector("#mean-error"),
  updated: document.querySelector("#updated-at"),
  hourly: document.querySelector("#hourly-list"),
  daily: document.querySelector("#daily-list"),
};

function showToast(message) {
  toast.textContent = message;
  toast.classList.add("show");
  window.setTimeout(() => toast.classList.remove("show"), 3600);
}

function formatLocation(location) {
  return [location.name, location.admin1, location.country].filter(Boolean).join(", ");
}

function setLoading(city) {
  fields.location.textContent = city;
  fields.condition.textContent = "Training model with live weather data...";
  fields.temperature.textContent = "--";
}

function drawChart(points) {
  lastWeatherPoints = points;
  const rect = chart.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  chart.width = Math.max(680, Math.floor(rect.width * ratio));
  chart.height = Math.floor(360 * ratio);
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);

  const width = chart.width / ratio;
  const height = chart.height / ratio;
  const padding = { top: 28, right: 28, bottom: 44, left: 52 };
  const temps = points.map((point) => point.temperature);
  const min = Math.floor(Math.min(...temps) - 2);
  const max = Math.ceil(Math.max(...temps) + 2);
  const usableWidth = width - padding.left - padding.right;
  const usableHeight = height - padding.top - padding.bottom;

  const x = (index) => padding.left + (usableWidth * index) / Math.max(points.length - 1, 1);
  const y = (temp) => padding.top + usableHeight - ((temp - min) / (max - min || 1)) * usableHeight;

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);

  ctx.strokeStyle = "#d8e0e7";
  ctx.lineWidth = 1;
  ctx.fillStyle = "#647181";
  ctx.font = "12px Inter, sans-serif";

  for (let tick = 0; tick <= 4; tick += 1) {
    const value = min + ((max - min) * tick) / 4;
    const lineY = y(value);
    ctx.beginPath();
    ctx.moveTo(padding.left, lineY);
    ctx.lineTo(width - padding.right, lineY);
    ctx.stroke();
    ctx.fillText(`${Math.round(value)}°`, 12, lineY + 4);
  }

  const gradient = ctx.createLinearGradient(0, padding.top, 0, height - padding.bottom);
  gradient.addColorStop(0, "rgba(251, 143, 103, 0.34)");
  gradient.addColorStop(1, "rgba(38, 119, 186, 0.06)");

  ctx.beginPath();
  points.forEach((point, index) => {
    const px = x(index);
    const py = y(point.temperature);
    if (index === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  });
  ctx.lineTo(x(points.length - 1), height - padding.bottom);
  ctx.lineTo(x(0), height - padding.bottom);
  ctx.closePath();
  ctx.fillStyle = gradient;
  ctx.fill();

  ctx.beginPath();
  points.forEach((point, index) => {
    const px = x(index);
    const py = y(point.temperature);
    if (index === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  });
  ctx.strokeStyle = "#2677ba";
  ctx.lineWidth = 3;
  ctx.stroke();

  points.forEach((point, index) => {
    if (index % 4 !== 0) return;
    ctx.fillStyle = "#16202a";
    ctx.beginPath();
    ctx.arc(x(index), y(point.temperature), 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#647181";
    ctx.fillText(point.time, x(index) - 16, height - 18);
  });
}

function render(data) {
  fields.location.textContent = formatLocation(data.location);
  fields.condition.textContent = data.current.label;
  fields.temperature.textContent = Math.round(data.current.temperature);
  fields.feelsLike.textContent = `${Math.round(data.current.feelsLike)}°C`;
  fields.humidity.textContent = `${data.current.humidity}%`;
  fields.wind.textContent = `${Math.round(data.current.wind)} km/h`;
  fields.model.textContent = data.ml.model;
  fields.samples.textContent = data.ml.trainingSamples;
  fields.error.textContent = `±${data.ml.meanError}°C`;
  fields.updated.textContent = `Updated ${new Date(data.updatedAt).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`;

  fields.hourly.innerHTML = data.ml.points.slice(0, 8).map((point) => `
    <div class="hour-card">
      <strong>${point.time}</strong>
      <span>ML temp ${point.temperature}°C</span>
      <span>Rain ${point.rain ?? 0}%</span>
      <span>Wind ${Math.round(point.wind)} km/h</span>
    </div>
  `).join("");

  fields.daily.innerHTML = data.daily.map((day) => `
    <div class="day-card">
      <div>
        <strong>${new Date(`${day.date}T00:00:00`).toLocaleDateString([], { weekday: "short", month: "short", day: "numeric" })}</strong>
        <span>${day.label}</span>
        <span>Rain chance ${day.rain ?? 0}%</span>
      </div>
      <strong>${Math.round(day.max)}° / ${Math.round(day.min)}°</strong>
    </div>
  `).join("");

  drawChart(data.ml.points);
}

async function loadWeather(city) {
  setLoading(city);
  try {
    const response = await fetch(`/api/weather?city=${encodeURIComponent(city)}`);
    const data = await response.json();
    if (!response.ok || data.error) throw new Error(data.error || "Weather request failed");
    render(data);
  } catch (error) {
    fields.condition.textContent = "Could not load live weather";
    showToast(error.message);
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  loadWeather(input.value.trim() || "New York");
});

window.addEventListener("resize", () => {
  if (lastWeatherPoints.length) drawChart(lastWeatherPoints);
});

loadWeather(input.value);
