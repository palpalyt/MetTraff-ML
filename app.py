"""
MetTraff — Traffic Volume Prediction (Gradio app for Hugging Face Spaces)
Loads the tuned CatBoost model and exposes a simple UI.
"""

import gradio as gr
import joblib
import numpy as np
import pandas as pd

# ---- Load artifacts saved from the notebook ----
model         = joblib.load("traffic_model.pkl")
feature_order = joblib.load("feature_order.pkl")

# Bin edges for Low / Medium / High (use the same bin edges from training)
# These will be filled in from the notebook printout
BIN_EDGES = [0, 1193.33, 4933.33, 8000]  # example; replace with values printed by your notebook

WEATHER_OPTIONS = [
    "Clear", "Clouds", "Drizzle", "Fog", "Haze",
    "Mist", "Rain", "Smoke", "Snow", "Squall", "Thunderstorm"
]


def predict(hour, dayofweek, month, temp_c, rain_1h, snow_1h,
            clouds_all, weather_main):
    # Convert Celsius -> Kelvin to match training units
    temp_k = temp_c + 273.15

    is_weekend   = 1 if dayofweek >= 5 else 0
    is_rush_hour = 1 if (7 <= hour <= 9 or 16 <= hour <= 18) else 0

    row = {col: 0 for col in feature_order}
    row.update({
        "temp": temp_k, "rain_1h": rain_1h, "snow_1h": snow_1h,
        "clouds_all": clouds_all, "hour": hour, "dayofweek": dayofweek,
        "month": month, "is_weekend": is_weekend, "is_rush_hour": is_rush_hour,
    })
    one_hot = f"weather_main_{weather_main}"
    if one_hot in row:
        row[one_hot] = 1

    X = pd.DataFrame([row])[feature_order]
    pred = float(model.predict(X)[0])

    # Map to category
    if pred < BIN_EDGES[1]:
        category = "🟢 Low traffic"
    elif pred < BIN_EDGES[2]:
        category = "🟡 Medium traffic"
    else:
        category = "🔴 High traffic"

    return f"{pred:,.0f} vehicles/hour", category


with gr.Blocks(title="MetTraff — Traffic Prediction") as demo:
    gr.Markdown("# 🚗 MetTraff — Traffic Volume Prediction")
    gr.Markdown(
        "Targeted Feature Expansion of Meteorological Data for Dynamic Traffic Flow Prediction. "
        "Enter time and weather information to predict hourly traffic volume."
    )

    with gr.Row():
        with gr.Column():
            gr.Markdown("### ⏰ Time")
            hour      = gr.Slider(0, 23, value=8, step=1, label="Hour of day")
            dayofweek = gr.Slider(0, 6,  value=1, step=1, label="Day of week (0=Mon, 6=Sun)")
            month     = gr.Slider(1, 12, value=10, step=1, label="Month")

            gr.Markdown("### 🌦️ Weather")
            temp_c       = gr.Slider(-30, 40, value=15, step=0.5, label="Temperature (°C)")
            rain_1h      = gr.Slider(0, 50,  value=0, step=0.5, label="Rain in last hour (mm)")
            snow_1h      = gr.Slider(0, 20,  value=0, step=0.5, label="Snow in last hour (mm)")
            clouds_all   = gr.Slider(0, 100, value=40, step=5,  label="Cloud cover (%)")
            weather_main = gr.Dropdown(WEATHER_OPTIONS, value="Clouds", label="Weather category")

            btn = gr.Button("🔮 Predict", variant="primary")

        with gr.Column():
            out_volume   = gr.Textbox(label="Predicted traffic volume")
            out_category = gr.Textbox(label="Traffic category")

    btn.click(predict,
              inputs=[hour, dayofweek, month, temp_c, rain_1h, snow_1h,
                      clouds_all, weather_main],
              outputs=[out_volume, out_category])

    gr.Markdown(
        "**Model:** Tuned CatBoost (R² = 0.96 on holdout) — Final Project, "
        "Machine Learning Course, ITS 2026"
    )

if __name__ == "__main__":
    demo.launch()
