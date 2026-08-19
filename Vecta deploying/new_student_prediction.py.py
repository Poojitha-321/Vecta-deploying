
import pandas as pd
from prophet import Prophet


def create_student_model(attendance_data):

    records = []

    for month, value in attendance_data.items():

        if value is None:
            continue

        if str(value).strip() == "":
            continue

        try:
            value = float(value)
        except (ValueError, TypeError):
            continue

        records.append({
            "ds": pd.to_datetime(month),
            "y": value
        })

    if len(records) < 2:
        raise ValueError(
            "At least two months of attendance are required."
        )

    history = pd.DataFrame(records)

    history = history.sort_values("ds")
    history = history.drop_duplicates(
        subset=["ds"]
    )

    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False
    )

    model.fit(history)

    return model, history


def predict_month(model, history, target_date):

    target_date = pd.to_datetime(target_date)

    last_date = history["ds"].max()

    if target_date <= last_date:
        raise ValueError(
            "Prediction month must be after the latest attendance month."
        )

    future_dates = pd.date_range(
        start=last_date,
        end=target_date,
        freq="MS"
    )

    future = pd.DataFrame({
        "ds": future_dates
    })

    forecast = model.predict(future)

    result = forecast[
        forecast["ds"] == target_date
    ]

    if result.empty:
        raise ValueError(
            "Unable to generate prediction for the selected month."
        )

    value = float(
        result.iloc[0]["yhat"]
    )

    value = max(
        0,
        min(100, value)
    )

    return round(value, 2)


def get_attendance_level(value):

    if value < 75:
        return "Low"

    if value <= 90:
        return "Medium"

    return "High"


def get_status(value):

    if value >= 75:
        return "Safe"

    if value >= 65:
        return "At Risk"

    return "High Risk"

