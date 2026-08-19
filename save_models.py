import logging
logging.disable(logging.CRITICAL)

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import joblib
from prophet import Prophet


FILE = "student_attendance (2).xlsx"

months = [
    "Jan", "Feb", "Mar", "Apr",
    "May", "Jun", "Jul", "Aug",
    "Sep", "Oct", "Nov"
]


df = pd.read_excel(FILE)

models = {}


for _, student in df.iterrows():

    student_id = str(student["Student_ID"])
    student_name = str(student["Student_Name"])

    history = pd.DataFrame({
        "ds": pd.date_range(
            start="2026-01-01",
            periods=len(months),
            freq="MS"
        ),
        "y": [
            float(student[month])
            for month in months
        ]
    })

    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False
    )

    model.fit(history)

    models[student_id] = {
        "student_name": student_name,
        "model": model,
        "history": history
    }


joblib.dump(
    models,
    "attendance_forecasting_models.pkl"
)

print()
print("Model saved successfully.")
print("Students stored:", len(models))
print("File: attendance_forecasting_models.pkl")
