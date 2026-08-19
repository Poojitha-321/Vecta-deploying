
from new_student_prediction import (
    create_student_model,
    predict_month,
    get_attendance_level,
    get_status
)


attendance = {

    "2026-01-01": 82,
    "2026-02-01": 84,
    "2026-03-01": 80,
    "2026-04-01": 86,
    "2026-05-01": 88,
    "2026-06-01": 87,
    "2026-07-01": 89,
    "2026-08-01": 91,
    "2026-09-01": 90,
    "2026-10-01": 92,
    "2026-11-01": 93
}


model, history = create_student_model(
    attendance
)


prediction = predict_month(
    model,
    history,
    "2026-12-01"
)


print("Latest Attendance:",
      history.iloc[-1]["y"])

print("Predicted December:",
      prediction)

print("Attendance Level:",
      get_attendance_level(prediction))

print("Status:",
      get_status(prediction))

