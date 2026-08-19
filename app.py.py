from flask import Flask, render_template, request, redirect, url_for, session, send_file
import pandas as pd
import joblib
from prophet import Prophet
from werkzeug.utils import secure_filename
import os

import pandas as pd
import joblib

from new_student_prediction import (
    create_student_model,
    predict_month,
    get_attendance_level,
    get_status
)
app = Flask(__name__)

app.secret_key = "attendance-ai-secret-key"


PROFILE_FILE = "data/student_profiles.xlsx"
MODEL_FILE = "attendance_forecasting_models.pkl"

UPLOAD_FOLDER = "data/uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# ============================================================
# FACULTY LOGIN
# ============================================================

FACULTY_USERNAME = "faculty"
FACULTY_PASSWORD = "faculty123"


# ============================================================
# LOAD DATA
# ============================================================

students = pd.read_excel(PROFILE_FILE)

students.columns = students.columns.str.strip()

students["Student_ID"] = (
    students["Student_ID"]
    .astype(str)
    .str.strip()
)


# ============================================================
# LOAD SAVED MODELS
# ============================================================

try:
    models = joblib.load(MODEL_FILE)
except Exception:
    models = {}


# ============================================================
# SUBJECTS
# ============================================================

SUBJECTS = [
    "NLP",
    "AI",
    "IOT",
    "CNS",
    "Python"
]


# ============================================================
# MONTHS
# ============================================================

MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December"
]


MONTH_NUMBERS = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12
}


# ============================================================
# FIND STUDENT
# ============================================================

def find_student(student_id):

    student_id = str(student_id).strip()

    result = students[
        students["Student_ID"] == student_id
    ]

    if result.empty:
        return None

    return result.iloc[0].to_dict()


# ============================================================
# ATTENDANCE LEVEL
# ============================================================

def get_level(value):

    if value < 75:
        return "Low"

    if value <= 90:
        return "Medium"

    return "High"


# ============================================================
# CURRENT ATTENDANCE
# ============================================================

def get_current_attendance(student_id):

    student_model = models.get(student_id)

    if not student_model:
        return None

    history = student_model.get("history")

    if history is None or history.empty:
        return None

    value = history.iloc[-1]["y"]

    return round(float(value), 2)


# ============================================================
# STUDENT HISTORY
# ============================================================

def get_student_history(student_id):

    student_model = models.get(student_id)

    if not student_model:
        return [], []

    history = student_model.get("history")

    if history is None or history.empty:
        return [], []

    history = history.copy()

    history["ds"] = pd.to_datetime(history["ds"])

    labels = history["ds"].dt.strftime("%B").tolist()

    values = history["y"].astype(float).round(2).tolist()

    return labels, values


# ============================================================
# PREDICT USING SAVED MODEL
# ============================================================

def predict_saved_model(student_id, target_month):

    student_model = models.get(student_id)

    if not student_model:
        return None

    model = student_model.get("model")

    history = student_model.get("history")

    if model is None or history is None:
        return None

    history = history.copy()

    history["ds"] = pd.to_datetime(history["ds"])

    last_date = history["ds"].max()

    target_month_number = MONTH_NUMBERS[target_month]

    target_year = last_date.year

    if target_month_number <= last_date.month:
        target_year += 1

    target_date = pd.Timestamp(
        year=target_year,
        month=target_month_number,
        day=1
    )

    months_ahead = (
        (target_date.year - last_date.year) * 12
        + target_date.month
        - last_date.month
    )

    if months_ahead <= 0:
        months_ahead = 1

    future = model.make_future_dataframe(
        periods=months_ahead,
        freq="MS"
    )

    forecast = model.predict(future)

    result = forecast[
        forecast["ds"] == target_date
    ]

    if result.empty:
        return None

    value = result.iloc[0]["yhat"]

    value = max(
        0,
        min(100, float(value))
    )

    return round(value, 2)


# ============================================================
# CREATE MODEL FOR NEW STUDENT
# ============================================================

def predict_new_student(attendance_data, target_month):

    rows = []

    for month, value in attendance_data.items():

        if value in [None, ""]:
            continue

        try:
            value = float(value)
        except:
            continue

        month_number = MONTH_NUMBERS[month]

        rows.append({
            "ds": pd.Timestamp(
                year=2026,
                month=month_number,
                day=1
            ),
            "y": value
        })

    if len(rows) < 2:
        return None

    history = pd.DataFrame(rows)

    history = history.sort_values("ds")

    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False
    )

    model.fit(history)

    last_date = history["ds"].max()

    target_month_number = MONTH_NUMBERS[target_month]

    target_year = last_date.year

    if target_month_number <= last_date.month:
        target_year += 1

    target_date = pd.Timestamp(
        year=target_year,
        month=target_month_number,
        day=1
    )

    months_ahead = (
        (target_date.year - last_date.year) * 12
        + target_date.month
        - last_date.month
    )

    if months_ahead <= 0:
        months_ahead = 1

    future = model.make_future_dataframe(
        periods=months_ahead,
        freq="MS"
    )

    forecast = model.predict(future)

    result = forecast[
        forecast["ds"] == target_date
    ]

    if result.empty:
        return None

    value = float(result.iloc[0]["yhat"])

    value = max(
        0,
        min(100, value)
    )

    return round(value, 2)


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():
    return redirect(url_for("login"))


# ============================================================
# LOGIN
# ============================================================
@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "GET":
        return render_template("login.html")

    login_type = request.form.get("login_type", "student")

    username = request.form.get("username", "").strip()
    password = request.form.get("password", "").strip()

    # STUDENT LOGIN
    if login_type == "student":

        if not username or not password:
            return render_template(
                "login.html",
                error="Please enter Student ID and Password."
            )

        if username != password:
            return render_template(
                "login.html",
                error="For student login, password must be the Student ID."
            )

        student = find_student(username)

        if student is None:
            return render_template(
                "login.html",
                error="Student ID was not found."
            )

        session.clear()
        session["student_id"] = username

        return redirect(url_for("student_home"))

    # FACULTY LOGIN
    elif login_type == "faculty":

        if (
            username == FACULTY_USERNAME
            and password == FACULTY_PASSWORD
        ):
            session.clear()
            session["faculty"] = True

            return redirect(url_for("faculty_home"))

        return render_template(
            "login.html",
            error="Invalid faculty username or password."
        )

    return render_template(
        "login.html",
        error="Please select Student or Faculty login."
    )
# ============================================================
# STUDENT HOME
# ============================================================

@app.route("/student-home")
def student_home():

    if "student_id" not in session:
        return redirect(url_for("login"))

    student_id = session["student_id"]

    student = find_student(student_id)

    if student is None:
        session.clear()
        return redirect(url_for("login"))

    return render_template(
        "student_home.html",
        student=student,
        student_id=student_id
    )


# ============================================================
# STUDENT PROFILE
# ============================================================

@app.route("/profile")
def profile():

    if "student_id" not in session:
        return redirect(url_for("login"))

    student_id = session["student_id"]

    student = find_student(student_id)

    if student is None:
        session.clear()
        return redirect(url_for("login"))

    return render_template(
        "profile.html",
        student=student,
        student_id=student_id
    )


# ============================================================
# STUDENT ATTENDANCE PREDICTION
# ============================================================

@app.route("/prediction", methods=["GET", "POST"])
def prediction():

    if "student_id" not in session:
        return redirect(url_for("login"))

    student_id = str(session["student_id"]).strip()

    student = find_student(student_id)

    if student is None:
        session.clear()
        return redirect(url_for("login"))

    if student_id not in models:
        return render_template(
            "prediction.html",
            student=student,
            student_id=student_id,
            error="Prediction model is not available for this student."
        )

    student_model = models[student_id]

    model = student_model["model"]
    history = student_model["history"]

    history = history.copy()

    history["ds"] = pd.to_datetime(history["ds"])

    history["y"] = pd.to_numeric(
        history["y"],
        errors="coerce"
    )

    history = history.dropna(
        subset=["ds", "y"]
    )

    history = history.sort_values(
        "ds"
    )

    if history.empty:
        return render_template(
            "prediction.html",
            student=student,
            student_id=student_id,
            error="Attendance history is not available."
        )

    # ---------------------------------------------------------
    # LAST ACTUAL ATTENDANCE
    # ---------------------------------------------------------

    last_actual_value = float(
        history.iloc[-1]["y"]
    )

    last_actual_value = max(
        0,
        min(
            100,
            last_actual_value
        )
    )

    last_actual_value = round(
        last_actual_value,
        2
    )

    last_date = pd.to_datetime(
        history.iloc[-1]["ds"]
    )

    last_month = last_date.strftime(
        "%B %Y"
    )

    # ---------------------------------------------------------
    # AVAILABLE FUTURE MONTHS
    # ---------------------------------------------------------

    start_month = (
        last_date
        + pd.offsets.MonthBegin(1)
    ).replace(
        day=1
    )

    future_months = []

    for i in range(1, 25):

        month_date = (
            start_month
            + pd.DateOffset(months=i - 1)
        )

        future_months.append({
            "value": month_date.strftime("%Y-%m"),
            "label": month_date.strftime("%B %Y")
        })

    # ---------------------------------------------------------
    # DEFAULT VALUES
    # ---------------------------------------------------------

    predicted_value = None
    predicted_month = None
    attendance_level = None
    selected_month = None

    # ---------------------------------------------------------
    # PREDICT
    # ---------------------------------------------------------

    if request.method == "POST":

        selected_month = request.form.get(
            "prediction_month",
            ""
        ).strip()

        if not selected_month:

            return render_template(
                "prediction.html",
                student=student,
                student_id=student_id,
                last_actual=last_actual_value,
                last_month=last_month,
                future_months=future_months,
                error="Please select a month."
            )

        try:

            selected_date = pd.to_datetime(
                selected_month + "-01"
            )

        except Exception:

            return render_template(
                "prediction.html",
                student=student,
                student_id=student_id,
                last_actual=last_actual_value,
                last_month=last_month,
                future_months=future_months,
                error="Invalid prediction month."
            )

        # -----------------------------------------------------
        # TARGET MUST BE AFTER LAST ACTUAL MONTH
        # -----------------------------------------------------

        if selected_date <= last_date.replace(day=1):

            return render_template(
                "prediction.html",
                student=student,
                student_id=student_id,
                last_actual=last_actual_value,
                last_month=last_month,
                future_months=future_months,
                selected_month=selected_month,
                error=(
                    "Please select a future month. "
                    "You cannot predict a month already present "
                    "in the attendance history."
                )
            )

        # -----------------------------------------------------
        # NUMBER OF MONTHS TO PREDICT
        # -----------------------------------------------------

        months_ahead = (
            (selected_date.year - last_date.year) * 12
            + (selected_date.month - last_date.month)
        )

        # -----------------------------------------------------
        # CREATE FUTURE DATA
        # -----------------------------------------------------

        future = model.make_future_dataframe(
            periods=months_ahead,
            freq="MS"
        )

        # -----------------------------------------------------
        # GENERATE FORECAST
        # -----------------------------------------------------

        forecast = model.predict(
            future
        )

        target_rows = forecast[
            forecast["ds"].dt.to_period("M")
            == selected_date.to_period("M")
        ]

        if target_rows.empty:

            return render_template(
                "prediction.html",
                student=student,
                student_id=student_id,
                last_actual=last_actual_value,
                last_month=last_month,
                future_months=future_months,
                selected_month=selected_month,
                error="Prediction could not be generated for the selected month."
            )

        predicted_value = float(
            target_rows.iloc[-1]["yhat"]
        )

        # -----------------------------------------------------
        # KEEP ATTENDANCE BETWEEN 0 AND 100
        # -----------------------------------------------------

        predicted_value = max(
            0,
            min(
                100,
                predicted_value
            )
        )

        predicted_value = round(
            predicted_value,
            2
        )

        predicted_month = selected_date.strftime(
            "%B %Y"
        )

        # -----------------------------------------------------
        # ATTENDANCE LEVEL
        # -----------------------------------------------------

        attendance_level = get_level(
            predicted_value
        )

    # ---------------------------------------------------------
    # SEND DATA TO TEMPLATE
    # ---------------------------------------------------------

    return render_template(

        "prediction.html",

        student=student,

        student_id=student_id,

        last_actual=last_actual_value,

        last_month=last_month,

        future_months=future_months,

        selected_month=selected_month,

        predicted_value=predicted_value,

        predicted_month=predicted_month,

        attendance_level=attendance_level
    )
 # ============================================================
# STUDENT DASHBOARD
# ============================================================

@app.route("/student-dashboard")
def student_dashboard():

    if "student_id" not in session:
        return redirect(url_for("login"))

    student_id = str(
        session["student_id"]
    ).strip()

    student = find_student(
        student_id
    )

    if student is None:
        session.clear()
        return redirect(url_for("login"))

    # --------------------------------------------------------
    # ATTENDANCE HISTORY FOR CHART
    # --------------------------------------------------------

    chart_labels = []
    chart_values = []

    if student_id in models:

        student_model = models[
            student_id
        ]

        history = student_model.get(
            "history"
        )

        if (
            history is not None
            and not history.empty
        ):

            history = history.copy()

            history["ds"] = pd.to_datetime(
                history["ds"]
            )

            history["y"] = pd.to_numeric(
                history["y"],
                errors="coerce"
            )

            history = history.dropna(
                subset=["ds", "y"]
            )

            history = history.sort_values(
                "ds"
            )

            for _, row in history.iterrows():

                chart_labels.append(
                    row["ds"].strftime(
                        "%B %Y"
                    )
                )

                chart_values.append(
                    round(
                        float(row["y"]),
                        2
                    )
                )

    # --------------------------------------------------------
    # CURRENT ATTENDANCE
    # --------------------------------------------------------

    current_attendance = None

    if chart_values:

        current_attendance = (
            chart_values[-1]
        )

    # --------------------------------------------------------
    # NEXT MONTH PREDICTION
    # --------------------------------------------------------

    predicted_value = None
    predicted_month = None

    if student_id in models:

        student_model = models[
            student_id
        ]

        model = student_model.get(
            "model"
        )

        history = student_model.get(
            "history"
        )

        if (
            model is not None
            and history is not None
            and not history.empty
        ):

            history = history.copy()

            history["ds"] = pd.to_datetime(
                history["ds"]
            )

            history["y"] = pd.to_numeric(
                history["y"],
                errors="coerce"
            )

            history = history.dropna(
                subset=["ds", "y"]
            )

            history = history.sort_values(
                "ds"
            )

            if not history.empty:

                last_date = pd.to_datetime(
                    history.iloc[-1]["ds"]
                )

                next_month = (
                    last_date
                    + pd.offsets.MonthBegin(1)
                )

                future = (
                    model.make_future_dataframe(
                        periods=1,
                        freq="MS"
                    )
                )

                forecast = model.predict(
                    future
                )

                target_rows = forecast[
                    forecast["ds"]
                    == next_month
                ]

                if not target_rows.empty:

                    predicted_value = float(
                        target_rows.iloc[-1]["yhat"]
                    )

                    predicted_value = max(
                        0,
                        min(
                            100,
                            predicted_value
                        )
                    )

                    predicted_value = round(
                        predicted_value,
                        2
                    )

                    predicted_month = (
                        next_month.strftime(
                            "%B %Y"
                        )
                    )

    # --------------------------------------------------------
    # SEND DATA TO TEMPLATE
    # --------------------------------------------------------

    return render_template(
        "student_dashboard.html",

        student=student,

        student_id=student_id,

        current_attendance=current_attendance,

        predicted_value=predicted_value,

        predicted_month=predicted_month,

        chart_labels=chart_labels,

        chart_values=chart_values
    )   
# ============================================================
# FACULTY HOME
# ============================================================

@app.route("/faculty-home")
def faculty_home():

    if "faculty" not in session:
        return redirect(url_for("login"))

    return render_template(
        "faculty_home.html"
    )


# ============================================================
# FACULTY ATTENDANCE
# ============================================================

@app.route("/faculty-attendance")
def faculty_attendance():

    if "faculty" not in session:
        return redirect(url_for("login"))

    departments = sorted(
        students["Department"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    return render_template(
        "faculty_attendance.html",
        departments=departments
    )


# ============================================================
# DEPARTMENT ATTENDANCE
# ============================================================

@app.route("/department-attendance")
def department_attendance():

    if "faculty" not in session:
        return redirect(url_for("login"))

    department = request.args.get(
        "department",
        ""
    ).strip()

    if not department:
        return redirect(
            url_for("faculty_attendance")
        )

    department_students = students[
        students["Department"]
        .astype(str)
        .str.strip()
        .str.lower()
        == department.lower()
    ].copy()

    result = []

    for _, student in department_students.iterrows():

        student_id = str(
            student["Student_ID"]
        ).strip()

        student_data = {
            "Student_ID": student_id,
            "Roll_No": student.get("Roll_No", ""),
            "Name": student.get("Name", ""),
            "Email": student.get("Email", ""),
            "Gender": student.get("Gender", ""),
            "Year": student.get("Year", ""),
            "Attendance": None,
            "Predicted_Attendance": None,
            "Predicted_Month": None,
            "Attendance_Level": "-",
            "Status": "-"
        }

        # ====================================================
        # GET SAVED MODEL
        # ====================================================

        student_model = models.get(student_id)

        if student_model:

            try:

                model = student_model.get("model")
                history = student_model.get("history")

                if model is not None and history is not None:

                    history = history.copy()

                    history["ds"] = pd.to_datetime(
                        history["ds"]
                    )

                    history["y"] = pd.to_numeric(
                        history["y"],
                        errors="coerce"
                    )

                    history = history.dropna(
                        subset=["ds", "y"]
                    )

                    history = history.sort_values(
                        "ds"
                    )

                    if not history.empty:

                        # ====================================================
                        # CURRENT ATTENDANCE
                        # ====================================================

                        current_attendance = float(
                            history.iloc[-1]["y"]
                        )

                        current_attendance = max(
                            0,
                            min(
                                100,
                                current_attendance
                            )
                        )

                        student_data["Attendance"] = round(
                            current_attendance,
                            2
                        )

                        # ====================================================
                        # LAST ACTUAL MONTH
                        # ====================================================

                        last_date = pd.to_datetime(
                            history.iloc[-1]["ds"]
                        )

                        # ====================================================
                        # NEXT MONTH
                        # ====================================================

                        next_month = (
                            last_date
                            + pd.offsets.MonthBegin(1)
                        ).replace(
                            day=1
                        )

                        # ====================================================
                        # PREDICTION
                        # ====================================================

                        future = model.make_future_dataframe(
                            periods=1,
                            freq="MS"
                        )

                        forecast = model.predict(
                            future
                        )

                        target_rows = forecast[
                            forecast["ds"].dt.to_period("M")
                            == next_month.to_period("M")
                        ]

                        if not target_rows.empty:

                            prediction_value = float(
                                target_rows.iloc[-1]["yhat"]
                            )

                            prediction_value = max(
                                0,
                                min(
                                    100,
                                    prediction_value
                                )
                            )

                            prediction_value = round(
                                prediction_value,
                                2
                            )

                            student_data[
                                "Predicted_Attendance"
                            ] = prediction_value

                            # ====================================================
                            # PREDICTED MONTH
                            # ====================================================

                            student_data[
                                "Predicted_Month"
                            ] = next_month.strftime(
                                "%B %Y"
                            )

                            # ====================================================
                            # ATTENDANCE LEVEL
                            # ====================================================

                            student_data[
                                "Attendance_Level"
                            ] = get_level(
                                prediction_value
                            )

                            # ====================================================
                            # STATUS
                            # ====================================================

                            if prediction_value >= 75:

                                student_data[
                                    "Status"
                                ] = "Safe"

                            elif prediction_value >= 65:

                                student_data[
                                    "Status"
                                ] = "At Risk"

                            else:

                                student_data[
                                    "Status"
                                ] = "High Risk"

            except Exception as e:

                print(
                    f"Prediction error for {student_id}: {e}"
                )

        result.append(
            student_data
        )

    # ========================================================
    # DEPARTMENT CURRENT AVERAGE
    # ========================================================

    valid_current = [
        student["Attendance"]
        for student in result
        if student["Attendance"] is not None
    ]

    department_current_average = None

    if valid_current:

        department_current_average = round(
            sum(valid_current)
            / len(valid_current),
            2
        )

    # ========================================================
    # DEPARTMENT PREDICTED AVERAGE
    # ========================================================

    valid_predictions = [
        student["Predicted_Attendance"]
        for student in result
        if student["Predicted_Attendance"] is not None
    ]

    department_predicted_average = None

    if valid_predictions:

        department_predicted_average = round(
            sum(valid_predictions)
            / len(valid_predictions),
            2
        )

    # ========================================================
    # PREDICTED MONTH
    # ========================================================

    predicted_month = None

    for student in result:

        if student["Predicted_Month"]:

            predicted_month = student[
                "Predicted_Month"
            ]

            break

    # ========================================================
    # SHOW DEPARTMENT ATTENDANCE
    # ========================================================

    return render_template(

        "department_attendance.html",

        department=department,

        students=result,

        total_students=len(result),

        department_current_average=
            department_current_average,

        department_predicted_average=
            department_predicted_average,

        predicted_month=
            predicted_month
    )

# ============================================================
# FACULTY OVERALL PREDICTION
# ============================================================

@app.route(
    "/faculty-overall-prediction",
    methods=["GET", "POST"]
)
def faculty_overall_prediction():

    if "faculty" not in session:
        return redirect(url_for("login"))

    departments = sorted(
        students["Department"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    years = sorted(
        students["Year"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    genders = sorted(
        students["Gender"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    result = None

    if request.method == "POST":

        department = request.form.get(
            "department"
        )

        year = request.form.get(
            "year"
        )

        gender = request.form.get(
            "gender"
        )

        target_month = request.form.get(
            "target_month"
        )

        filtered = students.copy()

        if department:
            filtered = filtered[
                filtered["Department"].astype(str)
                == department
            ]

        if year:
            filtered = filtered[
                filtered["Year"].astype(str)
                == year
            ]

        if gender:
            filtered = filtered[
                filtered["Gender"].astype(str)
                == gender
            ]

        predictions = []

        for _, student in filtered.iterrows():

            student_id = str(
                student["Student_ID"]
            ).strip()

            prediction_value = predict_saved_model(
                student_id,
                target_month
            )

            if prediction_value is not None:

                predictions.append(
                    prediction_value
                )

        if predictions:

            result = {
                "count": len(predictions),
                "average": round(
                    sum(predictions)
                    / len(predictions),
                    2
                ),
                "highest": round(
                    max(predictions),
                    2
                ),
                "lowest": round(
                    min(predictions),
                    2
                ),
                "month": target_month
            }

    return render_template(
        "faculty_overall_prediction.html",

        departments=departments,

        years=years,

        genders=genders,

        months=MONTHS,

        result=result
    )


@app.route("/faculty-dashboard", methods=["GET", "POST"])
def faculty_dashboard():

    if "faculty" not in session:
        return redirect(url_for("login"))

    departments = sorted(
        students["Department"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    selected_department = request.args.get(
        "department",
        ""
    ).strip()

    result = []

    total_students = 0
    current_average = None
    predicted_average = None
    highest_attendance = None
    lowest_attendance = None

    safe_count = 0
    at_risk_count = 0
    high_risk_count = 0

    high_count = 0
    medium_count = 0
    low_count = 0

    if selected_department:

        department_students = students[
            students["Department"]
            .astype(str)
            .str.strip()
            .str.lower()
            == selected_department.lower()
        ].copy()

        total_students = len(department_students)

        for _, student in department_students.iterrows():

            student_id = str(
                student["Student_ID"]
            ).strip()

            student_data = {

                "Student_ID": student_id,

                "Roll_No": student.get(
                    "Roll_No",
                    ""
                ),

                "Name": student.get(
                    "Name",
                    ""
                ),

                "Email": student.get(
                    "Email",
                    ""
                ),

                "Gender": student.get(
                    "Gender",
                    ""
                ),

                "Year": student.get(
                    "Year",
                    ""
                ),

                "Attendance": None,

                "Predicted_Attendance": None,

                "Predicted_Month": None,

                "Attendance_Level": "-",

                "Status": "-"

            }

            student_model = models.get(
                student_id
            )

            if student_model:

                try:

                    model = student_model.get(
                        "model"
                    )

                    history = student_model.get(
                        "history"
                    )

                    if (
                        model is not None
                        and history is not None
                        and not history.empty
                    ):

                        history = history.copy()

                        history["ds"] = pd.to_datetime(
                            history["ds"]
                        )

                        history["y"] = pd.to_numeric(
                            history["y"],
                            errors="coerce"
                        )

                        history = history.dropna(
                            subset=[
                                "ds",
                                "y"
                            ]
                        )

                        history = history.sort_values(
                            "ds"
                        )

                        if not history.empty:

                            # CURRENT ATTENDANCE

                            current = float(
                                history.iloc[-1]["y"]
                            )

                            current = max(
                                0,
                                min(
                                    100,
                                    current
                                )
                            )

                            current = round(
                                current,
                                2
                            )

                            student_data[
                                "Attendance"
                            ] = current

                            # NEXT MONTH

                            last_date = pd.to_datetime(
                                history.iloc[-1]["ds"]
                            )

                            next_month = (
                                last_date
                                + pd.offsets.MonthBegin(1)
                            ).replace(
                                day=1
                            )

                            future = (
                                model.make_future_dataframe(
                                    periods=1,
                                    freq="MS"
                                )
                            )

                            forecast = model.predict(
                                future
                            )

                            target_rows = forecast[
                                forecast["ds"].dt.to_period("M")
                                == next_month.to_period("M")
                            ]

                            if not target_rows.empty:

                                prediction = float(
                                    target_rows.iloc[-1]["yhat"]
                                )

                                prediction = max(
                                    0,
                                    min(
                                        100,
                                        prediction
                                    )
                                )

                                prediction = round(
                                    prediction,
                                    2
                                )

                                student_data[
                                    "Predicted_Attendance"
                                ] = prediction

                                student_data[
                                    "Predicted_Month"
                                ] = next_month.strftime(
                                    "%B %Y"
                                )

                                # ATTENDANCE LEVEL

                                if prediction > 90:

                                    level = "High"

                                    high_count += 1

                                elif prediction >= 75:

                                    level = "Medium"

                                    medium_count += 1

                                else:

                                    level = "Low"

                                    low_count += 1

                                student_data[
                                    "Attendance_Level"
                                ] = level

                                # STATUS

                                if prediction >= 75:

                                    status = "Safe"

                                    safe_count += 1

                                elif prediction >= 65:

                                    status = "At Risk"

                                    at_risk_count += 1

                                else:

                                    status = "High Risk"

                                    high_risk_count += 1

                                student_data[
                                    "Status"
                                ] = status

                except Exception as e:

                    print(
                        f"Dashboard prediction error "
                        f"for {student_id}: {e}"
                    )

            result.append(
                student_data
            )

        # CURRENT ATTENDANCE

        current_values = [

            s["Attendance"]

            for s in result

            if s["Attendance"] is not None

        ]

        if current_values:

            current_average = round(
                sum(current_values)
                / len(current_values),
                2
            )

            highest_attendance = round(
                max(current_values),
                2
            )

            lowest_attendance = round(
                min(current_values),
                2
            )

        # PREDICTED ATTENDANCE

        prediction_values = [

            s["Predicted_Attendance"]

            for s in result

            if s["Predicted_Attendance"] is not None

        ]

        if prediction_values:

            predicted_average = round(
                sum(prediction_values)
                / len(prediction_values),
                2
            )

    return render_template(

        "faculty_dashboard.html",

        departments=departments,

        selected_department=selected_department,

        students=result,

        total_students=total_students,

        current_average=current_average,

        predicted_average=predicted_average,

        highest_attendance=highest_attendance,

        lowest_attendance=lowest_attendance,

        safe_count=safe_count,

        at_risk_count=at_risk_count,

        high_risk_count=high_risk_count,

        high_count=high_count,

        medium_count=medium_count,

        low_count=low_count

    )

# ============================================================
# UPLOAD DATA
# ============================================================

@app.route(
    "/upload-data",
    methods=["GET", "POST"]
)
def upload_data():

    if "faculty" not in session:
        return redirect(url_for("login"))

    result = None
    error = None
    selected_months = []
    prediction_count = 1

    if request.method == "POST":

        file = request.files.get("file")

        try:
            prediction_count = int(
                request.form.get(
                    "prediction_count",
                    "1"
                )
            )
        except:
            prediction_count = 1

        if prediction_count not in [1, 2, 3]:
            prediction_count = 1

        if file is None or file.filename == "":

            error = "Please select an Excel file."

        elif not file.filename.lower().endswith(
            (".xlsx", ".xls")
        ):

            error = "Please upload an Excel file."

        else:

            filename = secure_filename(
                file.filename
            )

            filepath = os.path.join(
                app.config["UPLOAD_FOLDER"],
                filename
            )

            file.save(filepath)

            try:

                # ========================================================
                # READ EXCEL
                # ========================================================

                uploaded = pd.read_excel(
                    filepath
                )

                uploaded.columns = (
                    uploaded.columns
                    .astype(str)
                    .str.strip()
                )

                print("\n===================================")
                print("UPLOADED EXCEL COLUMNS")
                print("===================================")

                print(
                    uploaded.columns.tolist()
                )

                print(
                    "Rows:",
                    len(uploaded)
                )

                print("===================================\n")


                # ========================================================
                # REQUIRED STUDENT COLUMNS
                # ========================================================

                required_columns = [
                    "Student_ID",
                    "Roll_No",
                    "Name",
                    "Email",
                    "DOB",
                    "Gender",
                    "Department",
                    "Year"
                ]

                missing = [
                    column
                    for column in required_columns
                    if column not in uploaded.columns
                ]


                if missing:

                    error = (
                        "Missing columns: "
                        + ", ".join(missing)
                    )

                else:

                    # ====================================================
                    # MONTH ALIASES
                    # ====================================================

                    month_aliases = {

                        "january": 1,
                        "jan": 1,

                        "february": 2,
                        "feb": 2,

                        "march": 3,
                        "mar": 3,

                        "april": 4,
                        "apr": 4,

                        "may": 5,

                        "june": 6,
                        "jun": 6,

                        "july": 7,
                        "jul": 7,

                        "august": 8,
                        "aug": 8,

                        "september": 9,
                        "sep": 9,
                        "sept": 9,

                        "october": 10,
                        "oct": 10,

                        "november": 11,
                        "nov": 11,

                        "december": 12,
                        "dec": 12
                    }


                    # ====================================================
                    # FIND MONTH COLUMNS
                    # ====================================================

                    month_data = {}

                    for column in uploaded.columns:

                        original_column = str(
                            column
                        ).strip()

                        normalized_column = (
                            original_column
                            .lower()
                            .replace("-", "_")
                            .replace(" ", "_")
                        )

                        found_month = None

                        # ------------------------------------------------
                        # EXACT MONTH
                        # ------------------------------------------------

                        if normalized_column in month_aliases:

                            found_month = (
                                month_aliases[
                                    normalized_column
                                ]
                            )

                        # ------------------------------------------------
                        # COLUMN ENDING WITH MONTH
                        #
                        # Examples:
                        # NLP_Jan
                        # AI_Mar
                        # Attendance_Nov
                        # ------------------------------------------------

                        if found_month is None:

                            for alias, month_number in (
                                month_aliases.items()
                            ):

                                if (
                                    normalized_column.endswith(
                                        "_" + alias
                                    )
                                    or
                                    normalized_column.endswith(
                                        alias
                                    )
                                ):

                                    found_month = (
                                        month_number
                                    )

                                    break


                        if found_month is not None:

                            month_data[
                                found_month
                            ] = original_column


                    # ====================================================
                    # ALSO CHECK DATE-TYPE COLUMN HEADERS
                    # ====================================================

                    if len(month_data) < 2:

                        for column in uploaded.columns:

                            try:

                                parsed = pd.to_datetime(
                                    str(column),
                                    errors="coerce"
                                )

                                if pd.notna(parsed):

                                    month_number = (
                                        parsed.month
                                    )

                                    if (
                                        1
                                        <= month_number
                                        <= 12
                                    ):

                                        month_data[
                                            month_number
                                        ] = str(column)

                            except:

                                pass


                    # ====================================================
                    # SORT MONTHS
                    # ====================================================

                    month_numbers = sorted(
                        month_data.keys()
                    )


                    print(
                        "Detected attendance months:"
                    )

                    for month_number in month_numbers:

                        print(
                            month_number,
                            "->",
                            month_data[
                                month_number
                            ]
                        )


                    # ====================================================
                    # CHECK MONTH COUNT
                    # ====================================================

                    if len(month_numbers) < 2:

                        error = (
                            "Attendance columns were not detected. "
                            "Please make sure your Excel contains "
                            "at least two month columns such as "
                            "Jan, Feb, Mar... or "
                            "January, February, March..."
                        )

                    else:

                        # =================================================
                        # LAST AVAILABLE MONTH
                        # =================================================

                        last_month_number = (
                            month_numbers[-1]
                        )

                        last_year = 2026

                        last_date = pd.Timestamp(
                            year=last_year,
                            month=last_month_number,
                            day=1
                        )


                        # =================================================
                        # FUTURE MONTHS
                        # =================================================

                        selected_months = []

                        for i in range(
                            1,
                            prediction_count + 1
                        ):

                            future_date = (
                                last_date
                                + pd.DateOffset(
                                    months=i
                                )
                            )

                            selected_months.append(
                                future_date
                            )


                        print(
                            "\nPrediction months:"
                        )

                        for date in selected_months:

                            print(
                                date.strftime(
                                    "%B %Y"
                                )
                            )


                        # =================================================
                        # PROCESS STUDENTS
                        # =================================================

                        students_found = []


                        for _, row in uploaded.iterrows():

                            student_id = str(
                                row["Student_ID"]
                            ).strip()


                            # ---------------------------------------------
                            # ATTENDANCE HISTORY
                            # ---------------------------------------------

                            history_rows = []


                            for month_number in month_numbers:

                                column_name = (
                                    month_data[
                                        month_number
                                    ]
                                )

                                value = row[
                                    column_name
                                ]


                                if pd.isna(value):
                                    continue


                                try:

                                    value = float(
                                        value
                                    )

                                except:

                                    continue


                                if (
                                    value < 0
                                    or value > 100
                                ):
                                    continue


                                history_rows.append({

                                    "ds":
                                        pd.Timestamp(
                                            year=2026,
                                            month=month_number,
                                            day=1
                                        ),

                                    "y":
                                        value
                                })


                            # ---------------------------------------------
                            # MINIMUM TWO MONTHS
                            # ---------------------------------------------

                            if len(history_rows) < 2:

                                print(
                                    f"Skipping {student_id}: "
                                    f"less than two valid months"
                                )

                                continue


                            history = pd.DataFrame(
                                history_rows
                            )


                            history = (
                                history
                                .sort_values("ds")
                                .drop_duplicates(
                                    subset=["ds"],
                                    keep="last"
                                )
                            )


                            # ---------------------------------------------
                            # CURRENT ATTENDANCE
                            # ---------------------------------------------

                            current_attendance = round(
                                float(
                                    history.iloc[-1]["y"]
                                ),
                                2
                            )


                            # ---------------------------------------------
                            # CREATE PROPHET MODEL
                            # ---------------------------------------------

                            prediction_values = []


                            try:

                                model = Prophet(

                                    yearly_seasonality=False,

                                    weekly_seasonality=False,

                                    daily_seasonality=False
                                )


                                model.fit(
                                    history
                                )


                                # -----------------------------------------
                                # NUMBER OF MONTHS
                                # -----------------------------------------

                                months_ahead = (
                                    (
                                        selected_months[-1].year
                                        - history.iloc[-1]["ds"].year
                                    )
                                    * 12
                                    +
                                    (
                                        selected_months[-1].month
                                        - history.iloc[-1]["ds"].month
                                    )
                                )


                                # -----------------------------------------
                                # FUTURE DATA
                                # -----------------------------------------

                                future = (
                                    model
                                    .make_future_dataframe(
                                        periods=months_ahead,
                                        freq="MS"
                                    )
                                )


                                # -----------------------------------------
                                # FORECAST
                                # -----------------------------------------

                                forecast = model.predict(
                                    future
                                )


                                # -----------------------------------------
                                # GET EACH REQUIRED MONTH
                                # -----------------------------------------

                                for future_date in selected_months:

                                    target = forecast[
                                        forecast["ds"]
                                        .dt.to_period("M")
                                        ==
                                        future_date.to_period("M")
                                    ]


                                    if target.empty:

                                        prediction_values.append(
                                            None
                                        )

                                    else:

                                        prediction = float(
                                            target.iloc[-1]["yhat"]
                                        )


                                        prediction = max(
                                            0,
                                            min(
                                                100,
                                                prediction
                                            )
                                        )


                                        prediction = round(
                                            prediction,
                                            2
                                        )


                                        prediction_values.append(
                                            prediction
                                        )


                            except Exception as e:

                                print(
                                    f"Prediction error "
                                    f"for {student_id}: "
                                    f"{e}"
                                )


                                prediction_values = [
                                    None
                                    for _
                                    in selected_months
                                ]


                            # ---------------------------------------------
                            # STUDENT RESULT
                            # ---------------------------------------------

                            student_result = {

                                "Student_ID":
                                    student_id,

                                "Roll_No":
                                    row["Roll_No"],

                                "Name":
                                    row["Name"],

                                "Email":
                                    row["Email"],

                                "DOB":
                                    row["DOB"],

                                "Gender":
                                    row["Gender"],

                                "Department":
                                    row["Department"],

                                "Year":
                                    row["Year"],

                                "Current_Attendance":
                                    current_attendance
                            }


                            # ---------------------------------------------
                            # ADD PREDICTIONS
                            # ---------------------------------------------

                            for index, future_date in enumerate(
                                selected_months
                            ):

                                month_key = (
                                    future_date.strftime(
                                        "%Y_%m"
                                    )
                                )


                                student_result[
                                    "Prediction_"
                                    + month_key
                                ] = (
                                    prediction_values[
                                        index
                                    ]
                                )


                            students_found.append(
                                student_result
                            )


                        # =================================================
                        # FINAL RESULT
                        # =================================================

                        result = students_found


                        # =================================================
                        # CREATE DOWNLOAD FILE
                        # =================================================

                        download_rows = []


                        for student in result:

                            download_row = {

                                "Student_ID":
                                    student[
                                        "Student_ID"
                                    ],

                                "Roll_No":
                                    student[
                                        "Roll_No"
                                    ],

                                "Name":
                                    student[
                                        "Name"
                                    ],

                                "Email":
                                    student[
                                        "Email"
                                    ],

                                "Gender":
                                    student[
                                        "Gender"
                                    ],

                                "Department":
                                    student[
                                        "Department"
                                    ],

                                "Year":
                                    student[
                                        "Year"
                                    ],

                                "Current_Attendance":
                                    student[
                                        "Current_Attendance"
                                    ]
                            }


                            for future_date in selected_months:

                                month_key = (
                                    future_date.strftime(
                                        "%Y_%m"
                                    )
                                )


                                download_row[
                                    future_date.strftime(
                                        "%B %Y"
                                    )
                                ] = student.get(
                                    "Prediction_"
                                    + month_key
                                )


                            download_rows.append(
                                download_row
                            )


                        prediction_dataframe = pd.DataFrame(
                            download_rows
                        )


                        download_file = os.path.join(
                            app.config[
                                "UPLOAD_FOLDER"
                            ],
                            "predicted_attendance.xlsx"
                        )


                        prediction_dataframe.to_excel(
                            download_file,
                            index=False
                        )


                        print(
                            "\nPrediction completed."
                        )

                        print(
                            "Students predicted:",
                            len(result)
                        )

                        print(
                            "File saved:",
                            download_file
                        )


            except Exception as e:

                print(
                    "UPLOAD ERROR:",
                    e
                )

                error = (
                    "Error while processing Excel: "
                    + str(e)
                )


    return render_template(

        "upload_data.html",

        result=result,

        error=error,

        selected_months=selected_months,

        prediction_count=prediction_count
    )


# ============================================================
# DOWNLOAD PREDICTED FILE
# ============================================================

@app.route("/download-predictions")
def download_predictions():

    if "faculty" not in session:
        return redirect(url_for("login"))

    download_file = os.path.join(
        app.config["UPLOAD_FOLDER"],
        "predicted_attendance.xlsx"
    )

    if not os.path.exists(download_file):

        return redirect(
            url_for("upload_data")
        )

    from flask import send_file

    return send_file(

        download_file,

        as_attachment=True,

        download_name="predicted_attendance.xlsx"
    )

# ============================================================
# ADD NEW STUDENT
# ============================================================



@app.route("/add-student", methods=["GET", "POST"])
def add_student():

    if "faculty" not in session:
        return redirect(url_for("login"))

    if request.method == "GET":

        return render_template(
            "add_student.html"
        )

    # Your existing add-student processing code goes here

    # ========================================================
    # STUDENT DETAILS
    # ========================================================

    student = {

        "Student_ID":
            request.form.get(
                "student_id",
                ""
            ).strip(),

        "Roll_No":
            request.form.get(
                "roll_no",
                ""
            ).strip(),

        "Name":
            request.form.get(
                "name",
                ""
            ).strip(),

        "Email":
            request.form.get(
                "email",
                ""
            ).strip(),

        "DOB":
            request.form.get(
                "dob",
                ""
            ).strip(),

        "Gender":
            request.form.get(
                "gender",
                ""
            ).strip(),

        "Department":
            request.form.get(
                "department",
                ""
            ).strip(),

        "Year":
            request.form.get(
                "year",
                ""
            ).strip()
    }


    # ========================================================
    # ATTENDANCE
    # ========================================================

    month_fields = {

        "2026-01-01": "jan",
        "2026-02-01": "feb",
        "2026-03-01": "mar",
        "2026-04-01": "apr",
        "2026-05-01": "may",
        "2026-06-01": "jun",
        "2026-07-01": "jul",
        "2026-08-01": "aug",
        "2026-09-01": "sep",
        "2026-10-01": "oct",
        "2026-11-01": "nov"
    }


    attendance = {}


    for month, field in month_fields.items():

        value = request.form.get(
            field,
            ""
        ).strip()


        if value != "":

            try:

                value = float(value)

            except ValueError:

                return render_template(
                    "add_student.html",
                    error=f"Invalid attendance value for {field.upper()}."
                )


            if value < 0 or value > 100:

                return render_template(
                    "add_student.html",
                    error=f"Attendance for {field.upper()} must be between 0 and 100."
                )


            attendance[month] = value


    # ========================================================
    # CHECK MINIMUM DATA
    # ========================================================

    if len(attendance) < 2:

        return render_template(
            "add_student.html",
            error="Please enter attendance for at least two months."
        )


    # ========================================================
    # PREDICTION MONTH
    # ========================================================

    prediction_month = request.form.get(
        "prediction_month",
        ""
    ).strip()


    if not prediction_month:

        return render_template(
            "add_student.html",
            error="Please select a prediction month."
        )


    # ========================================================
    # CREATE MODEL
    # ========================================================

    try:

        model, history = create_student_model(
            attendance
        )


        prediction_value = predict_month(
            model,
            history,
            prediction_month
        )


    except Exception as e:

        return render_template(
            "add_student.html",
            error=str(e)
        )


    # ========================================================
    # CURRENT ATTENDANCE
    # ========================================================

    latest_attendance = float(
        history.iloc[-1]["y"]
    )

    latest_attendance = round(
        latest_attendance,
        2
    )


    # ========================================================
    # PREDICTION INFORMATION
    # ========================================================

    prediction_date = pd.to_datetime(
        prediction_month
    )


    predicted_month = prediction_date.strftime(
        "%B %Y"
    )


    attendance_level = get_attendance_level(
        prediction_value
    )


    status = get_status(
        prediction_value
    )


    # ========================================================
    # GRAPH DATA
    # ========================================================

    chart_labels = [
        date.strftime("%B")
        for date in history["ds"]
    ]


    chart_values = [
        round(float(value), 2)
        for value in history["y"]
    ]


    chart_labels.append(
        predicted_month
    )


    chart_values.append(
        prediction_value
    )


    # ========================================================
    # SHOW RESULT
    # ========================================================

    return render_template(

        "new_student_result.html",

        student=student,

        current_attendance=latest_attendance,

        predicted_value=prediction_value,

        predicted_month=predicted_month,

        attendance_level=attendance_level,

        status=status,

        chart_labels=chart_labels,

        chart_values=chart_values
    )



# ============================================================
# LOGOUT
# ============================================================

@app.route("/logout")
def logout():

    session.clear()

    return redirect(
        url_for("login")
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )
