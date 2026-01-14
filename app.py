from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# load model & preprocessor
model = pickle.load(open("Artifacts/model.pkl", "rb"))
preprocessor = pickle.load(open("Artifacts/preprocessor.pkl", "rb"))


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        # ===== รับค่าจากฟอร์ม =====
        time_of_day = request.form["Time_of_Day"]
        traffic = request.form["Traffic_Level"]

        # ===== Feature Engineering (Peak) =====
        # Is Peak Hour
        is_peak = 1 if time_of_day in ["Lunch", "Dinner"] else 0

        # Peak Score
        peak_score = 0
        if is_peak == 1:
            peak_score += 1

        if traffic == "High":
            peak_score += 2
        elif traffic == "Medium":
            peak_score += 1

        # ===== สร้าง DataFrame =====
        data = {
            "Distance_km": float(request.form["Distance_km"]),
            "Preparation_Time_min": float(request.form["Preparation_Time_min"]),
            "Courier_Experience_yrs": float(request.form["Courier_Experience_yrs"]),
            "Weather": request.form["Weather"],
            "Time_of_Day": time_of_day,
            "Traffic_Level": traffic,
            "Vehicle_Type": request.form["Vehicle_Type"],
            "Peak_Score": peak_score,
            "Is_Peak_Hour": is_peak,
        }

        df = pd.DataFrame([data])

        # ===== Predict =====
        X = preprocessor.transform(df)
        prediction = round(float(model.predict(X)[0]), 2)

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
