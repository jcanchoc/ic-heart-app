from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("model_svm_pipeline.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        age = int(request.form["Age"])
        sex = request.form["Sex"]
        chest_pain = request.form["ChestPainType"]
        cholesterol = int(request.form["Cholesterol"])
        fasting_bs = int(request.form["FastingBS"])
        max_hr = int(request.form["MaxHR"])
        exercise_angina = request.form["ExerciseAngina"]
        oldpeak = float(request.form["Oldpeak"])
        st_slope = request.form["ST_Slope"]
        
        input_df = pd.DataFrame([{
            "Age": age,
            "Sex": sex,
            "ChestPainType": chest_pain,
            "Cholesterol": cholesterol,
            "FastingBS": fasting_bs,
            "MaxHR": max_hr,
            "ExerciseAngina": exercise_angina,
            "Oldpeak": oldpeak,
            "ST_Slope": st_slope
        }])

        prediction = model.predict(input_df)[0]

        proba = model.predict_proba(input_df)[0][1]
        probability = round(proba * 100, 2)

        return render_template("index.html", prediction=int(prediction), probability=probability)

    return render_template("index.html", prediction=None)

if __name__ == "__main__":
    app.run(debug=True)
