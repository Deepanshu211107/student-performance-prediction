from flask import Flask, render_template, request
import pandas as pd
import joblib
import shap

app = Flask(__name__)


# ==========================================
# Load Model and Preprocessor
# ==========================================

model = joblib.load("gradient_boosting_regressor.pkl")
preprocessor = joblib.load("preprocessor.pkl")

explainer = shap.TreeExplainer(model)


# ==========================================
# Readable Feature Names
# ==========================================

readable_feature_names = {

    "school": "School Type",
    "sex": "Gender",
    "age": "Age",
    "address": "Address",
    "famsize": "Family Size",
    "Pstatus": "Parent Status",

    "Medu": "Mother's Education",
    "Fedu": "Father's Education",

    "Mjob": "Mother's Job",
    "Fjob": "Father's Job",

    "reason": "Reason for Choosing School",
    "guardian": "Guardian",

    "traveltime": "Travel Time",
    "studytime": "Study Time",

    "failures": "Previous Failures",

    "schoolsup": "School Support",
    "famsup": "Family Support",

    "paid": "Paid Classes",
    "activities": "Extracurricular Activities",

    "nursery": "Nursery Education",
    "higher": "Higher Education Intention",
    "internet": "Internet Access",

    "famrel": "Family Relationship Quality",
    "freetime": "Free Time",
    "goout": "Going Out",

    "health": "Health Status",
    "absences": "Absences"
}


# ==========================================
# Home Page
# ==========================================

@app.route("/")
def home():

    return render_template("index.html")


# ==========================================
# Prediction
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    # --------------------------------------
    # Collect Student Data
    # --------------------------------------

    student = {

        "school": request.form["school"],

        "sex": request.form["sex"],

        "age": int(request.form["age"]),

        "address": request.form["address"],

        "famsize": request.form["famsize"],

        "Pstatus": request.form["Pstatus"],

        "Medu": int(request.form["Medu"]),

        "Fedu": int(request.form["Fedu"]),

        "Mjob": request.form["Mjob"],

        "Fjob": request.form["Fjob"],

        "reason": request.form["reason"],

        "guardian": request.form["guardian"],

        "traveltime": int(request.form["traveltime"]),

        "studytime": int(request.form["studytime"]),

        "failures": int(request.form["failures"]),

        "schoolsup": request.form["schoolsup"],

        "famsup": request.form["famsup"],

        "paid": request.form["paid"],

        "activities": request.form["activities"],

        "nursery": request.form["nursery"],

        "higher": request.form["higher"],

        "internet": request.form["internet"],

        "famrel": int(request.form["famrel"]),

        "freetime": int(request.form["freetime"]),

        "goout": int(request.form["goout"]),

        "health": int(request.form["health"]),

        "absences": int(request.form["absences"])
    }


    # --------------------------------------
    # Convert to DataFrame
    # --------------------------------------

    student_df = pd.DataFrame([student])


    # --------------------------------------
    # Preprocess
    # --------------------------------------

    student_processed = preprocessor.transform(
        student_df
    )


    # --------------------------------------
    # Prediction
    # --------------------------------------

    prediction = model.predict(
        student_processed
    )[0]


    # Keep prediction between 0 and 20

    prediction = max(
        0,
        min(20, prediction)
    )


    # --------------------------------------
    # Performance Level
    # --------------------------------------

    if prediction < 10:

        performance_level = "Below Passing"

    elif prediction < 15:

        performance_level = "Average"

    else:

        performance_level = "Good"


    # --------------------------------------
    # Score Percentage
    # --------------------------------------

    # score_percentage = (
    #     prediction / 20
    # ) * 100


    # ======================================
    # SHAP Explanation
    # ======================================

    shap_values = explainer(
        student_processed
    )

    shap_values_student = shap_values.values[0]

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )


    # --------------------------------------
    # Aggregate SHAP Values
    # by Original Feature
    # --------------------------------------

    aggregated_shap = {}

    original_columns = list(
        student_df.columns
    )


    for feature_name, shap_value in zip(
        feature_names,
        shap_values_student
    ):

        # Numerical feature

        if feature_name.startswith(
            "numerical__"
        ):

            base_feature = feature_name.replace(
                "numerical__",
                "",
                1
            )


        # Categorical feature

        elif feature_name.startswith(
            "categorical__"
        ):

            encoded_feature = feature_name.replace(
                "categorical__",
                "",
                1
            )

            base_feature = None


            # Find original column name

            for column in original_columns:

                if (
                    encoded_feature == column
                    or encoded_feature.startswith(
                        column + "_"
                    )
                ):

                    base_feature = column

                    break


            # Safety check

            if base_feature is None:

                continue


        else:

            continue


        # Add SHAP contribution

        aggregated_shap[base_feature] = (
            aggregated_shap.get(
                base_feature,
                0
            )
            + shap_value
        )


    # --------------------------------------
    # Sort Features by Importance
    # --------------------------------------

    sorted_features = sorted(
        aggregated_shap.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )


    # --------------------------------------
    # Create Explanation List
    # --------------------------------------

    explanations = []


    for feature, shap_value in sorted_features[:8]:

        actual_value = (
            student_df.iloc[0][feature]
        )

        display_name = (
            readable_feature_names.get(
                feature,
                feature
            )
        )


        if shap_value > 0:

            direction = (
                "Increased predicted score"
            )

            arrow = "↑"

            positive = True

        else:

            direction = (
                "Decreased predicted score"
            )

            arrow = "↓"

            positive = False


        explanations.append({

            "feature": display_name,

            "value": actual_value,

            "direction": direction,

            "arrow": arrow,

            "shap": round(
                float(shap_value),
                3
            ),

            "positive": positive
        })


    # ======================================
    # Send Result to HTML
    # ======================================

    return render_template(
        "result.html",
        prediction=round(prediction, 2),
        performance_level=performance_level,
        # score_percentage=round(score_percentage, 2),
        explanations=explanations,

        model_name="Gradient Boosting",

        model_metrics={
            "mae": 3.05,
            "mse": 14.64,
            "r2": 0.29
        },

        model_comparison=[
            {
                "name": "Linear Regression",
                "mae": 3.51,
                "mse": 18.19,
                "r2": 0.11
            },
            {
                "name": "Decision Tree",
                "mae": 3.45,
                "mse": 19.67,
                "r2": 0.04
            },
            {
                "name": "Random Forest",
                "mae": 3.09,
                "mse": 14.63,
                "r2": 0.29
            },
            {
                "name": "Gradient Boosting",
                "mae": 3.05,
                "mse": 14.64,
                "r2": 0.29
            }
        ]
    )


# ==========================================
# Run Flask
# ==========================================

if __name__ == "__main__":

    app.run(
        debug=True
    )



