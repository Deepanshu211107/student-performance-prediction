import pandas as pd
import joblib
import shap


# Load regression model and preprocessor
model = joblib.load("random_forest_regressor.pkl")
preprocessor = joblib.load("preprocessor.pkl")


# New student
student = {
    "school": "GP",
    "sex": "M",
    "age": 17,
    "address": "U",
    "famsize": "GT3",
    "Pstatus": "T",
    "Medu": 3,
    "Fedu": 2,
    "Mjob": "services",
    "Fjob": "services",
    "reason": "course",
    "guardian": "mother",
    "traveltime": 2,
    "studytime": 2,
    "failures": 1,
    "schoolsup": "yes",
    "famsup": "yes",
    "paid": "no",
    "activities": "no",
    "nursery": "yes",
    "higher": "yes",
    "internet": "yes",
    "romantic": "no",
    "famrel": 4,
    "freetime": 3,
    "goout": 3,
    "Dalc": 1,
    "Walc": 1,
    "health": 3,
    "absences": 6
}


# Convert to DataFrame
student_df = pd.DataFrame([student])


# Preprocess student
student_processed = preprocessor.transform(student_df)


# Predict score
predicted_score = model.predict(student_processed)[0]

# Keep score within 0-20
predicted_score = max(0, min(20, predicted_score))


# SHAP explainer
explainer = shap.TreeExplainer(model)

shap_values = explainer(student_processed)

# SHAP values for regression
shap_values_student = shap_values.values[0]


# Get encoded feature names
feature_names = preprocessor.get_feature_names_out()


# Aggregate one-hot encoded features
aggregated_shap = {}

for feature_name, shap_value in zip(
    feature_names,
    shap_values_student
):

    if "__" in feature_name:
        original_name = feature_name.split("__", 1)[1]
    else:
        original_name = feature_name

    if feature_name.startswith("numerical__"):
        base_feature = original_name
    else:
        base_feature = original_name.split("_")[0]

    aggregated_shap[base_feature] = (
        aggregated_shap.get(base_feature, 0) + shap_value
    )


# Human-readable feature names
readable_feature_names = {
    "school": "School",
    "sex": "Sex",
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
    "activities": "Extra-curricular Activities",
    "nursery": "Nursery Education",
    "higher": "Higher Education",
    "internet": "Internet Access",
    "romantic": "Romantic Relationship",
    "famrel": "Family Relationship",
    "freetime": "Free Time",
    "goout": "Going Out",
    "Dalc": "Workday Alcohol Consumption",
    "Walc": "Weekend Alcohol Consumption",
    "health": "Health",
    "absences": "Absences"
}


# Sort by absolute SHAP value
sorted_features = sorted(
    aggregated_shap.items(),
    key=lambda x: abs(x[1]),
    reverse=True
)


print("=" * 60)
print("STUDENT PERFORMANCE PREDICTION")
print("=" * 60)

print(f"\nPredicted Final Score: {predicted_score:.2f} / 20")

if predicted_score < 10:
    print("Performance Level: Below Passing")
elif predicted_score < 15:
    print("Performance Level: Average")
else:
    print("Performance Level: Good")


print("\n" + "=" * 60)
print("MAIN FACTORS INFLUENCING THE PREDICTION")
print("=" * 60)


# Display top 8 factors
for feature, shap_value in sorted_features[:8]:

    actual_value = student_df.iloc[0][feature]

    display_name = readable_feature_names.get(
        feature,
        feature
    )

    if shap_value > 0:
        direction = "increased the predicted score"
        arrow = "↑"
    else:
        direction = "decreased the predicted score"
        arrow = "↓"

    print(f"\n{display_name} = {actual_value}")
    print(
        f"  {arrow} {direction} "
        f"({shap_value:+.3f})"
    )