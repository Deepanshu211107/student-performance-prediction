import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

df = pd.read_csv("student/student-mat.csv", sep=";")


# --------------------------------------------------
# 2. Create target
# --------------------------------------------------

# 1 = At Risk
# 0 = Not At Risk
df["performance"] = (df["G3"] < 10).astype(int)


# --------------------------------------------------
# 3. Separate features and target
# --------------------------------------------------

X = df.drop(columns=["G3", "G1", "G2", "performance"])
y = df["performance"]


# --------------------------------------------------
# 4. Identify column types
# --------------------------------------------------

categorical_columns = X.select_dtypes(include=["object"]).columns
numerical_columns = X.select_dtypes(exclude=["object"]).columns


# --------------------------------------------------
# 5. Train-test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# 6. Preprocessing
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        ),
        (
            "numerical",
            "passthrough",
            numerical_columns
        )
    ]
)


# Fit only on training data
X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)


# --------------------------------------------------
# 7. Get processed feature names
# --------------------------------------------------

feature_names = preprocessor.get_feature_names_out()


# --------------------------------------------------
# 8. Train Random Forest
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train_processed, y_train)


# --------------------------------------------------
# 9. Select one student
# --------------------------------------------------

student_index = 0

student_data = X_test_processed[
    student_index:student_index + 1
]


# --------------------------------------------------
# 10. Prediction
# --------------------------------------------------

prediction = model.predict(student_data)[0]

probability = model.predict_proba(student_data)[0]


print("\nPrediction:")

if prediction == 1:
    print("AT RISK")
else:
    print("NOT AT RISK")


print("\nProbabilities:")

print(f"Not At Risk: {probability[0] * 100:.2f}%")
print(f"At Risk: {probability[1] * 100:.2f}%")


# --------------------------------------------------
# 11. SHAP explanation
# --------------------------------------------------

explainer = shap.TreeExplainer(model)

shap_values = explainer(student_data)


print("\nSHAP values shape:")
print(shap_values.values.shape)

print("\nNumber of features:")
print(len(feature_names))


# --------------------------------------------------
# 12. Get SHAP values for At-Risk class
# --------------------------------------------------

at_risk_shap_values = shap_values.values[0, :, 1]


# --------------------------------------------------
# 13. Convert feature names to readable names
# --------------------------------------------------

def make_readable_name(feature_name):

    # Remove preprocessing prefixes
    name = feature_name.replace(
        "numerical__",
        ""
    )

    name = name.replace(
        "categorical__",
        ""
    )

    numerical_names = {
        "failures": "Previous Failures",
        "absences": "Absences",
        "studytime": "Study Time",
        "age": "Age",
        "famrel": "Family Relationship",
        "freetime": "Free Time",
        "goout": "Going Out",
        "Dalc": "Workday Alcohol Consumption",
        "Walc": "Weekend Alcohol Consumption",
        "health": "Health",
        "Medu": "Mother's Education",
        "Fedu": "Father's Education",
        "traveltime": "Travel Time"
    }

    if name in numerical_names:
        return numerical_names[name]

    categorical_names = {
        "school": "School",
        "sex": "Gender",
        "address": "Address",
        "famsize": "Family Size",
        "Pstatus": "Parent Status",
        "Mjob": "Mother's Job",
        "Fjob": "Father's Job",
        "reason": "Reason for Choosing School",
        "guardian": "Guardian",
        "schoolsup": "School Support",
        "famsup": "Family Support",
        "paid": "Extra Paid Classes",
        "activities": "Extracurricular Activities",
        "nursery": "Nursery Education",
        "higher": "Higher Education Intention",
        "internet": "Internet Access",
        "romantic": "Romantic Relationship"
    }

    parts = name.split("_")

    if len(parts) >= 2:

        feature = parts[0]

        value = "_".join(parts[1:])

        readable_feature = categorical_names.get(
            feature,
            feature
        )

        return f"{readable_feature} = {value}"

    return name


readable_names = [
    make_readable_name(name)
    for name in feature_names
]


# --------------------------------------------------
# 14. Create readable SHAP explanation
# --------------------------------------------------

student_values = (
    student_data.toarray()[0]
    if hasattr(student_data, "toarray")
    else student_data[0]
)


readable_explanation = shap.Explanation(
    values=at_risk_shap_values,
    base_values=shap_values.base_values[0, 1],
    data=student_values,
    feature_names=readable_names
)


# --------------------------------------------------
# 15. Display SHAP chart
# --------------------------------------------------

shap.plots.bar(
    readable_explanation,
    max_display=15,
    show=False
)

plt.tight_layout()
plt.show()


# --------------------------------------------------
# 16. Generate human-readable explanations
# --------------------------------------------------

# Create a DataFrame containing SHAP information
explanation_df = pd.DataFrame({
    "feature": readable_names,
    "shap_value": at_risk_shap_values,
    "student_value": student_values
})


# Sort by absolute SHAP value
explanation_df["absolute_shap"] = (
    explanation_df["shap_value"].abs()
)

explanation_df = explanation_df.sort_values(
    by="absolute_shap",
    ascending=False
)


print("\n" + "=" * 60)
print("WHY DID THE MODEL MAKE THIS PREDICTION?")
print("=" * 60)


# Display the 8 most influential features
top_features = explanation_df.head(8)


for _, row in top_features.iterrows():

    feature = row["feature"]
    shap_value = row["shap_value"]

    if shap_value > 0:

        print(
            f"↑ {feature} "
            f"contributed toward an AT-RISK prediction."
        )

    else:

        print(
            f"↓ {feature} "
            f"contributed away from an AT-RISK prediction."
        )
        