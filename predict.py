import pandas as pd
import joblib


# Load model and preprocessor
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


# Preprocess
student_processed = preprocessor.transform(student_df)


# Predict score
predicted_score = model.predict(student_processed)[0]


# Keep prediction within valid G3 range
predicted_score = max(0, min(20, predicted_score))


print("=" * 50)
print("STUDENT PERFORMANCE PREDICTION")
print("=" * 50)

print(f"\nPredicted Final Score: {predicted_score:.2f} / 20")

if predicted_score < 10:
    print("Performance Level: Below Passing")
elif predicted_score < 15:
    print("Performance Level: Average")
else:
    print("Performance Level: Good")

print("=" * 50)
