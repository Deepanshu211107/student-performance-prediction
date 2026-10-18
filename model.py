import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==============================
# 1. Load Dataset
# ==============================

df = pd.read_csv("student/student-mat.csv", sep=";")


# ==============================
# 2. Separate Features and Target
# ==============================

X = df.drop(
    columns=[
        "G3",
        "G1",
        "G2",
        "romantic",
        "Dalc",
        "Walc"
    ]
)

y = df["G3"]


# ==============================
# 3. Identify Column Types
# ==============================

categorical_columns = X.select_dtypes(
    include=["object", "str"]
).columns

numerical_columns = X.select_dtypes(
    exclude=["object", "str"]
).columns


# ==============================
# 4. Preprocessing
# ==============================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            "passthrough",
            numerical_columns
        ),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ]
)


# ==============================
# 5. Train-Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==============================
# 6. Transform Data
# ==============================

X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)


# ==============================
# 7. Create Gradient Boosting Model
# ==============================

model = GradientBoostingRegressor(
    n_estimators=100,
    max_depth=3,
    random_state=42
)


# ==============================
# 8. Train Model
# ==============================

model.fit(
    X_train_processed,
    y_train
)


# ==============================
# 9. Make Predictions
# ==============================

y_pred = model.predict(
    X_test_processed
)


# ==============================
# 10. Evaluate Model
# ==============================

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

r2 = r2_score(
    y_test,
    y_pred
)


print("Model Training Complete")
print("-----------------------")

print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")


# ==============================
# 11. Save Model and Preprocessor
# ==============================

joblib.dump(
    model,
    "gradient_boosting_regressor.pkl"
)

joblib.dump(
    preprocessor,
    "preprocessor.pkl"
)


print("\nModel saved successfully!")

print("gradient_boosting_regressor.pkl")
print("preprocessor.pkl")