import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==============================
# 1. Load Dataset
# ==============================

df = pd.read_csv("student/student-mat.csv", sep=";")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


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
# 7. Define Models
# ==============================

models = {

    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        max_depth=6,
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        max_depth=3,
        random_state=42
    )
}


# ==============================
# 8. Train and Evaluate Models
# ==============================

results = []

print("\nModel Comparison")
print("================")

for name, model in models.items():

    model.fit(
        X_train_processed,
        y_train
    )

    y_pred = model.predict(
        X_test_processed
    )

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

    results.append({
        "Model": name,
        "MAE": mae,
        "MSE": mse,
        "R2": r2
    })

    print(f"\n{name}")
    print(f"MAE : {mae:.2f}")
    print(f"MSE : {mse:.2f}")
    print(f"R²  : {r2:.2f}")


# ==============================
# 9. Display Comparison
# ==============================

results_df = pd.DataFrame(results)

print("\n\nFinal Comparison")
print("================")

print(
    results_df.to_string(
        index=False,
        formatters={
            "MAE": "{:.2f}".format,
            "MSE": "{:.2f}".format,
            "R2": "{:.2f}".format
        }
    )
)


# ==============================
# 10. Find Best Model
# ==============================

best_model = results_df.loc[
    results_df["MAE"].idxmin()
]

print("\n\nBest Model")
print("==========")
print("Model:", best_model["Model"])
print(f"MAE: {best_model['MAE']:.2f}")
print(f"MSE: {best_model['MSE']:.2f}")
print(f"R²: {best_model['R2']:.2f}")
