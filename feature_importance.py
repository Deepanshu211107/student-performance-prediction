import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier


# Load dataset
df = pd.read_csv("student/student-mat.csv", sep=";")


# Create target
# 1 = At Risk
# 0 = Not At Risk
df["performance"] = (df["G3"] < 10).astype(int)


# Features and target
X = df.drop(columns=["G3", "G1", "G2", "performance"])
y = df["performance"]


# Identify column types
categorical_columns = X.select_dtypes(include=["object"]).columns
numerical_columns = X.select_dtypes(exclude=["object"]).columns


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Preprocessing
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


# Transform training data
X_train_processed = preprocessor.fit_transform(X_train)


# Create Random Forest
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    random_state=42,
    class_weight="balanced"
)


# Train
model.fit(X_train_processed, y_train)


# Get feature names after one-hot encoding
feature_names = preprocessor.get_feature_names_out()


# Get feature importance
importances = model.feature_importances_


# Create a DataFrame
importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
})


# Sort from highest to lowest
importance_df = importance_df.sort_values(
    by="importance",
    ascending=False
)


# Display top 15
print("\nTop 15 features:")
print(importance_df.head(15).to_string(index=False))