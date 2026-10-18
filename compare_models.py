import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
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


# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42,
        class_weight="balanced"
    )
}


# 5-fold stratified cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# Compare models
for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    accuracy = cross_val_score(
        pipeline,
        X,
        y,
        cv=cv,
        scoring="accuracy"
    )

    recall = cross_val_score(
        pipeline,
        X,
        y,
        cv=cv,
        scoring="recall"
    )

    f1 = cross_val_score(
        pipeline,
        X,
        y,
        cv=cv,
        scoring="f1"
    )

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    print(f"Accuracy: {accuracy.mean():.3f}")
    print(f"At-Risk Recall: {recall.mean():.3f}")
    print(f"At-Risk F1: {f1.mean():.3f}")