import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


# Load dataset
df = pd.read_csv("student/student-mat.csv", sep=";")


# Create target
# 1 = At Risk
# 0 = Not At Risk
df["performance"] = (df["G3"] < 10).astype(int)


# Remove target and G1/G2 from features
X = df.drop(columns=["G3", "G1", "G2", "performance"])
y = df["performance"]


# Find categorical and numerical columns
categorical_columns = X.select_dtypes(include=["object"]).columns
numerical_columns = X.select_dtypes(exclude=["object"]).columns


print("Categorical columns:")
print(categorical_columns.tolist())

print("\nNumerical columns:")
print(numerical_columns.tolist())


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
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
        ("numerical", "passthrough", numerical_columns)
    ]
)


# Fit preprocessing only on training data
X_train_processed = preprocessor.fit_transform(X_train)

# Transform test data using the same fitted preprocessing
X_test_processed = preprocessor.transform(X_test)


print("\nTraining data shape:")
print(X_train_processed.shape)

print("\nTesting data shape:")
print(X_test_processed.shape)