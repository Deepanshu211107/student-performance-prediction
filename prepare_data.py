import pandas as pd

# Load dataset
df = pd.read_csv("student/student-mat.csv", sep=";")

# Create target
# 1 = At Risk, 0 = Not At Risk
df["performance"] = (df["G3"] < 10).astype(int)

# Features
X = df.drop(columns=["G3", "G1", "G2", "performance"])

# Target
y = df["performance"]

print("Features:")
print(X.columns.tolist())

print("\nNumber of features:", X.shape[1])

print("\nTarget distribution:")
print(y.value_counts())

print("\nTarget percentages:")
print(y.value_counts(normalize=True) * 100)