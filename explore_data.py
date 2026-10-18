import pandas as pd

# Load the student dataset
df = pd.read_csv("student/student-mat.csv", sep=";")

# Display the first 5 rows
print("First 5 rows:")
print(df.head())

# Display number of rows and columns
print("\nDataset shape:")
print(df.shape)

# Display column names
print("\nColumn names:")
print(df.columns.tolist())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())



# Check final grades
print("\nFinal grade counts:")
print(df["G3"].value_counts().sort_index())

# Check how many students passed and failed
print("\nFailed students (G3 < 10):")
print((df["G3"] < 10).sum())

print("\nPassed students (G3 >= 10):")
print((df["G3"] >= 10).sum())

# Percentage of failed and passed students
print("\nPercentage failed:")
print((df["G3"] < 10).mean() * 100)

print("\nPercentage passed:")
print((df["G3"] >= 10).mean() * 100)