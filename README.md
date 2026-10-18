# Explainable AI-Based Student Performance Prediction System

## 1. Project Overview

The Explainable AI-Based Student Performance Prediction System is a
machine learning web application that predicts a student's final
academic score on a scale of 0 to 20.

The system uses student demographic, family, social, and academic-related
information to estimate the final grade (G3).

In addition to providing a prediction, the system uses SHAP
(SHapley Additive exPlanations) to explain which features contributed
most strongly to the predicted score.

---

## 2. Objectives

The main objectives of this project are:

- Predict a student's final academic score.
- Use machine learning for student performance prediction.
- Compare different regression algorithms.
- Select the best-performing model.
- Provide understandable explanations for individual predictions.
- Demonstrate the use of Explainable AI using SHAP.

---

## 3. Dataset

The project uses the UCI Student Performance dataset.

The dataset contains information about students from secondary schools
in Portugal.

The target variable used in this project is:

- G3 - Final grade

The final grade ranges from 0 to 20.

G1 and G2 are excluded from the prediction model because they are
earlier-period grades and could make the prediction less useful as an
early performance estimation.

The features related to romantic relationships and alcohol consumption
were also excluded from the final model.

---

## 4. Technologies Used

### Programming Language

- Python

### Libraries

- Pandas
- NumPy
- Scikit-learn
- Joblib
- SHAP

### Web Technologies

- HTML
- CSS
- JavaScript

### Web Framework

- Flask

---

## 5. Machine Learning Approach

The project follows these main steps:

1. Load the student dataset.
2. Separate input features and target variable.
3. Remove excluded features.
4. Identify numerical and categorical features.
5. Encode categorical features using One-Hot Encoding.
6. Split the dataset into training and testing sets.
7. Train multiple regression models.
8. Compare their performance.
9. Select the best model.
10. Generate predictions.
11. Explain individual predictions using SHAP.

---

## 6. Models Compared

Four regression models were evaluated:

### Linear Regression

A basic regression model used as a baseline.

### Decision Tree Regressor

A tree-based model that makes predictions using decision rules.

### Random Forest Regressor

An ensemble of multiple decision trees.

### Gradient Boosting Regressor

An ensemble method that builds models sequentially to improve
prediction performance.

---

## 7. Model Comparison

The models were evaluated using MAE, MSE, and R².

| Model | MAE | MSE | R² |
|---|---:|---:|---:|
| Linear Regression | 3.51 | 18.19 | 0.11 |
| Decision Tree | 3.45 | 19.67 | 0.04 |
| Random Forest | 3.09 | 14.63 | 0.29 |
| Gradient Boosting | 3.05 | 14.64 | 0.29 |

Gradient Boosting was selected as the final model because it achieved
the lowest MAE among the tested models.

---

## 8. Final Model

The final model used by the application is:

**Gradient Boosting Regressor**

Performance:

- MAE: 3.05
- MSE: 14.64
- R²: 0.29

The model predicts the student's final score on a scale of 0 to 20.

---

## 9. Explainable AI

SHAP (SHapley Additive exPlanations) is used to explain individual
predictions.

For each student, the system displays the features that had the
strongest influence on the predicted score.

A positive SHAP contribution indicates that the feature pushed the
model's prediction higher.

A negative SHAP contribution indicates that the feature pushed the
model's prediction lower.

SHAP explanations describe the model's learned behavior and should not
be interpreted as proof of causation.

---

## 10. Web Application

The web application contains two main pages.

### Home Page

The user enters student information through a form.

The form includes information such as:

- School type
- Gender
- Age
- Address
- Family size
- Parent status
- Parents' education
- Parents' jobs
- Reason for choosing school
- Guardian
- Travel time
- Study time
- Previous failures
- School support
- Family support
- Paid classes
- Extracurricular activities
- Nursery education
- Higher education intention
- Internet access
- Family relationship quality
- Free time
- Going out
- Health
- Absences

### Result Page

The result page displays:

- Predicted final score
- Performance level
- Score progress bar
- SHAP explanations
- Model performance
- Model comparison

---

## 11. Example Predictions

The system was tested using three different student profiles.

| Test | Profile | Predicted Score | Performance |
|---|---|---:|---|
| 1 | Strong Profile | 12.62 / 20 | Average |
| 2 | Average Profile | 10.55 / 20 | Average |
| 3 | Weak Profile | 7.00 / 20 | Below Passing |

These tests demonstrate that the application can produce different
predictions and explanations for different student profiles.

---

## 12. Project Structure

```text
AI Project/
│
├── student/
│   └── student-mat.csv
│
├── templates/
│   ├── index.html
│   └── result.html
│
├── app.py
├── model.py
├── compare_regression_models.py
├── predict.py
├── predict_explain.py
│
├── gradient_boosting_regressor.pkl
├── preprocessor.pkl
│
└── README.md



13. How to Run the Project


Step 1: Install required libraries
pip install pandas numpy scikit-learn flask joblib shap
Step 2: Train the model
python model.py
Step 3: Start the Flask application
python app.py
Step 4: Open the application

Open the local Flask address shown in the terminal, usually:

http://127.0.0.1:5000



14. Limitations

The model has an R² score of 0.29, which means that a substantial
amount of variation in final grades is not explained by the features
used by the model.

The MAE of 3.05 means that the model's predictions differ from the
actual final scores by about 3 points on average on the test set.

Therefore, the system should be considered a predictive aid rather
than an exact assessment of student performance.

The dataset is also limited to the students represented in the
original dataset, so the model may not generalize equally well to
students from different populations or educational systems.

SHAP explanations describe model behavior and do not establish
causal relationships.




15. Future Scope

Possible future improvements include:

Hyperparameter tuning.
Cross-validation.
Testing additional machine learning algorithms.
Increasing the size and diversity of the dataset.
Adding more relevant academic features.
Improving prediction accuracy.
Adding prediction confidence or uncertainty estimates.
Creating interactive visualizations of student performance.
Deploying the application online.




16. Conclusion

The project demonstrates how machine learning and Explainable AI can
be combined to create an understandable student performance prediction
system.

The Gradient Boosting Regressor was selected after comparing four
regression models.

The system predicts a student's final score and uses SHAP to provide
an explanation of the factors that influenced each prediction.

This combination of prediction and explainability makes the system
more transparent than a prediction-only machine learning application.
