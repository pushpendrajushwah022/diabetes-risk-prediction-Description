# Diabetes Risk Prediction System

A complete machine learning pipeline that predicts diabetes risk from patient health data, comparing multiple classification models and identifying the key clinical risk factors.

## Problem Statement

Early identification of diabetes risk allows for timely intervention. This project builds a classification system that predicts diabetes likelihood using routine clinical measurements (glucose, BMI, blood pressure, etc.), helping prioritize patients for further screening.

## Dataset

768 patient records with the following features:
- **Demographics:** age, pregnancies
- **Clinical measurements:** BMI, glucose level, blood pressure, insulin
- **Risk factors:** family history of diabetes, physical activity level
- **Target:** diabetes diagnosis (0 = No, 1 = Yes)

The dataset includes realistic missing values (as real clinical data does), which are handled during preprocessing.

## Approach

1. **Data Generation** — Created a realistic synthetic patient dataset (in a real deployment, this would be actual hospital records)
2. **SQL Database** — Loaded patient records into a SQL database, simulating a real hospital records system
3. **SQL Exploratory Analysis** — Ran SQL queries directly against the database to understand risk patterns before any modeling, including:
   - Diabetes rate by age group
   - Average clinical measurements: diabetic vs non-diabetic patients
   - Window functions (`RANK() OVER PARTITION BY`) to rank patients by risk within categories
   - Diabetes rate by family history
4. **Data Cleaning** — Handled missing values in `insulin` and `blood_pressure` using median imputation
5. **Feature Engineering** — Encoded categorical activity levels; created a `high_risk_bmi` flag (BMI > 30)
6. **Model Comparison** — Trained and compared three classifiers: Logistic Regression, Random Forest, and SVM
7. **Hyperparameter Tuning** — Used `GridSearchCV` (5-fold cross-validation) to optimize the Random Forest, searching across `n_estimators` and `max_depth`
8. **Evaluation** — Assessed models using Accuracy, Precision, Recall, F1-Score, and ROC-AUC (not accuracy alone, since false negatives in a medical context carry real cost)

## SQL Analysis Highlights

- Diabetes rate increases with age: **36.5%** (under 35) → **48.3%** (35-55) → **57.6%** (over 55)
- Diabetic patients show meaningfully higher average glucose (129.7 vs 113.9) and BMI (29.4 vs 27.1)
- Patients with a family history of diabetes show a notably higher diabetes rate (61.3% vs 44.8%)
- Used `RANK() OVER (PARTITION BY ...)` to identify highest-risk patients within each activity level group

## Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.753 | 0.760 | 0.740 | 0.750 | 0.795 |
| Random Forest | 0.747 | 0.757 | 0.727 | 0.742 | 0.810 |
| SVM | 0.740 | 0.747 | 0.727 | 0.737 | 0.803 |
| **Random Forest (tuned)** | — | — | — | — | **0.809** |

**Key finding:** Glucose level was the single strongest predictor of diabetes risk, followed by BMI and age — consistent with established clinical understanding of diabetes risk factors.

## Files

- `diabetes_complete.py` — Full pipeline in a single file: generates data, builds a SQL database, loads from SQL, cleans, engineers features, trains and compares models, tunes hyperparameters, evaluates, and visualizes results
- `patient_data.csv` — The raw dataset
- `hospital_records.db` — SQLite database containing the patient records
- `model_results.png` — ROC curve, confusion matrix, and feature importance visualizations

## How to Run

```bash
pip install pandas numpy scikit-learn matplotlib
python diabetes_complete.py
```

## Tech Stack

Python, SQL (SQLite / SQL Server), Pandas, NumPy, Scikit-learn, Matplotlib

## Future Improvements

- Test additional models (XGBoost, Neural Networks)
- Deploy as a Flask API for real-time risk scoring
- Add SHAP values for per-patient explainability
