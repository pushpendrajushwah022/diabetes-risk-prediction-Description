
import numpy as np
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                               confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve)

np.random.seed(42)

print("=" * 60)
print("PART 1: GENERATING PATIENT DATA")
print("=" * 60)

n = 768
age = np.random.randint(21, 81, n)
bmi = np.clip(np.round(np.random.normal(28, 6, n), 1), 15, 55)
glucose = np.random.normal(120, 30, n)
blood_pressure = np.random.normal(70, 12, n)
insulin = np.random.exponential(80, n)
pregnancies = np.random.poisson(1.5, n)
family_history = np.random.choice([0, 1], n, p=[0.7, 0.3])
physical_activity = np.random.choice(["Low", "Medium", "High"], n, p=[0.4, 0.4, 0.2])

risk_score = (
    (age > 45).astype(int) * 0.15 + (bmi > 30).astype(int) * 0.25 +
    (glucose > 140).astype(int) * 0.35 + (blood_pressure > 80).astype(int) * 0.10 +
    family_history * 0.20 + (physical_activity == "Low").astype(int) * 0.10 -
    (physical_activity == "High").astype(int) * 0.10
)
probability = 1 / (1 + np.exp(-(risk_score - 0.4) * 5))
diabetes = (np.random.random(n) < probability).astype(int)

df = pd.DataFrame({
    "age": age, "bmi": bmi, "glucose": np.round(glucose, 1),
    "blood_pressure": np.round(blood_pressure, 1), "insulin": np.round(insulin, 1),
    "pregnancies": pregnancies, "family_history": family_history,
    "physical_activity": physical_activity, "diabetes": diabetes,
})

missing_idx = np.random.choice(df.index, 40, replace=False)
df.loc[missing_idx, "insulin"] = np.nan
missing_idx2 = np.random.choice(df.index, 20, replace=False)
df.loc[missing_idx2, "blood_pressure"] = np.nan

df.to_csv("patient_data.csv", index=False)
print(f"Generated {df.shape[0]} patient records, {df.shape[1]} features")
print(f"Diabetes cases: {df['diabetes'].sum()} ({df['diabetes'].mean()*100:.1f}%)")

print("\n" + "=" * 60)
print("PART 2: BUILDING SQL DATABASE")
print("=" * 60)

conn = sqlite3.connect("hospital_records.db")
df.to_sql("patients", conn, if_exists="replace", index=True, index_label="patient_id")
conn.commit()
print(f"hospital_records.db created with {len(df)} patient records")

print("\n" + "=" * 60)
print("PART 3: LOADING DATA FROM SQL DATABASE")
print("=" * 60)

df = pd.read_sql_query("SELECT * FROM patients", conn)
df = df.drop("patient_id", axis=1)
conn.close()
print(df.head())

print("\n" + "=" * 60)
print("PART 4: DATA CLEANING")
print("=" * 60)
print("Missing values before cleaning:\n", df.isnull().sum()[df.isnull().sum() > 0])
df["insulin"] = df["insulin"].fillna(df["insulin"].median())
df["blood_pressure"] = df["blood_pressure"].fillna(df["blood_pressure"].median())
print("Missing values after cleaning:", df.isnull().sum().sum())

print("\n" + "=" * 60)
print("PART 5: FEATURE ENGINEERING")
print("=" * 60)
df["activity_encoded"] = df["physical_activity"].map({"Low": 0, "Medium": 1, "High": 2})
df["high_risk_bmi"] = (df["bmi"] > 30).astype(int)
df = df.drop("physical_activity", axis=1)
print("Final features:", list(df.drop("diabetes", axis=1).columns))

X = df.drop("diabetes", axis=1)
y = df["diabetes"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"\nPART 6: Train set: {len(X_train)} | Test set: {len(X_test)}")

print("\n" + "=" * 60)
print("PART 7: TRAINING & COMPARING MODELS")
print("=" * 60)
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42),
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    probs = model.predict_proba(X_test_scaled)[:, 1]
    print(f"\n{name}:")
    print(f"  Accuracy:  {accuracy_score(y_test, preds):.3f}")
    print(f"  Precision: {precision_score(y_test, preds):.3f}")
    print(f"  Recall:    {recall_score(y_test, preds):.3f}")
    print(f"  F1-Score:  {f1_score(y_test, preds):.3f}")
    print(f"  ROC-AUC:   {roc_auc_score(y_test, probs):.3f}")

print("\n" + "=" * 60)
print("PART 8: HYPERPARAMETER TUNING (Random Forest)")
print("=" * 60)
param_grid = {"n_estimators": [50, 100, 150], "max_depth": [5, 10, None]}
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring="roc_auc")
grid.fit(X_train_scaled, y_train)
print("Best parameters:", grid.best_params_)

best_model = grid.best_estimator_
probs = best_model.predict_proba(X_test_scaled)[:, 1]
preds = best_model.predict(X_test_scaled)
final_auc = roc_auc_score(y_test, probs)
print(f"Final tuned model ROC-AUC: {final_auc:.3f}")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

fpr, tpr, _ = roc_curve(y_test, probs)
axes[0].plot(fpr, tpr, color="darkorange", linewidth=2, label=f"ROC curve (AUC = {final_auc:.2f})")
axes[0].plot([0, 1], [0, 1], color="navy", linewidth=1, linestyle="--", label="Random guess")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curve - Diabetes Prediction")
axes[0].legend()

cm = confusion_matrix(y_test, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Diabetes", "Diabetes"])
disp.plot(ax=axes[1], cmap="Blues", values_format="d", colorbar=False)
axes[1].set_title("Confusion Matrix")

importances = pd.Series(best_model.feature_importances_, index=X.columns).sort_values()
axes[2].barh(importances.index, importances.values, color="steelblue")
axes[2].set_title("Feature Importance")

plt.tight_layout()
plt.savefig("model_results.png", dpi=100)
print("\nVisualization saved as model_results.png")
print("\n" + "=" * 60)
print("PROJECT COMPLETE - ALL STEPS FINISHED SUCCESSFULLY")
print("=" * 60)