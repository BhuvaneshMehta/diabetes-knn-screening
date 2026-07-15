"""
DAAI SS26 - Diabetes risk screening with KNN
Trains, tunes and evaluates the model, then saves model + scaler.
Run:  python train_model.py
Needs diabetes.csv in the same folder.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

# ---------- 1. Load ----------
df = pd.read_csv("diabetes.csv")
print("Shape:", df.shape)
print(df.head())

# ---------- 2. Clean: zeros in these columns mean 'not recorded' ----------
zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in zero_cols:
    df[col] = df[col].replace(0, np.nan)
    df[col] = df[col].fillna(df[col].median())

X = df.drop(columns="Outcome")
y = df["Outcome"]

# ---------- 3. Split 80/20, stratified ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# ---------- 4. Scale ----------
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

print(f"\nTraining rows: {len(X_train)}  ->  sqrt(N) = {np.sqrt(len(X_train)):.1f}")

# ---------- 5. Error-rate plot for odd k from 1 to 31 ----------
ks = range(1, 32, 2)
errors = []
for k in ks:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_s, y_train, cv=5, scoring="accuracy")
    errors.append(1 - scores.mean())

plt.figure(figsize=(8, 5))
plt.plot(list(ks), errors, marker="o")
plt.xlabel("k")
plt.ylabel("5-fold CV error rate")
plt.title("Error rate vs k")
plt.grid(True)
plt.savefig("error_rate_plot.png", dpi=150)
print("Saved error_rate_plot.png")
best_k_from_plot = list(ks)[int(np.argmin(errors))]
print("Lowest CV error at k =", best_k_from_plot)

# ---------- 6. Grid search: k and distance metric together ----------
grid = GridSearchCV(
    KNeighborsClassifier(),
    param_grid={"n_neighbors": list(range(3, 32, 2)),
                "metric": ["euclidean", "manhattan"]},
    cv=5, scoring="accuracy")
grid.fit(X_train_s, y_train)
print("Grid search best params:", grid.best_params_)

# ---------- 7. Final model (k = 21 in the report; adjust if your plot says otherwise) ----------
K = 21
model = KNeighborsClassifier(n_neighbors=K, metric="euclidean")
model.fit(X_train_s, y_train)

# ---------- 8. Evaluate on the held-out test set ----------
y_prob = model.predict_proba(X_test_s)[:, 1]
y_pred = model.predict(X_test_s)

print(f"\n--- Test-set results (k={K}, threshold 0.5) — use these for Table 2 ---")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.2f}")
print(f"Precision: {precision_score(y_test, y_pred):.2f}")
print(f"Recall   : {recall_score(y_test, y_pred):.2f}")
print(f"F1-score : {f1_score(y_test, y_pred):.2f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, y_prob):.2f}")
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# ---------- 9. Lowered threshold to favour recall ----------
THRESHOLD = 0.4
y_pred_low = (y_prob >= THRESHOLD).astype(int)
print(f"\n--- With lowered threshold {THRESHOLD} (recall-first, as in the report) ---")
print(f"Precision: {precision_score(y_test, y_pred_low):.2f}")
print(f"Recall   : {recall_score(y_test, y_pred_low):.2f}")

# ---------- 10. Save model + scaler ----------
joblib.dump(model, "knn_model.joblib")
joblib.dump(scaler, "scaler.joblib")
print("\nSaved knn_model.joblib and scaler.joblib")
