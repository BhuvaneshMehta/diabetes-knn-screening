"""
DAAI SS26 - Diabetes screening GUI
Loads the saved KNN model + scaler and predicts from eight inputs.
Run:  python gui_app.py   (after train_model.py has been run once)
"""

import tkinter as tk
from tkinter import messagebox
import numpy as np
import joblib

model = joblib.load("knn_model.joblib")
scaler = joblib.load("scaler.joblib")

FIELDS = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
          "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

root = tk.Tk()
root.title("Diabetes Risk Screening (KNN)")

entries = {}
for i, name in enumerate(FIELDS):
    tk.Label(root, text=name).grid(row=i, column=0, sticky="w", padx=8, pady=3)
    e = tk.Entry(root, width=12)
    e.grid(row=i, column=1, padx=8, pady=3)
    entries[name] = e

result_var = tk.StringVar(value="Enter values and press Predict")
tk.Label(root, textvariable=result_var, font=("Arial", 11, "bold"),
         wraplength=280).grid(row=len(FIELDS) + 1, column=0, columnspan=2, pady=8)


def predict():
    try:
        values = [float(entries[f].get()) for f in FIELDS]
    except ValueError:
        messagebox.showerror("Input error", "Please fill all eight fields with numbers.")
        return
    x = scaler.transform(np.array(values).reshape(1, -1))
    pred = model.predict(x)[0]
    vote = model.predict_proba(x)[0][1]  # share of k neighbours voting positive
    label = "AT RISK of diabetes" if pred == 1 else "Not at risk"
    result_var.set(f"{label}\n{vote:.0%} of the {model.n_neighbors} nearest "
                   f"neighbours voted positive")


tk.Button(root, text="Predict", command=predict,
          width=15).grid(row=len(FIELDS), column=0, columnspan=2, pady=8)

root.mainloop()
