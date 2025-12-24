# ==========================================
# TASK 4: Responsible AI & Model Interpretation
# ==========================================

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

print("Task 4 started...\n")

# -------- 1. Create Dataset --------
# Added 'gender' as sensitive attribute for bias analysis

data = {
    "studytime": [3,1,4,2,3,1,4,2,3,1],
    "absences":  [2,10,1,6,3,12,0,7,4,15],
    "failures":  [0,1,0,1,0,2,0,1,0,2],
    "gender":    ["F","M","F","M","F","M","F","M","F","M"],  # Sensitive attribute
    "Result":    [1,0,1,0,1,0,1,0,1,0]   # Pass=1, Fail=0
}

df = pd.DataFrame(data)
print(df)

# -------- 2. Encode Gender --------
df["gender_encoded"] = df["gender"].map({"M":0, "F":1})

X = df[["studytime", "absences", "failures", "gender_encoded"]]
y = df["Result"]

# -------- 3. Train-Test Split --------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# -------- 4. Train Model --------
model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))

# -------- 5. SHAP Explainability --------
explainer = shap.LinearExplainer(model, X_train)
shap_values = explainer.shap_values(X_test)

print("\nDisplaying SHAP Summary Plot...")
shap.summary_plot(shap_values, X_test, show=True)

# -------- 6. Local Explanation for One Prediction --------
print("\nDisplaying SHAP explanation for one prediction...")
shap.force_plot(
    explainer.expected_value,
    shap_values[0],
    X_test.iloc[0],
    matplotlib=True
)

# -------- 7. Bias Analysis --------
# Compare pass rates by gender

results_df = X_test.copy()
results_df["Actual"] = y_test.values
results_df["Predicted"] = y_pred

gender_bias = results_df.groupby("gender_encoded")["Predicted"].mean()

print("\nBias Analysis (Predicted Pass Rate):")
print("Male (0):", gender_bias[0])
print("Female (1):", gender_bias[1])

# -------- 8. Bias Interpretation --------
if abs(gender_bias[0] - gender_bias[1]) > 0.2:
    print("\n⚠️ Potential bias detected between genders")
else:
    print("\n✅ No strong bias detected between genders")

print("\nTask 4 completed successfully.")
