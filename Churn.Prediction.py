import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# Load data
df = pd.read_csv(r"D:/N Future interns/Task-2/Churn_Modelling.csv")
print("Initial columns:", df.columns.tolist())

# Drop unneeded columns
columns_to_drop = ['RowNumber', 'Surname', 'CustomerId', 'customer_id']
df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)

# Define target and features
if 'churn' not in df.columns:
    raise ValueError("'churn' column not found in the dataset!")
y = df["churn"]
X = df.drop("churn", axis=1)

# One-hot encode
X = pd.get_dummies(X, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
y_probs = model.predict_proba(X_test)[:, 1]

# Metrics
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)
auc = roc_auc_score(y_test, y_probs)
fpr, tpr, _ = roc_curve(y_test, y_probs)

# Feature importance
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)[:10]

# Plot setup
fig = plt.figure(figsize=(16, 10))
fig.suptitle("Customer Churn Prediction Dashboard", fontsize=20, fontweight='bold')

# Confusion Matrix
ax1 = fig.add_subplot(2, 2, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
ax1.set_title("Confusion Matrix")
ax1.set_xlabel("Predicted")
ax1.set_ylabel("Actual")

# ROC Curve
ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(fpr, tpr, label=f"AUC = {auc:.2f}", color='orange')
ax2.plot([0, 1], [0, 1], linestyle='--', color='black')
ax2.set_title("ROC Curve")
ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
ax2.legend()

# Feature Importances
ax3 = fig.add_subplot(2, 2, 3)
importances.plot(kind='bar', ax=ax3, color='teal')
ax3.set_title("Top 10 Feature Importances")
ax3.set_ylabel("Importance Score")
ax3.set_xticklabels(importances.index, rotation=45)

# Summary Text
summary_text = f"""
Model Accuracy: {report['accuracy']:.2f}
Precision (Churn): {report['1']['precision']:.2f}
Recall (Churn): {report['1']['recall']:.2f}
F1-score (Churn): {report['1']['f1-score']:.2f}
ROC AUC Score: {auc:.2f}

BUSINESS INSIGHT:
Churned customers are likely older, with higher balances, and often not active members.

RECOMMENDATION:
Target retention for:
- Age > 45
- Balance > ₹100,000
- Not Active Members
"""

# Add text box outside plots
fig.text(0.53, 0.28, summary_text, fontsize=11, fontfamily='monospace', verticalalignment='top')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
