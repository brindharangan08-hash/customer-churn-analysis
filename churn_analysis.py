import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix

# -------- Load Data --------
df = pd.read_csv("customer_churn.csv")

# Basic checks
print("Shape:", df.shape)
print("Columns:", list(df.columns))
print("Nulls per column:\n", df.isna().sum())
print("\nSample:\n", df.head())

# -------- Feature Engineering --------
df_model = df.copy()
df_model["Gender"] = df_model["Gender"].map({"Male": 0, "Female": 1})

features = ["Gender", "Age", "Tenure", "Balance", "Products", "CreditScore", "EstimatedSalary"]
X = df_model[features].values
y = df_model["Exited"].values

# -------- Train/Test Split --------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7, stratify=y)

# -------- Scale --------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------- Model --------
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_scaled, y_train)

y_pred = clf.predict(X_test_scaled)
y_prob = clf.predict_proba(X_test_scaled)[:, 1]

print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# -------- Visualizations --------
# 1) Churn rate by gender
churn_by_gender = df.groupby("Gender")["Exited"].mean().sort_values(ascending=False)
plt.figure()
churn_by_gender.plot(kind="bar")
plt.title("Churn Rate by Gender")
plt.xlabel("Gender")
plt.ylabel("Churn Rate")
plt.tight_layout()
plt.savefig("churn_rate_by_gender.png")
plt.close()

# 2) Age distribution by churn (boxplot)
plt.figure()
data_box = [df.loc[df["Exited"] == 0, "Age"], df.loc[df["Exited"] == 1, "Age"]]
plt.boxplot(data_box, labels=["Retained (0)", "Churned (1)"])
plt.title("Age Distribution by Churn Status")
plt.ylabel("Age")
plt.tight_layout()
plt.savefig("age_boxplot.png")
plt.close()

# 3) Correlation heatmap
df_corr = df_model[features + ["Exited"]].corr()
plt.figure()
plt.imshow(df_corr, interpolation="nearest")
plt.title("Correlation Heatmap")
plt.xticks(range(len(df_corr.columns)), df_corr.columns, rotation=45, ha="right")
plt.yticks(range(len(df_corr.index)), df_corr.index)
plt.colorbar()
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.close()

# 4) ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.title("ROC Curve - Logistic Regression")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.close()

print("Saved figures: churn_rate_by_gender.png, age_boxplot.png, correlation_heatmap.png, roc_curve.png")
