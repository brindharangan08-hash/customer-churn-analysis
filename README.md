# 🧠 Customer Churn Prediction 

This repository contains a complete, intermediate-level data analysis project focused on **predicting customer churn**. It includes a dataset, a Python analysis script, generated charts, and a ready-to-post LinkedIn caption.

---

## 📂 Files in this Repo
- `customer_churn.csv` — dataset (30 customers, engineered churn signal)
- `churn_analysis.py` — full EDA + Logistic Regression model + charts (PNG)
- `churn_rate_by_gender.png`, `age_boxplot.png`, `correlation_heatmap.png`, `roc_curve.png` — output figures
- `model_report.txt` — classification report & confusion matrix

---

## 🛠 Tools & Skills
- **Python**: pandas, numpy, matplotlib, scikit-learn
- **Concepts**: EDA, feature engineering, train/test split, scaling, **Logistic Regression**, ROC/AUC
- **Visualization**: bar chart, boxplot, correlation heatmap, ROC curve

---

## 🚀 How to Run
```bash
# 1) Clone
git clone https://github.com/your-username/customer-churn-analysis.git
cd customer-churn-analysis

# 2) (Optional) Create a virtual environment
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate

# 3) Install dependencies
pip install -r requirements.txt
# or
pip install pandas numpy matplotlib scikit-learn

# 4) Run the analysis
python churn_analysis.py
```

> The script prints evaluation metrics and saves charts to the repository folder.

---

## 🧪 What You’ll Learn
- Calculate and visualize **churn rates**
- Explore relationships between **tenure, credit score, products** and churn
- Build a **baseline classifier** for churn using Logistic Regression
- Interpret **ROC/AUC**, a confusion matrix, and a classification report

---

## 📊 Example Insights
- Customers with **low tenure** and **lower credit scores** are more likely to churn.
- Churners often have **fewer products**.
- The **ROC-AUC** summarizes the trade-off between TPR/FPR; higher is better.

---

## 📦 Suggested `requirements.txt`
```
pandas
numpy
matplotlib
scikit-learn
```

---



