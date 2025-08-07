# 💼 Precision Banking: ML-Based Product Adoption Forecasting

> *An ML-driven system to predict customer likelihood of subscribing to a telemarketing bank product — built using the Portuguese Bank Marketing Dataset.*
(Data-driven approach to predict the success of Bank Telemarketing)
---

## 🧠 Introduction

In today’s competitive financial landscape, banks need more than intuition to market products effectively. This project harnesses machine learning to predict whether a client will subscribe to a bank product — enabling personalized marketing strategies and improved conversion rates.

This solution was developed using the **Portuguese Bank Marketing Dataset** from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing), offering real-world insights based on customer interaction data.

---

## 🎯 Project Objective

Develop and deploy a machine learning classification model to:
- Predict whether a client will purchase a marketed bank product.
- Empower financial institutions to run more efficient, targeted campaigns.
- Discover the most influential features that impact client conversion.

---

## 📦 Dataset Overview

- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
- **Rows**: 45,200  
- **Features**: 17 client-related variables (age, job, education, marital status, contact method, etc.)
- **Target Variable**: Whether or not the client subscribed to the product (`yes`/`no`)

---

## 🔧 Tools & Technologies Used

| Tool | Role |
|------|------|
| `Python` | Programming language |
| `Pandas`, `NumPy` | Data wrangling and transformation |
| `Matplotlib`, `Seaborn` | Exploratory data analysis and visualization |
| `Scikit-learn` | Model training, evaluation, and tuning |
| `Random Forest`, `XGBoost`, `CatBoost`  | ML models used |
| `Streamlit` | Lightweight app deployment and interaction |
| `Joblib` | Model persistence |
| `FastAPI` | Model Serving API |
| `Docker` | Environment Isolation Containerization |
| `MLFlow` | Experiment Tracking |
| `CI/CD Pipeline` | Automated Deployment Workflow |
---



### 🌳 Final Models & Selection Process

Multiple tree-based ensemble models were evaluated to maximize predictive accuracy and robustness:

| Model             | Key Strengths                                         | F1 Score |
|------------------|-------------------------------------------------------|----------|
| **Random Forest** | Robust to noise, highly interpretable, strong on categorical data | **0.9327** ✅ |
| **XGBoost**       | Excellent with imbalanced data, built-in regularization | 0.9274     |
| **CatBoost**      | Natively handles categorical variables, minimal tuning required | 0.9271     |

✅ **Random Forest** was chosen for deployment due to:
- Highest F1 score (after class rebalancing with **SMOTE**)
- Smooth integration with **SHAP** for explainability
- Compatibility with deployment tools like **FastAPI** and **Streamlit**

---

### 🧪 Evaluation Strategy
- **F1 Score** was selected to address class imbalance in the target variable
- **Confusion matrix** showed balanced performance with minimized false positives/negatives
- Comparative metrics across models ensured reliability and fairness

---

### 🔬 Explainability with SHAP
- SHAP values analyzed the **impact of each feature** on Class 1 (product adoption)
- **Bar plots** and **beeswarm plots** were used for intuitive visualization
- A sample-based SHAP summary was used to optimize speed and interpretability

> 🔍 **Top Predictive Drivers of Product Adoption**:
> - Education level
> - Job category
