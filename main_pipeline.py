# main_pipeline.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.special import softmax 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE

import shap
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv("bank-full.csv", sep=';')

# Display basic info
print("Dataset Info:")
print(df.info())
print("\nSample records:")
print(df.head())

# Target variable plot
print("\nTarget class distribution (raw):")
print(df['y'].value_counts())
sns.countplot(x='y', data=df)
plt.title("Target Variable Distribution")
plt.xlabel("Subscription (y)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Encode target
df['y'] = df['y'].map({'yes': 1, 'no': 0})

# Encode categorical features
categorical_features = ['job', 'marital', 'education', 'default',
                        'housing', 'loan', 'contact', 'month', 'poutcome']

le = LabelEncoder()
for col in categorical_features:
    df[col] = le.fit_transform(df[col])

# --- EDA Visualizations ---

# 1. Correlation Heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("Feature_Correlation_Heatmap.png")

# 2. Age Distribution by Subscription
plt.figure(figsize=(10,6))
sns.histplot(data=df, x='age', hue='y', multiple='stack', bins=30, palette='viridis')
plt.title("Age Distribution by Subscription Status")
plt.xlabel("Age")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("Age_Distribution_by_Subscription.png")

# 3. Balance Distribution by Subscription
plt.figure(figsize=(10,6))
sns.kdeplot(data=df, x='balance', hue='y', fill=True, palette='magma')
plt.title("Balance Distribution by Subscription Status")
plt.xlabel("Balance")
plt.ylabel("Density")
plt.tight_layout()
plt.savefig("Balance_Distribution_by_Subscription.png")

# 4. Subscription Rate by Education
plt.figure(figsize=(8,6))
sns.barplot(data=df, x='education', y='y', palette='viridis')
plt.title("Subscription Rate by Education Level")
plt.xlabel("Education")
plt.ylabel("Subscription Rate")
plt.tight_layout()
plt.savefig("Subscription_Rate_by_Education.png") 


# Split data
X = df.drop('y', axis=1)
y = df['y']

# Check for class imbalance
print("\nEncoded Target Distribution:")
print(y.value_counts(normalize=True))

imbalance_ratio = y.value_counts(normalize=True)
if imbalance_ratio.min() < 0.4:
    print("\n⚠️ Imbalance Detected — Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X, y = smote.fit_resample(X, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# ---- Models ----

# Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# XGBoost
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

# CatBoost
cat = CatBoostClassifier(verbose=0)
cat.fit(X_train, y_train)
y_pred_cat = cat.predict(X_test)

# Evaluation
def evaluate_model(name, y_true, y_pred):
    print(f"\n{name} Performance:")
    print(classification_report(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

evaluate_model("Random Forest", y_test, y_pred_rf)
evaluate_model("XGBoost", y_test, y_pred_xgb)
evaluate_model("CatBoost", y_test, y_pred_cat)


'''
# Select best model — CatBoost assumed best here
best_model = rf
joblib.dump(best_model, "best_model.pkl")
'''


print("\nComputing SHAP values for Class 1 prediction...")

# Sample test data for faster computation
X_sample = X_test.sample(n=300, random_state=42)

# Create explainer for tree model
explainer = shap.TreeExplainer(rf)  

# Get SHAP values for all classes — list of arrays
shap_values_all = explainer.shap_values(X_sample)

# Use only class 1 SHAP values
shap_values_class1 = shap_values_all[1]

# Manually compute mean absolute SHAP values
def print_feature_importances_shap_values(shap_values, features):
    importances = []
    for i in range(shap_values.shape[1]):
        importances.append(np.mean(np.abs(shap_values[:, i])))

    # Softmax for normalized interpretation
    importances_norm = softmax(importances)

    # Zip into dictionaries
    feature_importances = {fea: imp for imp, fea in zip(importances, features)}
    feature_importances_norm = {fea: imp for imp, fea in zip(importances_norm, features)}

    # Sort descending
    feature_importances = dict(sorted(feature_importances.items(), key=lambda item: item[1], reverse=True))
    feature_importances_norm = dict(sorted(feature_importances_norm.items(), key=lambda item: item[1], reverse=True))

    # Print top 10
    print("\nTop Feature Importances for Class 1:")
    for k, v in list(feature_importances.items())[:10]:
        print(f"{k} -> {v:.4f} (softmax = {feature_importances_norm[k]:.4f})")

    return feature_importances

# Get and print sorted importances
importances = print_feature_importances_shap_values(shap_values_class1, X_sample.columns)

# Plot top 10 features as bar chart
top_n = 10
top_features = list(importances.keys())[:top_n]
top_importance_vals = list(importances.values())[:top_n]

plt.figure(figsize=(10, 6))
plt.barh(top_features[::-1], top_importance_vals[::-1])
plt.xlabel("Mean |SHAP value| (impact on class 1 prediction)")
plt.title("Top Feature Importances for Predicting Product Adoption (Class 1)")
plt.tight_layout()
plt.show()


# Save predictions
X_test['predicted'] = rf.predict(X_test)
X_test['actual'] = y_test.values
X_test.to_csv("predictions.csv", index=False)

print("\n✅ Pipeline completed and outputs saved.")
