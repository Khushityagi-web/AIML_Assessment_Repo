import pandas as pd
# Load dataset
data = pd.read_csv('processed_gse20685_data.csv')

# Check shape & columns
print(data.shape)
print(data.columns[:20])  # Show first 20 columns
data.head()
# Identify clinical columns (the last few columns)
clinical_cols = ['subtype','t_stage','n_stage','m_stage','regional_relapse',
                 'adjuvant_chemotherapy','regimen (caf vs cmf)',
                 'time_to_relapse (years)','time_to_metastasis (years)',
                 'neoadjuvant chemotherapy']
# Features (gene expression only)
X = data.drop(columns=['Unnamed: 0'] + clinical_cols)

# Label: relapse (binary)
y = data['regional_relapse']

print("Features shape:", X.shape)
print("Label distribution:\n", y.value_counts())
print(y.isna().sum())
y.value_counts(dropna=False)
# Drop rows with NaN target
data_clean = data.dropna(subset=['regional_relapse'])
# Features and target
X = data_clean.drop(columns=['Unnamed: 0'] + clinical_cols)
y = data_clean['regional_relapse']

# Train-test split (stratified to maintain class ratio)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train:", X_train.shape, "Test:", X_test.shape)
print("Train label distribution:\n", y_train.value_counts())
non_numeric_cols = X.select_dtypes(exclude=['number']).columns
print(non_numeric_cols)
# Keep only numeric columns (drop tissue & gender)
X = X.select_dtypes(include=['number'])
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature selection: top 1000 most variable genes
variances = X_train.var(axis=0)
top_genes = variances.sort_values(ascending=False).head(1000).index
X_train = X_train[top_genes]
X_test = X_test[top_genes]

# Normalize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
# Logistic Regression (with class weights for imbalance)
lr = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
y_prob_lr = lr.predict_proba(X_test)[:,1]
# Random Forest
rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:,1]
print("Logistic Regression:")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("AUC:", roc_auc_score(y_test, y_prob_lr))
print(classification_report(y_test, y_pred_lr))
print("\nRandom Forest:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("AUC:", roc_auc_score(y_test, y_prob_rf))
print(classification_report(y_test, y_pred_rf))
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay
# ROC Curves
RocCurveDisplay.from_predictions(y_test, y_prob_lr, name="Logistic Regression")
RocCurveDisplay.from_predictions(y_test, y_prob_rf, name="Random Forest")
plt.plot([0,1], [0,1], 'k--')  # diagonal line
plt.title("ROC Curves")
plt.show()
# Confusion Matrices
ConfusionMatrixDisplay.from_estimator(lr, X_test, y_test)
plt.title("Logistic Regression Confusion Matrix")
plt.show()
ConfusionMatrixDisplay.from_estimator(rf, X_test, y_test)
plt.title("Random Forest Confusion Matrix")
plt.show()
# Feature Importance (Random Forest)
import numpy as np
importances = rf.feature_importances_
indices = np.argsort(importances)[-15:]  # top 15
plt.figure(figsize=(8,6))
plt.barh(range(len(indices)), importances[indices])
plt.yticks(range(len(indices)), [top_genes[i] for i in indices])
plt.title("Top 15 Features - Random Forest")
plt.show()
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

print("Before SMOTE:\n", y_train.value_counts())
print("\nAfter SMOTE:\n", y_train_res.value_counts())
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

xgb = XGBClassifier(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=(y_train.value_counts()[0] / y_train.value_counts()[1]),
    random_state=42
)

xgb.fit(X_train_res, y_train_res)
y_pred_xgb = xgb.predict(X_test)
y_prob_xgb = xgb.predict_proba(X_test)[:,1]

print("XGBoost:")
print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("AUC:", roc_auc_score(y_test, y_prob_xgb))
print(classification_report(y_test, y_pred_xgb))
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay

RocCurveDisplay.from_predictions(y_test, y_prob_xgb, name="XGBoost")
plt.plot([0,1], [0,1], 'k--')  # diagonal
plt.title("ROC Curve - XGBoost")
plt.show()
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_estimator(xgb, X_test, y_test)
plt.title("Confusion Matrix - XGBoost")
plt.show()
import numpy as np

importances = xgb.feature_importances_
indices = np.argsort(importances)[-15:]  # top 15
plt.figure(figsize=(8,6))
plt.barh(range(len(indices)), importances[indices])
plt.yticks(range(len(indices)), [top_genes[i] for i in indices])
plt.title("Top 15 Features - XGBoost")
plt.show()
