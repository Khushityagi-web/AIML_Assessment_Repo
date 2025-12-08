# Breast Cancer Relapse Prediction — GSE20685

Machine Learning + GEO Data Engineering + Clinical Metadata Extraction

This repository contains a complete workflow for processing the GEO dataset GSE20685 and training machine learning models to predict regional relapse in breast cancer patients.
The project demonstrates:

🔹 complex GEO metadata parsing

🔹 high-dimensional gene expression handling

🔹 clinical variable extraction

🔹 feature engineering

🔹 class imbalance methods

🔹 model comparison (LR, RF, XGBoost)

🔹 performance visualization

## 📌 1. Transcriptomic + Clinical Data Processing

process_gse20685.py

This script performs full reconstruction of the dataset from the raw GEO series matrix:

### GEO series matrix parsing

🔹 Identify "!series_matrix_table_begin" and "end" markers

🔹 Extract gene expression block dynamically

🔹 Remove trailing metadata rows

🔹 Transpose matrix so samples = rows, genes = columns

### Clinical metadata extraction

GEO stores metadata in complex lines like:

🔹 !Sample_characteristics_ch1 = subtype: Luminal A
🔹 !Sample_characteristics_ch1 = relapse: 0


### The script:

🔹 iterates through all !Sample_characteristics_ch1 fields

🔹 handles missing or unlabeled characteristics

🔹 splits key: value pairs

🔹 ensures clinical columns are unique and consistent

🔹 maps GEO accession IDs → sample titles

🔹 merges expression + clinical metadata

🔹 performs QC checks for missingness, inconsistent labels, and sample mismatches

🔹 The final merged dataset is saved as:

processed_gse20685_data.csv
gse20685_clinical_data.csv


This file becomes the input for machine learning.

### 📌 2. Machine Learning Pipeline

breast_cancer_relapse_prediction.py

The ML workflow includes:

### Data Cleaning

🔹 Remove non-numeric columns

🔹 Drop unknown relapse labels

🔹 Stratified train/test split

### Feature Engineering

🔹 Compute variance of each gene

🔹 Select top 1000 most variable genes

🔹 Standard scaling (StandardScaler)

### Class Imbalance Handling

🔹 class_weight="balanced" for LR & RF

🔹 SMOTE oversampling for XGBoost training

## Models Trained

🔹 Logistic Regression

🔹 Random Forest

🔹 XGBoost (best-performing)

## Evaluation

For each model:

🔹 Accuracy

🔹 ROC-AUC

🔹 Confusion matrices

🔹 Precision/Recall

🔹 Classification report

🔹 Top 15 feature importance scores

🔹 Plots include:

ROC curves

Confusion matrices

Feature importance barplots

All stored in Visualizations.pdf.

## 📈 Key Results

🔹 Best Model: XGBoost

🔹 Accuracy: 93.5%

🔹 AUC improved over baseline

🔹 Recall of relapse cases improved after SMOTE

Note: Due to class imbalance and small event counts, results should be interpreted as exploratory rather than clinical.

## 📂 Repository Structure
AIML_Assessment_Repo/
│── process_gse20685.py
│── breast_cancer_relapse_prediction.py
│── Visualizations.pdf
│── README.md


## 🎯 Purpose of This Project

This project was created to practice:

🔹 reconstructing structured datasets from raw GEO files

🔹 cleaning and merging gene expression with clinical metadata

🔹 applying ML methods in high-dimensional biomedical settings

🔹 evaluating performance under severe class imbalance

🔹 understanding limitations of relapse prediction from microarray data

It is an educational machine learning project, not a clinical model.

### 🤝 Author

Khushi Tyagi
Bioinformatics • Machine Learning • Cancer Genomics
