# Breast Cancer Relapse Prediction — GSE20685

**Machine Learning + GEO Data Engineering + Clinical Metadata Extraction**

This repository contains a complete workflow for processing the GEO dataset **GSE20685** and training machine learning models to predict regional relapse in breast cancer patients. The project demonstrates:

- Complex GEO metadata parsing  
- High-dimensional gene expression handling  
- Clinical variable extraction  
- Feature engineering  
- Class imbalance methods  
- Model comparison (LR, RF, XGBoost)  
- Performance visualization  

---

## 1. Transcriptomic + Clinical Data Processing

**File:** `process_gse20685.py`

This script performs full reconstruction of the dataset from the raw GEO series matrix.

### GEO Series Matrix Parsing

- Identify `!series_matrix_table_begin` and `end` markers  
- Extract gene expression block dynamically  
- Remove trailing metadata rows  
- Transpose matrix so samples = rows, genes = columns  

### Clinical Metadata Extraction

GEO stores metadata in complex lines such as:

    !Sample_characteristics_ch1 = subtype: Luminal A
    !Sample_characteristics_ch1 = relapse: 0

The script:

- Iterates through all `!Sample_characteristics_ch1` fields  
- Handles missing or unlabeled characteristics  
- Splits key:value pairs  
- Ensures clinical columns are unique and consistent  
- Maps GEO accession IDs to sample titles  
- Merges expression and clinical metadata  
- Performs QC checks for missingness, inconsistent labels, and sample mismatches  

The final merged datasets are saved as:

- `processed_gse20685_data.csv`  
- `gse20685_clinical_data.csv`  

These files become the input for machine learning.

---

## 2. Machine Learning Pipeline

**File:** `breast_cancer_relapse_prediction.py`

### Data Cleaning

- Remove non-numeric columns  
- Drop unknown relapse labels  
- Stratified train/test split  

### Feature Engineering

- Compute variance of each gene  
- Select top 1000 most variable genes  
- Standard scaling using `StandardScaler`  

### Class Imbalance Handling

- `class_weight="balanced"` for Logistic Regression and Random Forest  
- SMOTE oversampling for XGBoost training  

### Models Trained

- Logistic Regression  
- Random Forest  
- XGBoost (best-performing)  

### Evaluation

For each model:

- Accuracy  
- ROC-AUC  
- Confusion matrices  
- Precision / Recall  
- Classification report  
- Top 15 feature importance scores  

Plots include:

- ROC curves  
- Confusion matrices  
- Feature importance bar plots  

All outputs are stored in `Visualizations.pdf`.

---

## Key Results

- Best Model: **XGBoost**  
- Accuracy: **93.5%**  
- AUC improved over baseline  
- Recall of relapse cases improved after SMOTE  

**Note:** Due to class imbalance and small event counts, results should be interpreted as exploratory rather than clinical.

---

## Repository Structure

    AIML_Assessment_Repo/
    │── process_gse20685.py
    │── breast_cancer_relapse_prediction.py
    │── Visualizations.pdf
    │── README.md

---

## Purpose of This Project

This project was created to practice:

- Reconstructing structured datasets from raw GEO files  
- Cleaning and merging gene expression with clinical metadata  
- Applying ML methods in high-dimensional biomedical settings  
- Evaluating performance under severe class imbalance  
- Understanding limitations of relapse prediction from microarray data  

This is an **educational machine learning project**, not a clinical model.

---

## Author

**Khushi Tyagi**  
Bioinformatics • Machine Learning • Cancer Genomics
