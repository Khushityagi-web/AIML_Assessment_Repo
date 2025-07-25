# Breast Cancer Relapse Prediction – GSE20685

This project applies **machine learning** to predict **regional relapse** in breast cancer patients using the [GSE20685](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE20685) gene expression dataset.

It includes **data preprocessing, feature selection, class balancing (SMOTE)**, and model training using **Logistic Regression, Random Forest, and XGBoost**, along with visualizations for interpretation.

---

## Project Structure
AIML_Assessment_Repo/
├── process_gse20685.py # Script to process raw GEO data into cleaned dataset
├── breast_cancer_relapse_prediction.py # Final ML pipeline (training, evaluation, plots)
├── Visualizations.pdf # ROC curves, confusion matrices, feature importance plots
├── README.md # Project documentation
└── .gitignore # Ignore large/raw files

---

## Dataset
- **Source:** [NCBI GEO: GSE20685](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE20685)  
- **Platform:** Affymetrix Human Genome U133 Plus 2.0 Array  
- **Samples:** 327 breast cancer patients (gene expression + clinical metadata)  
- **Target Variable:** `regional_relapse` (binary: 0 = no relapse, 1 = relapse)  

### Accessing the Data
Due to size constraints, the **processed dataset (`processed_gse20685_data.csv`) is not included** in this repository.  

To reproduce:
1. **Download raw dataset**: `GSE20685_series_matrix.txt` from [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE20685).  
2. **Run** `process_gse20685.py` to generate the processed CSV.  
3. Alternatively, **contact the repository owner** for the preprocessed file.  

---

## Installation
Clone the repository:
```bash
git clone https://github.com/<your-username>/AIML_Assessment_Repo.git
cd AIML_Assessment_Repo
Install dependencies:
pip install pandas numpy scikit-learn imbalanced-learn xgboost matplotlib

Run the final modeling script:
python breast_cancer_relapse_prediction.py

## Results
Best Model: XGBoost – 93.5% accuracy, improved relapse case detection (recall: 20%).

Visuals: ROC curves, confusion matrices, and top 15 feature importance charts are in Visualizations.pdf.

## References
NCBI GEO: GSE20685

Pedregosa et al., Scikit-learn: Machine Learning in Python, JMLR, 2011.

Chen & Guestrin, XGBoost: A Scalable Tree Boosting System, KDD, 2016.

Imbalanced-learn Documentation.
