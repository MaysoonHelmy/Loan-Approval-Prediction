# Loan Approval Prediction System

## Overview

The **Loan Approval Prediction System** is part of the **Self-Paced Elevvo Machine Learning Internship Level 2**.  

This project develops a machine learning pipeline to predict **loan approval status** based on applicant demographics, income, and financial characteristics. Loan approval prediction plays a vital role in the banking and financial sector, enabling institutions to **automate decision-making, reduce default risk, and improve efficiency**.  

The workflow covers:  
- **Exploratory Data Analysis (EDA)** to understand dataset patterns.  
- **Data preprocessing & feature engineering** to clean and transform raw data.  
- **Feature selection & model development** with Logistic Regression and Decision Tree classifiers.  
- **Evaluation** using accuracy, precision, recall, and F1-score across different feature sets.  

## Table of Contents

- [Features](#features)  
- [Technologies Used](#technologies-used)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Data Sources](#data-sources)  
- [Project Structure](#project-structure)  
- [Results & Visualizations](#results--visualizations)  
- [Key Findings](#key-findings)  
- [Requirements](#requirements)  
- [Conclusion](#conclusion)  

## Features

- Exploratory Data Analysis (EDA) with visualizations and profiling report  
- Data preprocessing and feature engineering  
- Feature selection: **manual**, **SelectKBest**, **Recursive Feature Elimination (RFE)**, and **all features**  
- Model development with:  
  - **Logistic Regression**  
  - **Decision Tree**  
- Evaluation with cross-validation, accuracy, and classification metrics  
- Auto-generated HTML EDA report for easy insights  

## Technologies Used

- Python  
- Jupyter Notebook  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  
- Seaborn  
- Imbalanced-learn  
- ydata-profiling  

## Installation

To set up the project locally:  

```bash
# Clone the repository
git clone https://github.com/yourusername/LoanApprovalPrediction.git  

# Navigate to the project directory
cd LoanApprovalPrediction
````

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn ydata-profiling
```

## Usage

1. **Data Analysis**
   Run `Data Analysis & Visualization/Data_Analysis.ipynb` to explore the raw dataset and generate train/test splits.

2. **Preprocessing & Feature Engineering**
   Run `Data PreProcessing and Feature Engineering/Data_PreProcessing.ipynb` to preprocess and transform data into clean datasets.

3. **Model Development**
   Run `Feature Selection and Model Development/Models.ipynb` to perform feature selection, train models, and evaluate results.

4. **EDA Report**
   Open `LoanApprovalPredictionEDA.html` for a comprehensive static report of the exploratory analysis.

## Data Sources

The dataset used in this project is sourced from Kaggle:
📂 [Loan Approval Prediction Dataset](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset)

Files include:

* `loan_approval_dataset.csv` – Raw dataset containing applicant details and loan status.
* `train_data.csv` – Training dataset.
* `test_data.csv` – Testing dataset.
* `preprocessed_train_data.csv` – Cleaned training dataset.
* `preprocessed_test_data.csv` – Cleaned testing dataset.

## Project Structure

```
LoanApprovalPrediction/
│
├── LoanApprovalPredictionEDA.html
│
├── Data Analysis & Visualization/
│   ├── Data_Analysis.ipynb
│   ├── loan_approval_dataset.csv
│   ├── train_data.csv
│   └── test_data.csv
│
├── Data PreProcessing and Feature Engineering/
│   ├── Data_PreProcessing.ipynb
│   ├── preprocessed_train_data.csv
│   └── preprocessed_test_data.csv
│
├── Feature Selection and Model Development/
│   └── Models.ipynb
│
└── README.md
```

## Results & Visualizations

### Model Performance Comparison

| Model               | Feature Selection Method | CV Accuracy | Train Accuracy | Test Accuracy |
| ------------------- | ------------------------ | ----------- | -------------- | ------------- |
| Logistic Regression | Manual Features          | 94.22%      | 94.39%         | 92.62%        |
| Logistic Regression | SelectKBest              | 94.27%      | 94.32%         | 93.33%        |
| Logistic Regression | RFE Features             | 94.34%      | 94.41%         | 93.33%        |
| Logistic Regression | All Features             | 94.25%      | 94.48%         | 93.09%        |
| Decision Tree       | Manual Features          | 99.67%      | 99.74%         | 100.00%       |
| Decision Tree       | SelectKBest              | 99.67%      | 99.74%         | 100.00%       |
| Decision Tree       | RFE Features             | 99.67%      | 99.74%         | 100.00%       |
| Decision Tree       | All Features             | 99.67%      | 99.74%         | 100.00%       |

### Performance Across Feature Sets

![Model Accuracy Comparison](https://github.com/user-attachments/assets/09879ebf-b263-4d9a-b155-487973eb232c)

### Classification Metrics for Class 1 (Approved)

![Classification Metrics](https://github.com/user-attachments/assets/5fc2a046-b77c-4d96-9e8d-e8c8615e1dfc)

**Insights:**

* Decision Tree achieves perfect metrics across all feature sets.
* Logistic Regression provides balanced results with F1-scores between 0.90–0.91.

## Key Findings

1. **Decision Tree Overfitting**: Perfect scores across all sets suggest overfitting or possible data leakage.
2. **Logistic Regression Reliability**: More realistic and generalizable with \~93% accuracy and strong precision/recall balance.
3. **Feature Selection**: RFE (Recursive Feature Elimination) slightly outperforms other methods, simplifying models without performance loss.
4. **Robustness**: Both models perform consistently across feature selection techniques.

## Requirements

* Python 3.x
* pandas, numpy, matplotlib, seaborn
* scikit-learn, imbalanced-learn, ydata-profiling

## 🎯 Conclusion

While the Decision Tree classifier achieves perfect scores, its results indicate **overfitting and reduced real-world applicability**. Logistic Regression, on the other hand, demonstrates **reliable generalization, interpretability, and robust performance** with \~93% test accuracy.

