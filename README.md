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
| Logistic Regression | Manual Features          | 94.22%      | 94.39%         | 92.51%        |
| Logistic Regression | SelectKBest              | 94.29%      | 94.32%         | 93.21%        |
| Logistic Regression | RFE Features             | 94.41%      | 94.36%         | 93.21%        |
| Logistic Regression | All Features             | 94.25%      | 94.48%         | 93.09%        |
| Decision Tree       | Manual Features          | 78.09%      | 72.50%         | 70.14%        |
| Decision Tree       | SelectKBest              | 86.42%      | 90.90%         | 91.57%        |
| Decision Tree       | RFE Features             | 96.46%      | 95.77%         | 95.90%        |
| Decision Tree       | All Features             | 99.17%      | 99.15%         | 99.29%        |

### Performance Across Feature Sets

<img width="1392" height="430" alt="image" src="https://github.com/user-attachments/assets/bc639301-9cd5-44e3-947d-9faef9fa8f14" />

### Classification Metrics for Class 1 (Approved)

<img width="1375" height="208" alt="image" src="https://github.com/user-attachments/assets/7a93c1e4-40c2-4a9f-b874-88aa0e6a7be4" />

**Insights:**

* Logistic Regression shows **consistent \~93% accuracy** with balanced precision and recall.
* Decision Tree performance varies significantly:

  * With manual features, it underperforms (\~70%).
  * With SelectKBest or RFE, it improves drastically (91–96%).
  * With all features, it achieves **near-perfect 99% accuracy**, though this could indicate **overfitting**.

## Key Findings

1. **Decision Tree Variability**: Performance ranges widely depending on feature selection, from 70% to 99% accuracy.
2. **Logistic Regression Stability**: Provides consistently reliable results with \~93% accuracy and balanced classification metrics.
3. **Feature Selection Impact**: RFE boosts both Logistic Regression and Decision Tree performance without unnecessary complexity.
4. **Possible Overfitting**: The Decision Tree with all features may not generalize well despite its near-perfect results.

## Requirements

* Python 3.x
* pandas, numpy, matplotlib, seaborn
* scikit-learn, imbalanced-learn, ydata-profiling

## 🎯 Conclusion

The updated results reveal that:

* **Logistic Regression** remains the **most reliable model** for generalization, achieving \~93% accuracy with strong precision and recall across all feature sets.
* **Decision Trees** show **inconsistent behavior**: while they can reach up to **99% accuracy with all features**, this suggests **overfitting** and reduced real-world robustness.
* **Best Trade-off**: Logistic Regression with RFE or SelectKBest features offers an excellent balance of performance and interpretability.

👉 **Final Recommendation**: Use **Logistic Regression with RFE features** for deployment in real-world loan approval systems.
