# Loan Approval Prediction System

## Overview

This project develops a machine learning system to predict loan approval status based on applicant information and financial characteristics. The system follows a comprehensive pipeline from data analysis through preprocessing, feature engineering, model development, and evaluation.


##  Folder Structure

```
LoanApprovalPrediction/
│
├── LoanApprovalPredictionEDA.html
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

##  Components

### 1. Data Analysis & Visualization

- **Data_Analysis.ipynb**: Conducts exploratory data analysis using pandas, seaborn, and ydata-profiling. Includes data visualization, summary statistics, and pattern identification.
- **loan_approval_dataset.csv**: Raw dataset containing 13 features and the target variable.
- **train_data.csv / test_data.csv**: Training and testing datasets created from the raw data.
- **LoanApprovalPredictionEDA.html**: Auto-generated comprehensive EDA report.

### 2. Data Preprocessing & Feature Engineering

- **Data_PreProcessing.ipynb**: Implements the preprocessing pipeline 
- **preprocessed_train_data.csv / preprocessed_test_data.csv**: Cleaned and transformed datasets ready for modeling.

### 3. Feature Selection & Model Development

- **Models.ipynb**: Performs Feature Selection and Model Development and comparison between logistic Regression and decision trees

---

##  Results Summary

### Model Performance Comparison

| Model | Feature Selection Method | CV Accuracy | Train Accuracy | Test Accuracy |
|-------|--------------------------|-------------|----------------|---------------|
| **Logistic Regression** | Manual Features | 94.22% | 94.39% | 92.62% |
| **Logistic Regression** | SelectKBest | 94.27% | 94.32% | 93.33% |
| **Logistic Regression** | RFE Features | 94.34% | 94.41% | 93.33% |
| **Logistic Regression** | All Features | 94.25% | 94.48% | 93.09% |
| **Decision Tree** | Manual Features | 99.67% | 99.74% | 100.00% |
| **Decision Tree** | SelectKBest | 99.67% | 99.74% | 100.00% |
| **Decision Tree** | RFE Features | 99.67% | 99.74% | 100.00% |
| **Decision Tree** | All Features | 99.67% | 99.74% | 100.00% |

### Model Performance Across Feature Sets

<img width="1790" height="616" alt="image" src="https://github.com/user-attachments/assets/09879ebf-b263-4d9a-b155-487973eb232c" />

This chart compares cross-validation, training, and test accuracy across models and feature sets. The Decision Tree achieves perfect scores across all metrics, while Logistic Regression demonstrates consistent but slightly lower performance.

### Classification Metrics for Class 1 (Approved)

<img width="1390" height="590" alt="image" src="https://github.com/user-attachments/assets/5fc2a046-b77c-4d96-9e8d-e8c8615e1dfc" />

This grouped bar chart displays the precision, recall, and F1-score for approved loan predictions. Decision Tree shows perfect metrics across all feature sets, while Logistic Regression achieves strong but realistic performance with F1-scores between 0.90 and 0.91.

### Detailed Performance Metrics

#### Logistic Regression Performance
- **Best Parameters**: C=1, penalty='l1' (for most feature sets)
- **Test Accuracy Range**: 92.62% - 93.33%
- **Precision for Approved Loans**: 87-89%
- **Recall for Approved Loans**: 93-94%
- **F1-Score for Approved Loans**: 0.90-0.91

#### Decision Tree Performance
- **Best Parameters**: ccp_alpha=0, max_depth=3, min_samples_leaf=20, min_samples_split=50
- **Test Accuracy**: Perfect 100% across all feature sets
- **All Metrics**: Perfect scores (1.00) for precision, recall, and F1-score

---

##  Key Findings

1. **Decision Tree Overfitting**: The Decision Tree model achieves perfect accuracy on both training and test sets, which suggests potential overfitting or data leakage that requires further investigation.

2. **Logistic Regression Reliability**: Logistic Regression provides more realistic and generalizable results with test accuracy around 93% and balanced performance across metrics.

3. **Feature Selection Effectiveness**: RFE (Recursive Feature Elimination) performed slightly better than manual selection, maintaining predictive power while reducing model complexity.

4. **Model Consistency**: Both models showed consistent performance across different feature selection methods, indicating robustness in the feature engineering process.

---

##  How to Run

1. **Data Analysis**
   Open and run `Data Analysis & Visualization/Data_Analysis.ipynb` to explore the data and generate train/test splits.

2. **Preprocessing & Feature Engineering**
   Run `Data PreProcessing and Feature Engineering/Data_PreProcessing.ipynb` to preprocess the data and create new features. This will output preprocessed CSVs.

3. **Model Development**
   Run `Feature Selection and Model Development/Models.ipynb` to perform feature selection, train models, and visualize results.

---

##  Requirements

- Python 3.x
- pandas, numpy, matplotlib, seaborn, scikit-learn, imbalanced-learn, ydata-profiling

Install dependencies with:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn ydata-profiling
```

---

##  Performance Details

### Logistic Regression Results

**Manual Features:**
- Best Params: {'C': 0.1, 'penalty': 'l1'}
- CV Score: 94.22%
- Train Score: 94.39%
- Test Score: 92.62%

**SelectKBest Features:**
- Best Params: {'C': 1, 'penalty': 'l1'}
- CV Score: 94.27%
- Train Score: 94.32%
- Test Score: 93.33%

**RFE Features:**
- Best Params: {'C': 1, 'penalty': 'l1'}
- CV Score: 94.34%
- Train Score: 94.41%
- Test Score: 93.33%

**All Features:**
- Best Params: {'C': 1, 'penalty': 'l1'}
- CV Score: 94.25%
- Train Score: 94.48%
- Test Score: 93.09%

### Decision Tree Results

All feature selection methods achieved identical optimal performance:
- Best Params: {'ccp_alpha': 0, 'max_depth': 3, 'min_samples_leaf': 20, 'min_samples_split': 50}
- CV Score: 99.67%
- Train Score: 99.74%
- Test Score: 100.00%

---

## 🎯 Conclusion

Based on a comprehensive evaluation, Logistic Regression emerges as the superior choice for loan approval prediction, delivering reliable 93% accuracy with strong generalization capabilities, while the Decision Tree's perfect 100% performance indicates concerning overfitting that undermines its real-world credibility. The Logistic Regression model provides the optimal balance of predictive accuracy, interpretability, and trustworthiness required for financial decision-making systems.
