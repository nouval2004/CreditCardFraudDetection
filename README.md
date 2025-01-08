# Credit Card Fraud Detection

This project implements machine learning models to detect fraudulent credit card transactions. It uses various techniques including data preprocessing, feature scaling, and both undersampling and oversampling methods to handle imbalanced data.

## Project Overview

The project follows these main steps:
1. Exploratory Data Analysis (EDA)
2. Data Preprocessing
3. Model Selection and Training
4. Model Evaluation and Optimization

## Features

- **Data Processing**:
  - RobustScaler implementation
  - Feature scaling for Time and Amount
  - Handling imbalanced dataset

- **Models Implemented**:
  - Logistic Regression
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
  - Gaussian Naive Bayes
  - Random Forest Classifier

- **Balancing Techniques**:
  - Random Undersampling
  - SMOTE Oversampling

- **Evaluation Metrics**:
  - ROC AUC Score
  - Precision, Recall, F1-Score
  - Confusion Matrix
  - Learning Curves

## Requirements

```
pandas
numpy
scikit-learn
imbalanced-learn
matplotlib
seaborn
kagglehub
```

## Installation

1. Clone the repository:
```bash
git clone [your-repository-url]
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Jupyter notebook:
```bash
jupyter notebook credit_card_fraud_detection.ipynb
```

2. Follow the notebook sections:
   - EDA
   - Data Preprocessing
   - Model Training
   - Model Evaluation

## Model Performance

The project implements multiple models with the following key findings:
- Random Forest achieves the best performance after GridSearchCV optimization
- SMOTE oversampling helps improve model performance on the minority class
- Detailed evaluation metrics are provided for each model
