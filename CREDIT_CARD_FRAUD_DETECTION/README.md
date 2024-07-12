# CREDIT_CARD_FRAUD_DETECTION
FRAUD DETECTION : CREDIT_CARD

### DOWNLOAD THE DATASET FROM : 
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## Table of Contents
1. Introduction
2. Project Overview
3. Dataset
4. Sample Dataset Used
5. Solution Architecture
6. Model Details
7. Installation and Setup
8. Usage
9. Results
10. Visualization
11. Conclusion
12. Acknowledgements

## Introduction
- This project aims to detect fraudulent credit card transactions using machine learning techniques.

## Project Overview
- The goal is to create a model that can distinguish between normal and fraudulent transactions.
- The model uses a Random Forest classifier for detection.
- The project involves data analysis, preprocessing, model training, and evaluation.

## Dataset
- The dataset contains credit card transactions made by European cardholders in September 2013.
- It includes transactions over two days with 492 frauds out of 284,807 transactions.
- The dataset is highly unbalanced, with frauds accounting for 0.172% of all transactions.
- Features are the result of a PCA transformation, except for 'Time' and 'Amount'.
- 'Time' indicates seconds elapsed between each transaction and the first transaction.
- 'Amount' is the transaction amount.
- 'Class' is the response variable (1 for fraud, 0 for normal).

## Sample Dataset Used
- A representative sample of 28,481 transactions was used for training.
- The sample was created to reduce the computational load while maintaining the characteristics of the original dataset.

## Solution Architecture
- Data analysis to understand the dataset.
- Handling missing values and class imbalance using SMOTE.
- Feature scaling using StandardScaler.
- Training a Random Forest classifier.
- Evaluating the model using various metrics.

## Model Details
- Random Forest classifier used with 100 estimators.
- Model evaluation includes accuracy score, classification report, confusion matrix, and ROC curve.
- Isolation Forest used to identify and visualize outliers.

## Installation and Setup
- Install required libraries:
  - numpy: `pip install numpy`
  - pandas: `pip install pandas`
  - matplotlib: `pip install matplotlib`
  - seaborn: `pip install seaborn`
  - scikit-learn: `pip install scikit-learn`
  - imbalanced-learn: `pip install imbalanced-learn`

## Usage
- Import the necessary libraries and load the dataset.
- Perform data analysis and preprocessing.
- Train the Random Forest model.
- Evaluate the model performance.
- Visualize the results.

## Results
- Accuracy Score: 99.98%
- Classification Report: High precision and recall for both normal and fraud classes.
- Confusion Matrix: High accuracy in distinguishing between normal and fraudulent transactions.
- ROC Curve: Area under the curve indicates excellent model performance.

## Visualization
- Pie chart showing transaction class distribution.
- Histograms of transaction amounts and times.
- Scatter plots of time vs. amount by class.
- Correlation matrix heatmap.
- Box plot of transaction amounts by class.
- Scatter plot of outliers identified by Isolation Forest.
- Confusion matrix heatmap.
- ROC curve.

## Conclusion
- The Random Forest model demonstrates exceptional performance in detecting fraudulent transactions.
- Achieved an accuracy score of 99.98% and an F1-score of 1.00 for both classes.
- The model effectively distinguishes between fraudulent and legitimate transactions.
- The dataset used for training was a representative sample, indicating the model's robustness and generalizability.

## Acknowledgements
- Thanks to the dataset providers and the open-source community for their tools and resources.
