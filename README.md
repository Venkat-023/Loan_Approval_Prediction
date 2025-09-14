Loan Default Prediction Using Machine Learning
Project Overview
This project aims to build robust machine learning models to predict loan default risks based on borrower financial and demographic data. Accurately identifying potential defaulters helps lenders mitigate risk and improve credit decision-making.

Dataset
The dataset includes borrower attributes such as age, income, loan amount, credit score, employment details, loan purpose, and more. It contains both categorical and numerical features, and the target variable indicates loan default status.

Data Preprocessing
Categorical variables were handled using one-hot and label encoding.

Feature selection was informed by correlation analysis and confusion matrix visualization.

Class imbalance was carefully addressed to improve minority class (defaults) detection.

Modeling Techniques
Several classification algorithms were implemented and compared:

Random Forest (with and without PCA)

Logistic Regression (with and without PCA)

XGBoost (with and without PCA)

Stacking ensemble combining base models for improved performance

Evaluation Metrics
Models were evaluated on:

Accuracy

Precision, Recall, and F1-score, especially focusing on the minority class (defaults)

Confusion matrices to analyze true/false positives/negatives

Despite high overall accuracy (~88-89%), identifying defaulters remains challenging due to class imbalance, showing lower recall and F1 scores for the default class.

Key Findings
PCA led to reduced minority class detection performance.

Ensemble stacking improved overall predictive capabilities.

Models favor non-default class prediction; strategies to improve minority recall are recommended.

Usage
Clone this repository.

Install dependencies: pip install -r requirements.txt

Run the data preprocessing and feature engineering scripts.

Train individual models or run the stacking ensemble.

Evaluate models using the included evaluation scripts.

Future Work
Implement advanced imbalance handling techniques such as SMOTE.

Explore additional feature engineering and hyperparameter tuning.

Integrate explainability tools for model interpretability.

Acknowledgements
This project leverages open-source Python libraries including scikit-learn, XGBoost, pandas, and matplotlib.
