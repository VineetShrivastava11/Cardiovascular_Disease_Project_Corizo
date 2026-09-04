# 🫀 Cardiovascular Disease Prediction using Machine Learning

## 📌 Project Overview

This project focuses on predicting the presence of cardiovascular disease using Machine Learning techniques.

The dataset contains health and lifestyle information of patients, including age, gender, height, weight, blood pressure, cholesterol, glucose level, smoking, alcohol consumption, and physical activity.

The project follows a complete Machine Learning workflow, including data preprocessing, exploratory data analysis, visualization, feature engineering, model training, evaluation, and model comparison.
---

## 🎯 Objective

The main objective of this project is to:

- Analyze cardiovascular disease patient data
- Perform data cleaning and preprocessing
- Explore relationships between different health factors
- Visualize important patterns in the dataset
- Train multiple Machine Learning classification models
- Compare their performance
- Identify the best-performing model
---

## 📊 Dataset

The dataset contains approximately **70,000 patient records**.

### Important Features

| Feature | Description |
|---|---|
| `age` | Age of the patient in days |
| `gender` | Gender category |
| `height` | Height in cm |
| `weight` | Weight in kg |
| `ap_hi` | Systolic blood pressure |
| `ap_lo` | Diastolic blood pressure |
| `cholesterol` | Cholesterol level |
| `gluc` | Glucose level |
| `smoke` | Smoking indicator |
| `alco` | Alcohol consumption indicator |
| `active` | Physical activity indicator |
| `cardio` | Target variable |

### Target Variable

`cardio`

- `0` → No cardiovascular disease
- `1` → Cardiovascular disease
---

## 🧹 Data Preprocessing

The following preprocessing steps were performed:

- Checked for missing values
- Checked and handled duplicate records
- Converted age from days into years
- Created a new BMI feature
- Removed unrealistic height and weight values
- Removed implausible blood pressure values
- Removed records where systolic blood pressure was lower than diastolic blood pressure
- Removed the `id` column from the Machine Learning features
---

## 🔍 Exploratory Data Analysis

Several visualizations were created to understand the dataset, including:

- Cardiovascular disease distribution
- Age distribution
- Weight distribution
- Blood pressure distribution
- Cholesterol vs cardiovascular disease
- Glucose vs cardiovascular disease
- Gender vs cardiovascular disease
- Smoking vs cardiovascular disease
- Physical activity vs cardiovascular disease
- BMI distribution
- Age vs cardiovascular disease
- Blood pressure vs cardiovascular disease
---

## 📈 Correlation Analysis

A correlation matrix was created to understand the relationships between different features and the target variable.

Some of the features showing stronger associations with the target were:

- Systolic blood pressure (`ap_hi`)
- Diastolic blood pressure (`ap_lo`)
- Age
- Cholesterol
- BMI

> Correlation represents an association in this dataset and does not imply causation.
---

## 🤖 Machine Learning Models

The following classification algorithms were implemented:

1. Logistic Regression
2. Support Vector Machine (SVM)
3. K-Nearest Neighbors (KNN)
4. Decision Tree
5. Random Forest

The dataset was divided into:

- **80% Training Data**
- **20% Testing Data**

Feature scaling was applied where appropriate.
---

## 🏆 Model Performance

The models were evaluated using Accuracy, Precision, Recall, F1 Score, and ROC-AUC.

| Model | Accuracy |
|---|---:|
| Random Forest | **73.40%** |
| Decision Tree | 73.34% |
| Logistic Regression | 72.84% |
| SVM | 72.75% |
| KNN | 70.60% |

### 🥇 Best Model

**Random Forest** achieved the highest accuracy among the tested models, with an accuracy of approximately **73.40%**.
---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
---

## 📁 Project Structure

```text
Cardiovascular-Disease-Prediction/
│
├── cardio_train (1).csv
├── cardio_project.py
├── model_results.csv
├── README.md
│
└── plots/
    ├── target_distribution.png
    ├── age_distribution.png
    ├── weight_distribution.png
    ├── systolic_bp_distribution.png
    ├── diastolic_bp_distribution.png
    ├── cholesterol_vs_cardio.png
    ├── glucose_vs_cardio.png
    ├── gender_vs_cardio.png
    ├── activity_vs_cardio.png
    ├── smoking_vs_cardio.png
    ├── bmi_distribution.png
    ├── bmi_vs_cardio.png
    ├── age_vs_cardio.png
    ├── bp_vs_cardio.png
    ├── correlation_matrix.png
    ├── model_accuracy.png
    ├── confusion_matrix.png
    └── feature_importance.png
