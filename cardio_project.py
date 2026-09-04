# ============================================================
# CARDIOVASCULAR DISEASE PREDICTION
# ============================================================

# -----------------------------
# 1. Import Libraries
# -----------------------------

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# -----------------------------
# 2. Create plots folder
# -----------------------------

os.makedirs("plots", exist_ok=True)


# -----------------------------
# 3. Load Dataset
# -----------------------------

file_path = "cardio_train (1).csv"

df = pd.read_csv(file_path, sep=";")

print("\n================ DATASET INFORMATION ================\n")

print("Shape of dataset:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())


# -----------------------------
# 4. Basic Statistics
# -----------------------------

print("\n================ DESCRIPTIVE STATISTICS ================\n")

print(df.describe())


# -----------------------------
# 5. Data Preprocessing
# -----------------------------

# Convert age from days to years
df["age_years"] = df["age"] / 365.25

# Calculate BMI
df["bmi"] = df["weight"] / ((df["height"] / 100) ** 2)

print("\nAge and BMI columns created.")

# Remove unrealistic/outlier values
df = df[
    (df["height"].between(120, 220)) &
    (df["weight"].between(30, 150)) &
    (df["ap_hi"].between(80, 200)) &
    (df["ap_lo"].between(40, 150)) &
    (df["ap_hi"] >= df["ap_lo"])
].copy()

print("\nDataset shape after preprocessing:", df.shape)


# -----------------------------
# 6. Target Distribution
# -----------------------------

plt.figure(figsize=(7, 5))

sns.countplot(data=df, x="cardio")

plt.title("Distribution of Cardiovascular Disease")
plt.xlabel("Cardiovascular Disease")
plt.ylabel("Number of Patients")

plt.tight_layout()
plt.savefig("plots/target_distribution.png")
plt.show()


# -----------------------------
# 7. Age Distribution
# -----------------------------

plt.figure(figsize=(8, 5))

sns.histplot(df["age_years"], bins=30, kde=True)

plt.title("Age Distribution")
plt.xlabel("Age (Years)")
plt.ylabel("Number of Patients")

plt.tight_layout()
plt.savefig("plots/age_distribution.png")
plt.show()


# -----------------------------
# 8. Weight Distribution
# -----------------------------

plt.figure(figsize=(8, 5))

sns.histplot(df["weight"], bins=30, kde=True)

plt.title("Weight Distribution")
plt.xlabel("Weight (kg)")
plt.ylabel("Number of Patients")

plt.tight_layout()
plt.savefig("plots/weight_distribution.png")
plt.show()


# -----------------------------
# 9. Blood Pressure Distribution
# -----------------------------

plt.figure(figsize=(8, 5))

sns.histplot(df["ap_hi"], bins=30, kde=True)

plt.title("Systolic Blood Pressure Distribution")
plt.xlabel("Systolic Blood Pressure")
plt.ylabel("Number of Patients")

plt.tight_layout()
plt.savefig("plots/systolic_bp_distribution.png")
plt.show()


plt.figure(figsize=(8, 5))

sns.histplot(df["ap_lo"], bins=30, kde=True)

plt.title("Diastolic Blood Pressure Distribution")
plt.xlabel("Diastolic Blood Pressure")
plt.ylabel("Number of Patients")

plt.tight_layout()
plt.savefig("plots/diastolic_bp_distribution.png")
plt.show()


# -----------------------------
# 10. Cholesterol vs Cardio
# -----------------------------

plt.figure(figsize=(8, 5))

sns.countplot(data=df, x="cholesterol", hue="cardio")

plt.title("Cholesterol Level vs Cardiovascular Disease")
plt.xlabel("Cholesterol Level")
plt.ylabel("Number of Patients")

plt.tight_layout()
plt.savefig("plots/cholesterol_vs_cardio.png")
plt.show()


# -----------------------------
# 11. Glucose vs Cardio
# -----------------------------

plt.figure(figsize=(8, 5))

sns.countplot(data=df, x="gluc", hue="cardio")

plt.title("Glucose Level vs Cardiovascular Disease")
plt.xlabel("Glucose Level")
plt.ylabel("Number of Patients")

plt.tight_layout()
plt.savefig("plots/glucose_vs_cardio.png")
plt.show()


# -----------------------------
# 12. Gender vs Cardio
# -----------------------------

plt.figure(figsize=(8, 5))

sns.countplot(data=df, x="gender", hue="cardio")

plt.title("Gender vs Cardiovascular Disease")
plt.xlabel("Gender")
plt.ylabel("Number of Patients")

plt.tight_layout()
plt.savefig("plots/gender_vs_cardio.png")
plt.show()


# -----------------------------
# 13. Physical Activity vs Cardio
# -----------------------------

plt.figure(figsize=(8, 5))

sns.countplot(data=df, x="active", hue="cardio")

plt.title("Physical Activity vs Cardiovascular Disease")
plt.xlabel("Active")
plt.ylabel("Number of Patients")

plt.tight_layout()
plt.savefig("plots/activity_vs_cardio.png")
plt.show()


# -----------------------------
# 14. Smoking vs Cardio
# -----------------------------

plt.figure(figsize=(8, 5))

sns.countplot(data=df, x="smoke", hue="cardio")

plt.title("Smoking vs Cardiovascular Disease")
plt.xlabel("Smoking")
plt.ylabel("Number of Patients")

plt.tight_layout()
plt.savefig("plots/smoking_vs_cardio.png")
plt.show()


# -----------------------------
# 15. BMI Distribution
# -----------------------------

plt.figure(figsize=(8, 5))

sns.histplot(df["bmi"], bins=30, kde=True)

plt.title("BMI Distribution")
plt.xlabel("BMI")
plt.ylabel("Number of Patients")

plt.tight_layout()
plt.savefig("plots/bmi_distribution.png")
plt.show()


# -----------------------------
# 16. BMI vs Cardiovascular Disease
# -----------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(data=df, x="cardio", y="bmi")

plt.title("BMI vs Cardiovascular Disease")
plt.xlabel("Cardiovascular Disease")
plt.ylabel("BMI")

plt.tight_layout()
plt.savefig("plots/bmi_vs_cardio.png")
plt.show()


# -----------------------------
# 17. Age vs Cardiovascular Disease
# -----------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(data=df, x="cardio", y="age_years")

plt.title("Age vs Cardiovascular Disease")
plt.xlabel("Cardiovascular Disease")
plt.ylabel("Age (Years)")

plt.tight_layout()
plt.savefig("plots/age_vs_cardio.png")
plt.show()


# -----------------------------
# 18. Blood Pressure vs Cardio
# -----------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(data=df, x="cardio", y="ap_hi")

plt.title("Systolic Blood Pressure vs Cardiovascular Disease")
plt.xlabel("Cardiovascular Disease")
plt.ylabel("Systolic Blood Pressure")

plt.tight_layout()
plt.savefig("plots/bp_vs_cardio.png")
plt.show()


# -----------------------------
# 19. Correlation Matrix
# -----------------------------

correlation_columns = [
    "age_years",
    "gender",
    "height",
    "weight",
    "ap_hi",
    "ap_lo",
    "cholesterol",
    "gluc",
    "smoke",
    "alco",
    "active",
    "bmi",
    "cardio"
]

correlation_matrix = df[correlation_columns].corr()

print("\n================ CORRELATION WITH CARDIO ================\n")

print(
    correlation_matrix["cardio"]
    .sort_values(ascending=False)
)


plt.figure(figsize=(12, 9))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title("Correlation Matrix")

plt.tight_layout()
plt.savefig("plots/correlation_matrix.png")
plt.show()


# -----------------------------
# 20. Prepare Data for ML
# -----------------------------

features = [
    "age_years",
    "gender",
    "height",
    "weight",
    "ap_hi",
    "ap_lo",
    "cholesterol",
    "gluc",
    "smoke",
    "alco",
    "active",
    "bmi"
]

X = df[features]

y = df["cardio"]


# -----------------------------
# 21. Train-Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# -----------------------------
# 22. Define ML Models
# -----------------------------

models = {

    "Logistic Regression":
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=1000,
                random_state=42
            ))
        ]),

    "SVM":
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearSVC(
                random_state=42,
                dual="auto",
                max_iter=5000
            ))
        ]),

    "KNN":
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(
                n_neighbors=7,
                n_jobs=-1
            ))
        ]),

    "Decision Tree":
        DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=10,
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=3,
            n_jobs=-1,
            random_state=42
        )
}


# -----------------------------
# 23. Train and Evaluate Models
# -----------------------------

results = []

trained_models = {}

for name, model in models.items():

    print("\nTraining:", name)

    model.fit(X_train, y_train)

    trained_models[name] = model

    predictions = model.predict(X_test)

    # Probability / decision score
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X_test)[:, 1]
    else:
        scores = model.decision_function(X_test)

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions
    )

    recall = recall_score(
        y_test,
        predictions
    )

    f1 = f1_score(
        y_test,
        predictions
    )

    roc_auc = roc_auc_score(
        y_test,
        scores
    )

    results.append([
        name,
        accuracy,
        precision,
        recall,
        f1,
        roc_auc
    ])


# -----------------------------
# 24. Model Comparison
# -----------------------------

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ]
)

results_df = results_df.sort_values(
    by="Accuracy",
    ascending=False
)

print("\n================ MODEL COMPARISON ================\n")

print(results_df)


# Save results
results_df.to_csv(
    "model_results.csv",
    index=False
)


# -----------------------------
# 25. Model Accuracy Plot
# -----------------------------

plt.figure(figsize=(10, 6))

sns.barplot(
    data=results_df,
    x="Accuracy",
    y="Model"
)

plt.title("Machine Learning Model Accuracy")

plt.xlabel("Accuracy")
plt.ylabel("Model")

plt.xlim(0, 1)

plt.tight_layout()
plt.savefig("plots/model_accuracy.png")
plt.show()


# -----------------------------
# 26. Best Model
# -----------------------------

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[best_model_name]

print("\n================================================")
print("BEST MODEL:", best_model_name)
print("================================================")


# -----------------------------
# 27. Confusion Matrix
# -----------------------------

best_predictions = best_model.predict(X_test)

cm = confusion_matrix(
    y_test,
    best_predictions
)

plt.figure(figsize=(7, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Disease", "Disease"],
    yticklabels=["No Disease", "Disease"]
)

plt.title(f"Confusion Matrix - {best_model_name}")

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.savefig("plots/confusion_matrix.png")
plt.show()


# -----------------------------
# 28. Classification Report
# -----------------------------

print("\n================ CLASSIFICATION REPORT ================\n")

print(
    classification_report(
        y_test,
        best_predictions
    )
)


# -----------------------------
# 29. Random Forest Feature Importance
# -----------------------------

rf_model = trained_models["Random Forest"]

importance = rf_model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": features,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n================ FEATURE IMPORTANCE ================\n")

print(feature_importance)


plt.figure(figsize=(10, 6))

sns.barplot(
    data=feature_importance,
    x="Importance",
    y="Feature"
)

plt.title("Random Forest Feature Importance")

plt.xlabel("Importance")
plt.ylabel("Feature")

plt.tight_layout()
plt.savefig("plots/feature_importance.png")
plt.show()


# -----------------------------
# 30. Final Output
# -----------------------------

print("\n================================================")
print("PROJECT COMPLETED SUCCESSFULLY")
print("================================================")

print("\nBest Model:", best_model_name)

print(
    "Best Model Accuracy:",
    round(results_df.iloc[0]["Accuracy"] * 100, 2),
    "%"
)

print("\nAll plots have been saved inside the 'plots' folder.")

print("\nModel results saved as: model_results.csv")