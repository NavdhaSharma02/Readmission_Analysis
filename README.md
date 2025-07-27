# Hospital Readmission Prediction

This project aims to predict whether a patient will be readmitted to the hospital within 30 days based on historical clinical data. The model is trained using a real-world diabetic patient dataset and includes both traditional and advanced machine learning techniques.


---

## Technologies Used

- **Python**
- **Pandas & NumPy** – Data loading and preprocessing
- **Scikit-learn** – ML pipeline, logistic regression, evaluation metrics
- **XGBoost** – Advanced gradient boosting classifier
- **SHAP** – Model interpretability and feature impact analysis
- **Matplotlib & Seaborn** – Data and result visualization

---

## Problem Statement

The goal is to build a machine learning model that can predict if a patient is likely to be **readmitted within 30 days** after discharge. Accurate predictions can help improve healthcare planning, reduce costs, and improve patient care.

---

## Data Preprocessing

- Removed identifiers and irrelevant columns (`encounter_id`, `patient_nbr`, etc.)
- Converted target label: `readmitted` → 1 if `<30`, else 0
- Replaced `'?'` with `NaN`
- Categorical features: Imputed with most frequent values, then one-hot encoded
- Numerical features: Imputed with mean, then scaled

---

## Models Trained

### 1. **Logistic Regression**
- Implemented using a full Scikit-learn pipeline
- Interpreted coefficients to identify top predictive features

### 2. **XGBoost Classifier**
- Captures non-linear patterns and interactions
- Tuned for binary classification (`logloss`, `random_state=42`)
- Outperformed logistic regression on ROC-AUC

---

## Evaluation Metrics

- **ROC-AUC Score**
- **Precision-Recall Curve**
- **Classification Report**
- **Confusion Matrix**

---

##  Model Interpretability

Used **SHAP (SHapley Additive exPlanations)** to:
- Visualize global feature importance
- Understand how each feature impacts individual predictions

---

## Results

- Logistic Regression provided good baseline performance.
- XGBoost showed better accuracy and interpretability using SHAP plots.
- Top influencing features identified for hospital readmission prediction.

---

## Future Work

- Hyperparameter tuning for XGBoost
- Try other models like Random Forest or LightGBM
- Apply cross-validation for more robust results
- Deploy the model with a simple web interface (e.g., Flask or Streamlit)

---

## References

- [UCI Diabetes 130-US hospitals dataset](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008)
- SHAP documentation: https://shap.readthedocs.io/

---




