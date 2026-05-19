# 🧬 Clinical Trial Outcome Predictor

A machine learning web app for predicting clinical trial completion, estimating expected trial duration, and explaining individual predictions using SHAP.

🔗 **Live App:** https://clinical-trial-prediction-dashboard.streamlit.app/

---

## 📌 Project Overview

Clinical trials are expensive, time-consuming, and difficult to complete successfully. Delays or non-completion can occur due to recruitment challenges, funding issues, trial design complexity, safety concerns, or operational limitations.

This project uses historical interventional clinical trial records to build a machine learning dashboard that predicts whether a trial is likely to be **Completed** or **Not Completed**, estimates the expected duration for completed trials, and explains the main factors influencing each prediction.

The project demonstrates an end-to-end data science workflow, including exploratory data analysis, data preprocessing, feature engineering, classification modelling, regression modelling, model explainability, and Streamlit deployment.

Key techniques used include **One-Hot Encoding**, **TF-IDF vectorisation**, **imputation**, **feature scaling**, **XGBoost classification and regression**, and **SHAP explainability**.

---

## 📊 Dataset Source

The dataset was collected from **ClinicalTrials.gov**, a public clinical trial registry maintained by the U.S. National Library of Medicine.

This project used interventional clinical trial records with start dates between **2015 and 2025**. The dataset included trial-level information such as study status, enrollment, phase, eligibility criteria, funder type, locations, conditions, interventions, study design, start date, and completion date.

Two machine learning tasks were developed:

- **Classification:** predict whether a trial is likely to be completed or not completed
- **Regression:** estimate the expected duration of completed trials in months

---

## 📊 Exploratory Data Analysis

EDA was carried out to understand the structure, quality, and distribution of the clinical trial records before modelling.

Key findings included:

- The dataset contained over **60,000 interventional clinical trial records**
- The completion target was created from `Study Status`
- Completed trials formed the larger class, meaning the classification task had class imbalance
- Trial duration showed a right-skewed distribution, with some trials lasting much longer than most others
- Enrollment also contained extreme outliers, with some studies having very large participant counts
- Some columns contained missing values, which required imputation during preprocessing
- Date columns were important for creating `start_year` and calculating `duration_months`

These findings guided the preprocessing, leakage removal, feature engineering, and model selection steps.

---

## 🧹 Data Preprocessing

The main preprocessing steps included:

- Created the completion target:
  - `1 = Completed`
  - `0 = Not Completed`
- Calculated trial duration in months using start and completion dates
- Removed leakage columns such as study status, completion date, target status, and duration-related fields
- Handled missing values using imputation
- Applied **One-Hot Encoding** to categorical variables
- Applied **TF-IDF vectorisation** to condition and intervention text fields
- Scaled numerical features
- Created engineered features such as condition count, intervention count, location count, multisite flag, age-group flags, phase flags, intervention-type flags, country flags, and sponsor frequency
- Saved the preprocessing pipeline using `joblib` so the same transformations could be reused in the Streamlit app

---

## 🧠 Model Selection

Two baseline models were compared with XGBoost models:

| Task | Baseline Model | Final Model |
|---|---|---|
| Completion prediction | Logistic Regression | XGBoost Classifier |
| Duration estimation | Linear Regression | XGBoost Regressor |

XGBoost was selected because it performed better than the baseline models and was more suitable for this dataset. Clinical trial data contains mixed feature types, including numerical variables, categorical variables, engineered features, and text-derived TF-IDF features. XGBoost can capture non-linear patterns and interactions between these variables more effectively than simpler linear models.

---

## 📈 Model Results

### Completion Classification

The classification model predicts whether a clinical trial is likely to be **Completed** or **Not Completed**.

| Model | Accuracy | ROC-AUC | Precision Completed | Recall Completed | F1 Completed | Precision Not Completed | Recall Not Completed | F1 Not Completed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.695 | 0.761 | 0.880 | 0.697 | 0.778 | 0.408 | 0.688 | 0.512 |
| XGBoost | 0.851 | 0.889 | 0.916 | 0.886 | 0.901 | 0.662 | 0.733 | 0.696 |

XGBoost achieved stronger classification performance, improving accuracy from **0.695 to 0.851** and ROC-AUC from **0.761 to 0.889**. It also improved the F1-score for the **Not Completed** class from **0.512 to 0.696**, making it better at identifying trials at risk of non-completion.

### Duration Regression

The regression model estimates the expected duration of completed clinical trials in months.

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 11.982 | 16.115 | 0.465 |
| XGBoost Regressor | 11.132 | 15.185 | 0.525 |

XGBoost Regressor achieved better duration prediction performance, reducing MAE from **11.982 to 11.132 months** and improving R² from **0.465 to 0.525**. The improvement was moderate, but the model explained more variation in trial duration than the Linear Regression baseline.

---

## 🔍 SHAP Explainability

SHAP was used to explain individual completion predictions. Instead of only showing the predicted class, the app also shows which features pushed the prediction toward **Completed** or **Not Completed**.

Example output from the app:

```text
Predicted outcome: Completed
Completion probability: 91.5%

Main features influencing this prediction:
1. Enrollment: 60 participants — pushed toward Completed
2. Location count: 1 location — pushed toward Completed
3. Start year: 2022 — pushed toward Not Completed
4. Phase 1 — pushed toward Completed
5. Includes older adults — pushed toward Not Completed
```

SHAP improves transparency by showing model-based feature contributions. These explanations should be interpreted as prediction-level explanations, not as proof of real-world causation.

---

## 🖥️ Streamlit App Features

The deployed dashboard allows users to enter trial-level details and receive:

- Predicted trial outcome: **Completed** or **Not Completed**
- Completion or non-completion probability
- Estimated duration for trials predicted as completed
- SHAP explanation showing the main features influencing the prediction

The app includes three sections:

- **Home:** project introduction and clinical trial context
- **Predict Trial Outcome:** user input form, prediction result, duration estimate, and SHAP explanation
- **About the Model:** technical summary of models, inputs, and explainability

---

## ✅ Final Deployed Models

The deployed Streamlit app uses:

- **XGBoost Classifier** for completion prediction
- **XGBoost Regressor** for duration estimation
- **SHAP** for prediction-level explainability
- Saved preprocessing pipelines to ensure consistent app input transformation

---

## 📁 Project Structure

```text
clinical-trial-outcome-predictor/
│
├── app.py
├── requirements.txt
│
├── eda_clinical_trials.ipynb
├── 01_completion_classification_model.ipynb
├── 02_duration_regression_model.ipynb
│
├── completion_xgboost_model.pkl
├── completion_preprocessor.pkl
├── feature_builder.pkl
├── raw_cols_to_drop.pkl
│
├── duration_xgboost_model.pkl
├── duration_preprocessor.pkl
├── duration_feature_builder.pkl
├── duration_raw_cols_to_drop.pkl
│
├── clinical_trials.png
├── side_bar.jpg
└── README.md

```

---



## 👩‍💻 Author

**Pooja Bisht**  
MSc Data Science 
