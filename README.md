# 🩺 Stroke-Risk Prediction with Explainable Machine Learning

Predicts the likelihood of stroke from routine demographic and clinical factors using reproducible, end-to-end ML pipelines.

---

## 📊 Problem & Data

- **Objective:** Identify high-risk patients so clinicians can prioritise preventive action.
- **Dataset:** [Kaggle – Healthcare Stroke Data](https://www.kaggle.com/fedesoriano/stroke-prediction-dataset)  
  5 110 rows · 11 predictors · 4.9 % positive stroke cases (highly imbalanced).

---

## 🚀 Key Results

| Metric&nbsp;(class 1 = stroke) | Logistic (thr 0.5) | SVM (0.7) | BRF (0.5) | XGB (0.5) | XGB (0.6) | XGB (0.7) |
| --- | --- | --- | --- | --- | --- | --- |
| **Recall** | **0.80** | 0.72 | 0.72 | **0.80** | 0.74 | 0.50 |
| **Precision** | 0.14 | 0.20 | 0.13 | 0.13 | 0.15 | **0.25** |
| **F1** | 0.24 | 0.32 | 0.22 | 0.22 | 0.25 | **0.33** |
| **AUROC** | **0.842** | **0.842** | 0.812 | 0.829 | 0.829 | 0.829 |

> *Logistic Regression* offers the highest sensitivity (recall = 0.80).  
> *XGBoost* lets us trade recall for 2× higher precision by shifting the decision threshold.

---