__all__ = [
    "evaluate_at",
    "Display",
]
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, PrecisionRecallDisplay
import numpy as np

def evaluate_at(threshold, y_true, y_prob):
    y_pred = (y_prob >= threshold).astype(int)
    print(f"\nThreshold = {threshold:0.3f}")
    print(classification_report(y_true, y_pred, digits=3))
    print("ROC-AUC :", roc_auc_score(y_true, y_prob))

def Display(model,X_test,y_test,thresholds=None):
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    PrecisionRecallDisplay.from_predictions(y_test, model.predict_proba(X_test)[:, 1])