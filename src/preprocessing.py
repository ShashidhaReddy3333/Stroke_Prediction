from __future__ import annotations

from typing import List, Sequence, Tuple, Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.pipeline import Pipeline as Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


__all__ = [
    "get_feature_lists",
    "make_pipeline",
]

# -----------------------------------------------------------------------------
# 1. Auto‑detect numeric vs categorical columns
# -----------------------------------------------------------------------------

def get_feature_lists(df,target):
    X = df.drop(columns=[target])          # remove target once, up-front
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()
    return numeric_cols, categorical_cols

# -----------------------------------------------------------------------------
# 2. Pipeline builder – *very* lightweight
# -----------------------------------------------------------------------------

def make_pipeline( estimator, *,numeric_cols,categorical_cols,oversample=True,random_state=42,):
   
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", RobustScaler()),
    ])

    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocess = ColumnTransformer([
        ("num", num_pipe, numeric_cols),
        ("cat", cat_pipe, categorical_cols),
    ])

    steps = [
        ("preprocess", preprocess),
        ]
    if oversample:
        steps.append(("oversample", SMOTE(random_state=random_state)))
    steps.append(("model", estimator))


    return ImbPipeline(steps)