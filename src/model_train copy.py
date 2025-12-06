"""
model_train.py
Builds a scikit-learn Pipeline with preprocessing (scaling + one-hot encoding).
Fits a HistGradientBoostingClassifier for tabular healthcare risk prediction.
"""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier

def build_pipeline(X_train) -> Pipeline:
    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop"
    )

    model = HistGradientBoostingClassifier(
        learning_rate=0.07,
        max_depth=3,
        max_iter=300,
        random_state=42
    )

    clf = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model)
    ])
    return clf

def fit_pipeline(clf: Pipeline, X_train, y_train) -> Pipeline:
    clf.fit(X_train, y_train)
    return clf
