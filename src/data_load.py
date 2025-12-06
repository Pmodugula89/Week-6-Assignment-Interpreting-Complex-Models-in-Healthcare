"""
data_load.py
Loads a small synthetic healthcare-like dataset or reads from data/raw/.
Splits into train/test using stratification to handle imbalance.
"""

from typing import Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

def generate_synthetic(n: int = 2500, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    age = rng.integers(18, 90, size=n)
    heart_rate = rng.normal(85, 15, size=n).clip(40, 160)
    systolic_bp = rng.normal(125, 20, size=n).clip(80, 200)
    creatinine = rng.normal(1.0, 0.45, size=n).clip(0.3, 6.0)
    lactate = rng.normal(1.8, 0.9, size=n).clip(0.4, 10.0)
    temp_c = rng.normal(36.8, 0.7, size=n).clip(34.0, 41.5)
    sex = rng.choice(["F", "M"], size=n)
    smoker = rng.choice([0, 1], size=n, p=[0.72, 0.28])

    # Nonlinear risk logit; mimic sepsis/AKI risk drivers + noise
    logit = (
        0.025 * (age - 55)
        + 0.035 * (heart_rate - 90)
        - 0.015 * (systolic_bp - 120)
        + 0.9 * (creatinine - 1.1)
        + 0.3 * (lactate - 2.0)
        + 0.45 * (temp_c - 37.0)
        + 0.25 * smoker
        + rng.normal(0, 0.55, size=n)
    )
    prob = 1 / (1 + np.exp(-logit))
    y = (prob > 0.5).astype(int)

    df = pd.DataFrame({
        "age": age,
        "heart_rate": heart_rate,
        "systolic_bp": systolic_bp,
        "creatinine": creatinine,
        "lactate": lactate,
        "temp_c": temp_c,
        "sex": sex,
        "smoker": smoker,
        "at_risk": y
    })
    return df

def load_data(path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if path:
        df = pd.read_csv(path)
    else:
        df = generate_synthetic()

    # Ensure target exists
    if "at_risk" not in df.columns:
        raise ValueError("Dataset must contain 'at_risk' as target label.")

    X = df.drop(columns=["at_risk"])
    y = df["at_risk"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test
