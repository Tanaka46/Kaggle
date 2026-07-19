"""交差検証まわりの共通処理。"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def get_folds(X: pd.DataFrame, n_splits: int = 5, seed: int = 42) -> list[tuple[np.ndarray, np.ndarray]]:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(kf.split(X))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
