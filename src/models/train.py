"""configs/*.yaml を読み込んでCV学習を行うエントリポイント。

使い方:
    python -m src.models.train --config configs/lightgbm.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from src.models.cv import get_folds, rmse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(config: dict):
    model_name = config["model"]
    params = config["params"]
    if model_name in ("ridge", "lasso"):
        from sklearn.linear_model import Lasso, Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        estimator = Ridge(**params) if model_name == "ridge" else Lasso(**params)
        # Ridge/Lassoは係数に一律の正則化をかけるため、スケール差(one-hotの0/1 と 面積等の数値)を
        # 揃えないと正則化が不均等に効いてしまう。StandardScalerはfold内のtrainのみでfitされる。
        return make_pipeline(StandardScaler(), estimator)
    raise ValueError(f"未対応のmodel: {model_name}")


def train(config_path: str) -> float:
    config = load_config(config_path)

    X = pd.read_csv(PROCESSED_DIR / "train_features.csv")
    y = X.pop("SalePrice")
    if config["training"].get("target_log_transform", True):
        y = np.log1p(y)

    folds = get_folds(X, n_splits=config["training"]["n_splits"], seed=config["params"].get("seed", 42))
    oof_pred = np.zeros(len(X))

    for fold_idx, (train_idx, valid_idx) in enumerate(folds):
        model = build_model(config)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        oof_pred[valid_idx] = model.predict(X.iloc[valid_idx])

    score = rmse(y.to_numpy(), oof_pred)
    print(f"CV RMSE(log): {score:.5f}")

    final_model = build_model(config)
    final_model.fit(X, y)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{config['model']}.pkl"
    joblib.dump(final_model, model_path)
    print(f"saved model: {model_path}")

    return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train(args.config)
