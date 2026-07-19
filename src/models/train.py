"""configs/*.yaml を読み込んでCV学習を行うエントリポイント。

使い方:
    python -m src.models.train --config configs/lightgbm.yaml
"""

from __future__ import annotations

import argparse
import json
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


def load_train_features(config: dict) -> pd.DataFrame:
    # LightGBMはone-hotではなくordinalエンコード版(ネイティブカテゴリ分割用)を使う
    suffix = "_lgbm" if config["model"] == "lightgbm" else ""
    return pd.read_csv(PROCESSED_DIR / f"train_features{suffix}.csv")


def load_categorical_columns() -> list[str]:
    with open(PROCESSED_DIR / "categorical_columns.json", encoding="utf-8") as f:
        return json.load(f)


def build_model(config: dict, n_estimators: int | None = None):
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
    if model_name == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMRegressor(
            **params,
            n_estimators=n_estimators or config["training"]["num_boost_round"],
        )
    raise ValueError(f"未対応のmodel: {model_name}")


def train(config_path: str) -> float:
    config = load_config(config_path)
    is_lightgbm = config["model"] == "lightgbm"
    cat_cols = load_categorical_columns() if is_lightgbm else None

    X = load_train_features(config)
    y = X.pop("SalePrice")
    if config["training"].get("target_log_transform", True):
        y = np.log1p(y)

    folds = get_folds(X, n_splits=config["training"]["n_splits"], seed=config["params"].get("seed", 42))
    oof_pred = np.zeros(len(X))
    best_iterations = []

    for fold_idx, (train_idx, valid_idx) in enumerate(folds):
        X_train_fold, X_valid_fold = X.iloc[train_idx], X.iloc[valid_idx]
        y_train_fold, y_valid_fold = y.iloc[train_idx], y.iloc[valid_idx]

        model = build_model(config)
        if is_lightgbm:
            import lightgbm as lgb
            model.fit(
                X_train_fold, y_train_fold,
                eval_set=[(X_valid_fold, y_valid_fold)],
                callbacks=[lgb.early_stopping(config["training"]["early_stopping_rounds"], verbose=False)],
                categorical_feature=cat_cols,
            )
            best_iterations.append(model.best_iteration_)
        else:
            model.fit(X_train_fold, y_train_fold)
        oof_pred[valid_idx] = model.predict(X_valid_fold)

    score = rmse(y.to_numpy(), oof_pred)
    print(f"CV RMSE(log): {score:.5f}")

    final_n_estimators = int(np.mean(best_iterations)) if best_iterations else None
    final_model = build_model(config, n_estimators=final_n_estimators)
    if is_lightgbm:
        final_model.fit(X, y, categorical_feature=cat_cols)
    else:
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
