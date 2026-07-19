"""configs/*.yaml を読み込んでCV学習を行うエントリポイント。

使い方:
    python -m src.models.train --config configs/lightgbm.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.models.cv import get_folds, rmse

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def train(config_path: str) -> None:
    config = load_config(config_path)

    X = pd.read_csv(PROCESSED_DIR / "train_features.csv")
    y = X.pop("SalePrice")
    if config["training"].get("target_log_transform", True):
        y = np.log1p(y)

    folds = get_folds(X, n_splits=config["training"]["n_splits"], seed=config["params"].get("seed", 42))
    oof_pred = np.zeros(len(X))

    for fold_idx, (train_idx, valid_idx) in enumerate(folds):
        # TODO: config["model"] に応じてモデルを分岐して学習する
        raise NotImplementedError("モデル学習ロジックは特徴量エンジニアリング完了後に実装する")

    score = rmse(y.to_numpy(), oof_pred)
    print(f"CV RMSE(log): {score:.5f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train(args.config)
