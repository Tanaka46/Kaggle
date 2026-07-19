"""学習済みモデルでテストデータを推論し、提出用csvを作成するエントリポイント。

使い方:
    python -m src.models.predict --config configs/lightgbm.yaml --model models/lightgbm.pkl
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

PROCESSED_DIR = Path("data/processed")
SUBMISSIONS_DIR = Path("submissions")


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def predict(config_path: str, model_path: str) -> None:
    config = load_config(config_path)
    model = joblib.load(model_path)

    X_test = pd.read_csv(PROCESSED_DIR / "test_features.csv")
    test_ids = X_test.pop("Id")

    pred = model.predict(X_test)
    if config["training"].get("target_log_transform", True):
        pred = np.expm1(pred)

    submission = pd.DataFrame({"Id": test_ids, "SalePrice": pred})
    method = config["model"]
    date_str = dt.date.today().isoformat()
    out_path = SUBMISSIONS_DIR / f"submission_{method}_{date_str}.csv"
    submission.to_csv(out_path, index=False)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    predict(args.config, args.model)
