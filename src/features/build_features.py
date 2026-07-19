"""欠損補完・派生特徴量・カテゴリエンコーディングをtrain/testに一貫して適用する。

欠損値の意味付けは data/raw/data_description.txt に基づく(例: PoolQC の
NaN は "プールなし" を意味し、削除ではなく "None" カテゴリとして扱う)。
補完・エンコーディングの統計量(中央値・最頻値・ダミー列の集合)は必ず
train側でfitし、test側はtransformのみに使う(リーク防止)。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NONE_FILL_COLS = [
    "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
    "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "MasVnrType",
]
ZERO_FILL_COLS = [
    "GarageYrBlt", "GarageArea", "GarageCars",
    "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
    "BsmtFullBath", "BsmtHalfBath", "MasVnrArea",
]
MODE_FILL_COLS = [
    "MSZoning", "Utilities", "Exterior1st", "Exterior2nd",
    "Electrical", "KitchenQual", "Functional", "SaleType",
]
QUALITY_MAP = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
QUALITY_COLS = [
    "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC",
    "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC",
]


class FeatureBuilder:
    """train側の統計量でfitし、train/test両方に同じ変換をtransformするビルダー。"""

    def __init__(self) -> None:
        self.neighborhood_lotfrontage_median_: pd.Series | None = None
        self.mode_fill_values_: dict[str, object] = {}
        self.dummy_columns_: list[str] | None = None

    def fit(self, train: pd.DataFrame) -> "FeatureBuilder":
        self.neighborhood_lotfrontage_median_ = train.groupby("Neighborhood")["LotFrontage"].median()
        self.mode_fill_values_ = {col: train[col].mode().iloc[0] for col in MODE_FILL_COLS}
        return self

    def _impute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[NONE_FILL_COLS] = df[NONE_FILL_COLS].fillna("None")
        df[ZERO_FILL_COLS] = df[ZERO_FILL_COLS].fillna(0)

        fallback_median = self.neighborhood_lotfrontage_median_.median()
        neighborhood_medians = df["Neighborhood"].map(self.neighborhood_lotfrontage_median_)
        df["LotFrontage"] = df["LotFrontage"].fillna(neighborhood_medians).fillna(fallback_median)

        for col, value in self.mode_fill_values_.items():
            df[col] = df[col].fillna(value)
        return df

    @staticmethod
    def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
        df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
        df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]
        df["TotalBath"] = (
            df["FullBath"] + 0.5 * df["HalfBath"] + df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"]
        )
        return df

    @staticmethod
    def _encode_quality(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in QUALITY_COLS:
            df[col] = df[col].map(QUALITY_MAP)
        return df

    @staticmethod
    def _log_transform_skewed(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        candidate_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in ("Id", *QUALITY_COLS)
        ]
        skew = df[candidate_cols].apply(lambda s: s.skew())
        skewed_cols = skew[skew.abs() > 0.75].index
        for col in skewed_cols:
            df[col] = np.log1p(df[col].clip(lower=0))
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._impute(df)
        df = self._add_derived_features(df)
        df = self._encode_quality(df)
        df = self._log_transform_skewed(df)

        nominal_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
        df = pd.get_dummies(df, columns=nominal_cols)
        bool_cols = df.select_dtypes(include="bool").columns
        df[bool_cols] = df[bool_cols].astype(int)

        if self.dummy_columns_ is None:
            self.dummy_columns_ = df.columns.tolist()
        else:
            df = df.reindex(columns=self.dummy_columns_, fill_value=0)
        return df

    def fit_transform(self, train: pd.DataFrame) -> pd.DataFrame:
        self.fit(train)
        return self.transform(train)
