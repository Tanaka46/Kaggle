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
    """train側の統計量でfitし、train/test両方に同じ変換をtransformするビルダー。

    encoding="onehot": 名義変数をone-hotエンコーディング(Ridge/Lasso等の線形モデル向け)。
    encoding="ordinal": 名義変数をtrainのカテゴリ集合に基づく整数コードに変換し、one-hotはしない
        (LightGBMのネイティブカテゴリ分割向け。列名は `nominal_cols_` で取得できる)。
    """

    def __init__(self, encoding: str = "onehot") -> None:
        if encoding not in ("onehot", "ordinal"):
            raise ValueError(f"未対応のencoding: {encoding}")
        self.encoding = encoding
        self.neighborhood_lotfrontage_median_: pd.Series | None = None
        self.mode_fill_values_: dict[str, object] = {}
        self.skewed_cols_: list[str] | None = None
        self.nominal_cols_: list[str] | None = None
        self.category_levels_: dict[str, list] = {}
        self.dummy_columns_: list[str] | None = None

    def fit(self, train: pd.DataFrame) -> "FeatureBuilder":
        self.neighborhood_lotfrontage_median_ = train.groupby("Neighborhood")["LotFrontage"].median()
        self.mode_fill_values_ = {col: train[col].mode().iloc[0] for col in MODE_FILL_COLS}

        # 対数変換する列の判定はtrainの歪度のみで決める。test側で別途歪度を測って判定すると、
        # 列によってtrain/testで対数変換の有無がずれ、特徴量のスケールが食い違って予測が壊れる。
        prepared = self._encode_quality(self._add_derived_features(self._impute(train)))
        self.nominal_cols_ = prepared.select_dtypes(include=["object", "string"]).columns.tolist()
        self.category_levels_ = {
            col: sorted(prepared[col].dropna().unique().tolist()) for col in self.nominal_cols_
        }

        candidate_cols = [
            c for c in prepared.select_dtypes(include=[np.number]).columns
            if c not in ("Id", *QUALITY_COLS)
        ]
        skew = prepared[candidate_cols].apply(lambda s: s.skew())
        self.skewed_cols_ = skew[skew.abs() > 0.75].index.tolist()
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

    def _log_transform_skewed(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in self.skewed_cols_:
            df[col] = np.log1p(df[col].clip(lower=0))
        return df

    def _onehot_encode_nominal(self, df: pd.DataFrame) -> pd.DataFrame:
        df = pd.get_dummies(df, columns=self.nominal_cols_)
        bool_cols = df.select_dtypes(include="bool").columns
        df[bool_cols] = df[bool_cols].astype(int)

        if self.dummy_columns_ is None:
            self.dummy_columns_ = df.columns.tolist()
        else:
            df = df.reindex(columns=self.dummy_columns_, fill_value=0)
        return df

    def _ordinal_encode_nominal(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in self.nominal_cols_:
            categories = self.category_levels_[col]
            codes = pd.Categorical(df[col], categories=categories).codes
            # trainに存在しないカテゴリ(-1)は、既知カテゴリの外側の専用コードに割り当てる
            # (LightGBMのcategorical_featureは非負整数を要求するため-1は使えない)
            df[col] = np.where(codes == -1, len(categories), codes)
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._impute(df)
        df = self._add_derived_features(df)
        df = self._encode_quality(df)
        df = self._log_transform_skewed(df)

        if self.encoding == "onehot":
            df = self._onehot_encode_nominal(df)
        else:
            df = self._ordinal_encode_nominal(df)
        return df

    def fit_transform(self, train: pd.DataFrame) -> pd.DataFrame:
        self.fit(train)
        return self.transform(train)
