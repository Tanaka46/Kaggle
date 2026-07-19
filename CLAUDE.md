# プロジェクト概要

Kaggleコンペ「House Prices - Advanced Regression Techniques」の予測モデル作成プロジェクト。
目標は銀メダル圏内のスコア(RMSE(log)ベース)。

- コンペURL: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview
- 評価指標: 予測値と実測値の対数のRMSE

## ディレクトリ構成

- `data/raw/`: Kaggle生データ(Git管理外)
- `data/interim/`, `data/processed/`: 前処理済みデータ(Git管理外)
- `notebooks/`: EDA・実験用
- `src/data/`, `src/features/`, `src/models/`, `src/visualization/`: 再利用可能なスクリプト
- `models/`: 学習済みモデル(Git管理外)
- `submissions/`: 提出用csv(Git管理外)
- `reports/figures/`: 可視化結果

## 作業方針

- ノートブックでの試行錯誤結果は、再利用可能な処理を `src/` 配下のスクリプト/関数に切り出す
- 提出ファイルは `submissions/` に `submission_{手法}_{スコアor日付}.csv` のような命名で保存する
- 大きなデータファイルやモデルバイナリはコミットしない(.gitignore済み)
