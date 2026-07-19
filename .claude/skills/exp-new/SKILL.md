---
name: exp-new
description: 新しい実験(手法)を始める際の定型作業をまとめて行う。configの雛形作成とexperiments.csvへの行追加を自動化する。
---

新しい実験を始めるときに使う。引数として手法名(例: `xgboost`, `catboost`, `stacking`)を受け取る。

## 手順

1. `configs/` 配下の既存ファイル(`lightgbm.yaml`, `ridge.yaml`)を参考に、`configs/<手法名>.yaml` を作成する。`model`, `params`, `training`(`n_splits`, `target_log_transform`)のキーを踏襲する
2. `reports/experiments.csv` に、`date`(今日の日付), `method`(手法名), `config`(configファイルパス)を埋めた空行を追記する。`cv_rmse`/`lb_score`/`submission_file`/`notes` は空欄のままにする
3. 対応する学習コードが `src/models/train.py` の分岐で必要な場合は、その旨をユーザーに伝える(自動生成はしない。汎用スケルトンに手法固有ロジックを機械的に追加すると壊れやすいため)
4. `notebooks/04_gbdt_tuning.ipynb` など、関連しそうな既存ノートブックがあれば教えてセルの追加を提案する

## 注意

- 既存の `notebooks/01_eda〜05_ensemble` の命名規則・番号は変更しない
- `reports/experiments.csv` は追記のみ。既存行は編集しない(実験履歴として保持する)
