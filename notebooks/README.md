# notebooks 命名規則

`README.md`の「進め方の方針」の各ステップに対応させ、`NN_内容.ipynb`の形式で番号を振る。

- `01_eda.ipynb`: EDA(データ確認・欠損値・外れ値の把握)
- `02_feature_engineering.ipynb`: 特徴量エンジニアリング
- `03_baseline_model.ipynb`: ベースラインモデル作成(Ridge/Lasso等)
- `04_gbdt_tuning.ipynb`: 勾配ブースティング系モデルのチューニング
- `05_ensemble.ipynb`: アンサンブル・スタッキング

ノートブックで固まった処理は `src/` 配下のスクリプトに切り出す。
