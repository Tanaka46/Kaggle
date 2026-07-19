# 進捗まとめ(2026-07-19時点)

House Prices - Advanced Regression Techniques コンペの取り組み状況。生の数値ログは `reports/experiments.csv` を参照。

## 1. 環境・プロジェクトセットアップ

- ディレクトリ構成: `data/`(raw/interim/processed) `notebooks/`(01〜05) `src/`(data/features/models/visualization) `configs/` `models/` `submissions/` `reports/`(README.md参照)
- Git 2.55 / GitHub CLI 2.96 をwingetで導入し、`gh auth login`(ブラウザ認証)+ `gh auth setup-git` でcredential helperを設定。ローカルリポジトリを初期化し `Tanaka46/Kaggle`(main ブランチ)にpush
- Python 3.12.10 + `.venv` 仮想環境を作成し、`requirements.txt`(numpy, pandas, scikit-learn, scipy, matplotlib, seaborn, lightgbm, xgboost, catboost, optuna, jupyter, kaggle, pyyaml, joblib)をインストール
- Kaggle API(`kaggle.json`)を設定し、CLIで `data/raw/` に `train.csv`(1460行) `test.csv`(1459行) `sample_submission.csv` `data_description.txt` を取得
- `.claude/agents/`(data-analyst, code-reviewer, submission-validator, kaggle-researcher, competition-strategist の5種)・`.claude/skills/`(exp-new, submit-check, strategy, survey-kaggle の4種)を追加
- `CLAUDE.md` にフェーズ別ガード(序盤/中盤/終盤でやる・やらないこと)、堅実+爆発の提案原則、エラーハンドリング原則、エラー分析の原則を整備

## 2. EDA(`01_eda.ipynb`)

- データ形状: train 1460行×81列 / test 1459行×80列(test は `SalePrice` なし)
- 欠損値割合(上位): `PoolQC` 99.5% / `MiscFeature` 96.3% / `Alley` 93.8% / `Fence` 80.8% / `MasVnrType` 59.7% / `FireplaceQu` 47.3% / `LotFrontage` 17.7% / `GarageYrBlt`他Garage系 5.5% / Bsmt系 2.5〜2.6%。`data_description.txt` を確認し、これらの多くは「該当設備なし」を意味する構造的欠損と判断(単純削除ではなく明示的な補完方針を採用)
- ターゲット `SalePrice`: 平均180,921 / 中央値163,000 / 最小34,900 / 最大755,000。歪度1.88(右裾が長い分布)。`log1p` 変換後の歪度はほぼ0まで縮小し、線形モデルの前提(誤差の正規性)に近づくことを確認
- `SalePrice` との相関(数値特徴量、上位順): `OverallQual` 0.79 / `GrLivArea` 0.71 / `GarageCars` 0.64 / `GarageArea` 0.62 / `TotalBsmtSF` 0.61 / `1stFlrSF` 0.61 / `FullBath` 0.56 / `TotRmsAbvGrd` 0.53 / `YearBuilt` 0.52 / `YearRemodAdd` 0.51
- 既知の外れ値パターン: `GrLivArea`(延床面積)対`SalePrice`の散布図で、`GrLivArea > 4000` かつ `SalePrice < 300000` の2件(Id 524: GrLivArea 4676 / SalePrice 184,750、Id 1299: GrLivArea 5642 / SalePrice 160,000)を検出。広い割に安いという明らかな異常値のため、学習データから除外する方針とした
- カテゴリ変数のカーディナリティ: `Neighborhood` が最多で25水準、次いで `Exterior2nd` 16水準、`Exterior1st` 15水準

## 3. 特徴量エンジニアリング(`02_feature_engineering.ipynb`, `src/features/build_features.py`)

`FeatureBuilder` クラス(sklearn風の `fit` / `transform` / `fit_transform`)を実装。**すべての補完統計量・エンコーディング規則はtrain側のみでfitし、testにはtransformのみを適用**することでリークを防止している。

### 3.1 外れ値除外
`GrLivArea > 4000 かつ SalePrice < 300000` の2件を除外し、train は1460→1458行になる。

### 3.2 欠損値補完(`_impute`)
- **"None"補完(構造的欠損=設備なし)**: `PoolQC, MiscFeature, Alley, Fence, FireplaceQu, GarageType, GarageFinish, GarageQual, GarageCond, BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2, MasVnrType` の15列
- **0補完(数量が0=該当なし)**: `GarageYrBlt, GarageArea, GarageCars, BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, BsmtFullBath, BsmtHalfBath, MasVnrArea` の10列
- **近隣別中央値補完**: `LotFrontage` は `Neighborhood` ごとのtrain中央値で補完(train全体の中央値をフォールバック)
- **最頻値補完**: `MSZoning, Utilities, Exterior1st, Exterior2nd, Electrical, KitchenQual, Functional, SaleType` の8列(train側の最頻値をtest側にも適用)

### 3.3 派生特徴量(`_add_derived_features`)
- `TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF`(延床面積の合計)
- `HouseAge = YrSold - YearBuilt`(売却時点での築年数)
- `RemodAge = YrSold - YearRemodAdd`(売却時点でのリフォーム後経過年数)
- `TotalBath = FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath`(浴室数の統合指標)

### 3.4 品質系の序数エンコーディング(`_encode_quality`)
`ExterQual, ExterCond, BsmtQual, BsmtCond, HeatingQC, KitchenQual, FireplaceQu, GarageQual, GarageCond, PoolQC` の10列を `{None:0, Po:1, Fa:2, TA:3, Gd:4, Ex:5}` で整数化(単純なone-hotより順序情報を保持できる)

### 3.5 歪度に基づく対数変換(`_log_transform_skewed`)
`Id`と品質系10列を除いた数値列についてtrain側の歪度を計算し、`|skew| > 0.75` の列に `log1p(clip(lower=0))` を適用。**対象列のリストはfit時にtrain側でのみ確定し、test側にも同じ列リストを適用する**(後述のバグ参照)。

### 3.6 名義変数のエンコーディング(2方式を用意)
- `encoding="onehot"`(Ridge/Lasso向け): `pd.get_dummies` で展開。列数は266列(train統計量でダミー列集合を固定し、testで見なかった値・値がない列は`reindex(fill_value=0)`で揃える)
- `encoding="ordinal"`(LightGBM向け): 33個の名義列(`MSZoning, Street, Alley, LotShape, LandContour, Utilities, LotConfig, LandSlope, Neighborhood, Condition1, Condition2, BldgType, HouseStyle, RoofStyle, RoofMatl, Exterior1st, Exterior2nd, MasVnrType, Foundation, BsmtExposure, BsmtFinType1, BsmtFinType2, Heating, CentralAir, Electrical, Functional, GarageType, GarageFinish, PavedDrive, Fence, MiscFeature, SaleType, SaleCondition`)を、trainのカテゴリ集合に基づく整数コードに変換(列数84列)。test側で未知のカテゴリが出た場合は「既知カテゴリ数」を専用コードとして割り当て(LightGBMの`categorical_feature`は非負整数必須のため-1は使えない)

出力: `data/processed/train_features.csv`(1458行×266列)/ `test_features.csv`(1459行×266列)/ `train_features_lgbm.csv`・`test_features_lgbm.csv`(84列)/ `categorical_columns.json`(LightGBM用カテゴリ列名リスト)

### 3.7 発見・修正したバグ(重要)

初期実装では、対数変換対象列の歪度判定を `_log_transform_skewed` 内でtrain/testそれぞれ独立に計算していた。これにより以下4列で判定がズレていた:

| 列 | train歪度 | test歪度 | train判定 | test判定 |
|---|---|---|---|---|
| `LotFrontage` | 1.548 | 0.623 | 対数変換する | 対数変換しない |
| `TotalBsmtSF` | 0.512 | 0.805 | 対数変換しない | 対数変換する |
| `TotRmsAbvGrd` | 0.661 | 0.843 | 対数変換しない | 対数変換する |
| `Fireplaces` | 0.632 | 0.820 | 対数変換しない | 対数変換する |

CVスコアはtrain内で完結するKFoldのため影響を受けず(バグ修正前後で0.1236/0.1203と完全一致)、**提出して初めて表面化する種類のバグ**だった。実際、修正前のLasso提出は最大予測331万円(train実売価格の最大75.5万円の4倍超)、OverallQual=3・築1953年の小さな家(Id 2600)にまで220万円という異常値を出しており、`submission-validator`相当の検証(列・行数・Id整合・値域チェック)で発覚した。

修正方法: 歪度判定を `fit()` 内でtrain側のみで一度だけ計算して `self.skewed_cols_` に固定し、`transform()` はtrain/test問わずこのリストを参照するように変更。修正後、最大予測は885,903円(testに含まれる最大級の延床面積5095平方フィートの家)まで低下し、trainの分布と整合する妥当な範囲になった。

## 4. ベースラインモデル(`03_baseline_model.ipynb`, `src/models/train.py`)

### CV手法
`src/models/cv.py` の `get_folds()`: `sklearn.model_selection.KFold(n_splits=5, shuffle=True, random_state=42)` によるプレーンな5分割(グループ構造・時系列性は確認済みで不要と判断)。目的変数は `log1p(SalePrice)` で学習し、out-of-fold予測を集約した `rmse()`(`sqrt(mean((y_true-y_pred)**2))`)でCVスコアを算出(コンペの評価指標である対数RMSEと一致)。

### Ridge(`configs/ridge.yaml`)
`sklearn.linear_model.Ridge(alpha=10.0, random_state=42)`。`StandardScaler` とパイプライン化(`make_pipeline`)し、fold内のtrainのみでスケーラをfitしてリークを防止(one-hotの0/1と面積等の数値のスケール差を揃えないと正則化が不均等に効くため)。**CV RMSE(log) = 0.12360**

### Lasso(`configs/lasso.yaml`)
`sklearn.linear_model.Lasso(alpha=0.0005, random_state=42, max_iter=50000)`。Ridgeと同様にStandardScalerとパイプライン化。**CV RMSE(log) = 0.12027(現状ベスト)**

### 実装時に修正したバグ
`PROCESSED_DIR`/`MODELS_DIR` が `Path("data/processed")` のような**カレントディレクトリ相対パス**だったため、ノートブック(cwdが`notebooks/`)から呼び出すと `FileNotFoundError` になっていた。`Path(__file__).resolve().parents[2]` を起点にした絶対パスに変更して解消(`src/models/train.py` `src/models/predict.py` 共通)。

## 5. 初回提出

- `python -m src.models.predict --config configs/lasso.yaml --model models/lasso.pkl` で `submissions/submission_lasso_2026-07-19.csv` を生成
- 提出前検証: 列名(`Id`,`SalePrice`)・行数(1459)・`test.csv`とのId完全一致・重複なし・欠損なし・`SalePrice`が正の値・trainの価格帯との整合性を確認(この過程で上記3.7のバグを発見)
- **Kaggle提出結果: Public LB 0.13287 / 順位2383**。CV(0.1203)とLBのスコア差は妥当な範囲

## 6. GBDT系モデル(`04_gbdt_tuning.ipynb`)

参考にした公開モデル: [jek1wantaufik/buddy (ScikitLearn/house)](https://www.kaggle.com/models/jek1wantaufik/buddy/ScikitLearn/house)。数値=中央値補完、カテゴリ=最頻値補完+one-hot、**GradientBoostingRegressor**、ターゲット対数変換、というパイプライン構成が要点。前処理面は自前実装(3節)が既に上回っていたため、差分の「GBDT導入」をLightGBMで実装した。

### configs/lightgbm.yaml
`objective=regression, metric=rmse, learning_rate=0.05, num_leaves=31, max_depth=-1, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5, min_data_in_leaf=20, lambda_l1=0.0, lambda_l2=0.0, seed=42` / `training: num_boost_round=5000, early_stopping_rounds=100, n_splits=5`

### 学習方式
fold毎に `eval_set=[(X_valid, y_valid)]` と `lgb.early_stopping(100, verbose=False)` を指定してfold内の検証データで早期終了させ、各foldの `best_iteration_` を記録。最終モデルは、全foldの `best_iteration_` の平均値を `n_estimators` として全trainデータで再学習(early stoppingなし)して保存。

### 検証1: one-hotエンコーディング(266列)のまま投入
**CV RMSE(log) = 0.12193**。Lasso(0.12027)に届かず

### 検証2: ネイティブカテゴリ分割(84列、`categorical_feature`指定)に作り直し
one-hotだと1つのカテゴリごとに1軸の2値分割にしかならず、LightGBMの強みである「カテゴリを直接分割に使う」機能を活かせていないと判断。`FeatureBuilder(encoding="ordinal")` で33列の名義変数を整数コード化し、`categorical_feature=<33列のリスト>` を `fit()` に渡す方式に変更。
**CV RMSE(log) = 0.12103**(one-hot版から改善したが、依然Lassoに未到達)

### 考察
1458行という少数データに対し、LightGBMのデフォルト寄りパラメータ(`num_leaves=31`)がやや過剰(過学習しやすい)と推測。ハイパーパラメータ調整(正則化を強める方向)の余地がある。

## 現状のベスト

**Lasso**(`configs/lasso.yaml`): CV RMSE(log) 0.1203 / Public LB 0.13287 / 順位2383

## 次の候補(未着手)

- LightGBMのハイパーパラメータ調整(Optuna、`num_leaves`/`min_data_in_leaf`等の正則化強化)
- Lasso × LightGBMのアンサンブル(単純平均 or スタッキング)— フェーズガード上は序盤〜中盤のため時期尚早、複数モデルが出揃ってから検討
- XGBoost/CatBoostの追加検証
