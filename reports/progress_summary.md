# 進捗まとめ(2026-07-19時点)

House Prices - Advanced Regression Techniques コンペの取り組み状況。詳細な数値は `reports/experiments.csv` を参照。

## 1. 環境・プロジェクトセットアップ

- ディレクトリ構成: `data/` `notebooks/` `src/` `configs/` `models/` `submissions/` `reports/`(README.md参照)
- Git/GitHub CLIをwingetで導入し、`Tanaka46/Kaggle` にリポジトリを接続・push
- Python 3.12 + venv + `requirements.txt` の依存関係をセットアップ
- Kaggle CLIで `data/raw/` にtrain.csv/test.csv等を取得
- `.claude/agents/`(5種)・`.claude/skills/`(4種)を追加し、`CLAUDE.md`にフェーズ別ガード・提案原則・エラーハンドリング原則を整備

## 2. EDA(`01_eda.ipynb`)

- train 1460行×81列 / test 1459行×80列
- 高欠損列(`PoolQC` `MiscFeature` `Alley` `Fence` `MasVnrType` `FireplaceQu`等)は「設備なし」を意味すると判断(削除せず補完方針)
- `SalePrice` は右に歪んだ分布(歪度1.88) → `log1p`変換で正規分布に近づくことを確認
- `SalePrice`との相関上位: `OverallQual`(0.79) `GrLivArea`(0.71) `GarageCars`(0.64) `GarageArea`(0.62) `TotalBsmtSF`(0.61)
- 既知の外れ値: `GrLivArea > 4000` かつ低価格の2件(Id 524, 1299)を特定 → 学習時に除外

## 3. 特徴量エンジニアリング(`02_feature_engineering.ipynb`, `src/features/build_features.py`)

`FeatureBuilder`クラスを実装(train統計量でfit → train/testに同一transform、リーク防止)。

- 欠損値: 「設備なし」系はNone/0で補完、`LotFrontage`は近隣(Neighborhood)ごとの中央値で補完(train統計量のみ使用)
- 派生特徴量: `TotalSF` `HouseAge` `RemodAge` `TotalBath`
- 品質系10列を序数エンコーディング(Ex=5〜Po=1)
- 歪度|0.75|超の数値列に`log1p`変換
- 名義変数のエンコーディングを2種類用意:
  - `encoding="onehot"`: Ridge/Lasso向け(266列に展開)
  - `encoding="ordinal"`: LightGBM向け(整数コード化し`categorical_feature`でネイティブ分割)

### 発見・修正したバグ

対数変換対象列の歪度判定をtrain/testそれぞれ独立に計算していたため、`LotFrontage` `TotalBsmtSF` `TotRmsAbvGrd` `Fireplaces` の4列でtrain/test間の変換有無がズレていた。CVスコアには影響しない(train内で完結するため)が、提出時に最大331万(train実売価格の最大75.5万の4倍超)という異常な予測を生んでいた。歪度判定をtrainのみでfitして固定する方式に修正し解消(提出前検証で発覚)。

## 4. ベースラインモデル(`03_baseline_model.ipynb`)

Ridge/LassoはStandardScalerとのパイプライン化(fold内fitでリーク防止)。

| 手法 | CV RMSE(log) |
|---|---|
| Ridge | 0.1236 |
| **Lasso** | **0.1203**(現状ベスト) |

## 5. 初回提出

- `submission_lasso_2026-07-19.csv` を提出 → **Public LB 0.13287 / 順位2383**
- CVとLBのスコア差(0.120 vs 0.133)は妥当な範囲

## 6. GBDT系モデル(`04_gbdt_tuning.ipynb`)

参考: [jek1wantaufik/buddy (ScikitLearn/house)](https://www.kaggle.com/models/jek1wantaufik/buddy/ScikitLearn/house) — GradientBoostingRegressor + 対数変換ターゲットのパイプライン構成を参考にLightGBMを追加。

| 手法 | CV RMSE(log) |
|---|---|
| LightGBM(one-hot) | 0.1219 |
| LightGBM(ネイティブカテゴリ) | 0.1210(改善したがLassoに未到達) |

少数データ(1458行)にLightGBMのデフォルトパラメータがやや過剰と推測。

## 現状のベスト

Lasso(CV 0.1203 / LB 0.13287)。

## 次の候補

- LightGBMのハイパーパラメータ調整(Optuna、`num_leaves`等の正則化強化)
- Lasso × LightGBMのアンサンブル(平均 or スタッキング)
- XGBoost/CatBoostの追加検証
