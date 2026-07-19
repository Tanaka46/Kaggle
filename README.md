# House Prices - Advanced Regression Techniques

Kaggleコンペ「[House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview)」への取り組み用リポジトリ。

- 目標: 銀メダル圏内のスコア
- 評価指標: 予測住宅価格の対数とSalePriceの対数のRMSE

## ディレクトリ構成

```
.
├── configs/           # モデルごとのハイパーパラメータ(YAML)
├── data/
│   ├── raw/          # Kaggleからダウンロードした生データ(train.csv, test.csv等) ※Git管理外
│   ├── interim/      # 前処理の中間生成物 ※Git管理外
│   └── processed/    # モデル投入直前の特徴量データ ※Git管理外
├── notebooks/         # EDA・実験用ノートブック(01_eda.ipynb等、番号で流れを固定)
├── src/
│   ├── data/          # データ読み込み・前処理スクリプト
│   ├── features/      # 特徴量エンジニアリング
│   ├── models/        # train.py(学習) / predict.py(推論) / cv.py(交差検証)
│   └── visualization/ # 可視化スクリプト
├── models/            # 学習済みモデルの保存先 ※Git管理外
├── submissions/       # 提出用csvの保存先 ※Git管理外
├── reports/
│   ├── figures/       # 分析結果の図表
│   └── experiments.csv # 実験ログ(手法・CVスコア・LBスコア・提出ファイル名)
└── .claude/
    ├── agents/         # カスタムエージェント(5種)
    └── skills/         # Skill(4種)
```

## セットアップ

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## データ取得

Kaggle CLIを利用する場合:

```powershell
kaggle competitions download -c house-prices-advanced-regression-techniques -p data\raw
```

`data/raw/` にzipを展開し、`train.csv` / `test.csv` / `sample_submission.csv` / `data_description.txt` を配置する。

## 進め方の方針

1. EDA(データ確認・欠損値・外れ値の把握)
2. 特徴量エンジニアリング(欠損補完、カテゴリ変数エンコーディング、対数変換など)
3. ベースラインモデル作成(Ridge/Lasso等)
4. 勾配ブースティング系モデル(LightGBM/XGBoost/CatBoost)でのチューニング
5. アンサンブル・スタッキング
6. 提出・スコア確認・反復改善

フェーズごとの詳細な指針は `CLAUDE.md` の「フェーズ別ガード」を参照。

## 利用可能なAgents / Skills

Claude Codeでの作業を想定したカスタムエージェント・Skillを用意している。詳細は `CLAUDE.md` を参照。

| Agent | 用途 |
|---|---|
| `data-analyst` | EDA・可視化・特徴量分析 |
| `code-reviewer` | リーク検出・評価指標の実装ミスの横断チェック |
| `submission-validator` | 提出前のcsv検証 |
| `kaggle-researcher` | Kaggle Discussion・公開Notebookの調査 |
| `competition-strategist` | 実験ログの横断分析・次の一手の提案 |

| Skill | 用途 |
|---|---|
| `/exp-new <手法名>` | 新しい実験のconfig雛形作成 + 実験ログへの行追加 |
| `/submit-check <path>` | 提出前検証 |
| `/strategy [観点]` | 横断分析・次の一手の提案 |
| `/survey-kaggle [キーワード]` | Discussion/Notebook調査
