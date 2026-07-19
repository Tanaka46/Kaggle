# プロジェクト概要

Kaggleコンペ「House Prices - Advanced Regression Techniques」の予測モデル作成プロジェクト。
目標は銀メダル圏内のスコア(RMSE(log)ベース)。

- コンペURL: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview
- 評価指標: 予測値と実測値の対数のRMSE

## ディレクトリ構成

- `data/raw/`: Kaggle生データ(Git管理外)
- `data/interim/`, `data/processed/`: 前処理済みデータ(Git管理外)
- `notebooks/`: EDA・実験用(`01_eda.ipynb`〜`05_ensemble.ipynb`、番号で流れを固定)
- `src/data/`, `src/features/`, `src/models/`, `src/visualization/`: 再利用可能なスクリプト
- `configs/`: モデルごとのハイパーパラメータ(YAML)
- `models/`: 学習済みモデル(Git管理外)
- `submissions/`: 提出用csv(Git管理外)
- `reports/figures/`: 可視化結果
- `reports/experiments.csv`: 実験ログ(手法・CVスコア・LBスコア・提出ファイル名を1行ずつ記録)
- `.claude/agents/`, `.claude/skills/`: Claude Code用のカスタムエージェント・Skill

## 作業方針

- ノートブックでの試行錯誤結果は、再利用可能な処理を `src/` 配下のスクリプト/関数に切り出す
- 提出ファイルは `submissions/` に `submission_{手法}_{スコアor日付}.csv` のような命名で保存する
- 大きなデータファイルやモデルバイナリはコミットしない(.gitignore済み)
- 新しい実験を始めたら `reports/experiments.csv` に必ず行を追加する(過去分は編集せず追記のみ)

## フェーズ別ガード(いきなりアンサンブルしない)

**コンペには段階がある。終盤の最適化を序盤に持ち込むとリソースを溶かす。**

| フェーズ | 目安 | やる | **やらない** |
|---|---|---|---|
| 序盤 | 〜30% | EDA・欠損値/外れ値の把握・fold設計・1つの strong baseline(Ridge/Lasso)・動くsubmissionパイプライン | アンサンブル、重いハイパラ探索、スタッキング |
| 中盤 | 30-70% | 複数モデル(Ridge/LightGBM/XGBoost/CatBoost)を個別にチューニング・エラー分析・特徴量追加 | 本格的なアンサンブル |
| 終盤 | 70%- | アンサンブル・スタッキング・最終提出の選定・LB shake評価 | 新規特徴量の大規模追加、新モデル種の導入 |

進捗の目安は `reports/experiments.csv` の実験数・`notebooks/` の進み具合から判断する。フェーズ判定に迷ったら `/strategy` を使う。

## アイデア提案の原則(堅実+爆発)

アプローチやアイデアを提案するときは、必ず「堅実案」と「爆発案」の両方を出す。

- **堅実案**: 既知の手法・定石・段階的改善。確実にスコアが上がる見込みがあるもの
- **爆発案**: 常識外れ、異分野からの転用。失敗リスクは高いが当たれば大きいもの

## エラーハンドリングの原則(握りつぶさない)

- `except: pass` や広い `except Exception:` でデフォルト値・NaN・空を静かに返すことを禁止する
- `try/except` を使ってよいのは、回復可能な失敗・具体的な例外型・ログ記録・リトライ上限つきの場合のみ
- データ形状・前処理・指標・提出形式の不整合は `assert` で早期に落とす(隠さない)

## エラー分析の原則(スコアの前に出力を見る)

- スコアを上げようとする前に、まず予測値 vs 実測値の残差を目視で確認する(外れ値・特定価格帯での系統的な誤差など)
- 誤差の傾向を把握してから、前処理・特徴量・モデル選択への対策を打つ
- 数字だけを見てパラメータを闇雲に変える探索はしない

## Custom Agents(`.claude/agents/`)

| Agent | Model | 用途 |
|---|---|---|
| `data-analyst` | sonnet | EDA・可視化・特徴量分析 |
| `code-reviewer` | opus | リーク検出・評価指標の実装ミス・例外の握りつぶしを横断チェック |
| `submission-validator` | sonnet | 提出前のcsv検証(列・行数・欠損・値の範囲) |
| `kaggle-researcher` | sonnet | Kaggle Discussion・公開Notebookの解法調査 |
| `competition-strategist` | opus | `reports/experiments.csv` とnotebooksの横断分析・次の一手の提案 |

## 利用可能なSkills(`.claude/skills/`)

- `/exp-new <手法名>` — 新しい実験のconfig雛形作成 + `reports/experiments.csv` への行追加
- `/submit-check <path>` — 提出前検証を `submission-validator` に委譲。提出直前に必ず使う
- `/strategy [観点]` — `competition-strategist` による横断分析。CVスコア頭打ち時や週次で使う
- `/survey-kaggle [キーワード]` — `kaggle-researcher` によるDiscussion/Notebook調査
