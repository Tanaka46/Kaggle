---
name: survey-kaggle
description: House Prices関連のKaggle Discussion・公開Notebookを調査する。特徴量エンジニアリングや上位解法のアイデアが欲しい時に使う。
---

Kaggle の Discussion / Public Notebook から、特徴量エンジニアリングやモデリングのアイデアを調査する。キーワード(例: `feature engineering`, `stacking`, `outlier`)を引数として受け取る。

## 手順

1. `kaggle-researcher` エージェントに調査を委譲する(メインの会話コンテキストを汚さないよう、独立したタスクとして実行する)
2. エージェントの調査結果(出典・要点・本プロジェクトへの適用案)をユーザーに提示する
3. 適用案のうち実装に値するものがあれば、`src/features/` や `configs/` への反映方針をユーザーと確認する
