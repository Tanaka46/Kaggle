---
name: submit-check
description: submissions/配下の提出用csvを提出前に検証する。提出直前に必ず使う。
---

指定された、または直近に生成された `submissions/*.csv` を提出前に検証する。

## 手順

1. 対象ファイルのパスを引数から受け取る(未指定なら `submissions/` の最新ファイルを対象にする)
2. `submission-validator` エージェントに検証を委譲する
3. エージェントの検証結果(PASS/FAIL)をそのままユーザーに提示する
4. FAILがあれば、提出せず先に修正するようユーザーに伝える。修正後は再度このSkillを実行するよう促す
