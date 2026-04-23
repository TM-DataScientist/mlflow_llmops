# CHANGELOG

## 2026-04-23 (2)

### prompts/ 全スクリプトのモデルを gpt-5-nano-2025-08-07 に変更

対象ファイルのモデル指定をすべて `gpt-5-nano-2025-08-07` に統一した。

| ファイル | 変更箇所 |
|---|---|
| `04_evaluate_prompt.py` | `model=` × 2 |
| `05_optimize_metaprompt.py` | `model=` × 2, `reflection_model=` × 1 |
| `06_optimize_gepa.py` | `model=` × 2, `reflection_model=` × 1 |
| `08_model_config.py` | `model_name=` × 1 |
| `09_structured_output.py` | `model=` × 1 |

## 2026-04-23

### prompts/ 各スクリプトへの日本語コメント追加

各スクリプトの処理の流れがわかるよう、主要な処理ステップに日本語コメントを追加した。

対象ファイル:
- `prompts/01_register_prompt.py` — MLflow接続、プロンプト登録、テンプレート変数の説明
- `prompts/02_version_update.py` — 改善プロンプトの背景、同名登録による新バージョン作成、不変性確認
- `prompts/03_alias_management.py` — エイリアス設定の意味、@エイリアス記法、昇格/ロールバックの仕組み
- `prompts/04_evaluate_prompt.py` — .env読み込み、autolog、predict_fn/スコアラーの役割、評価フロー
- `prompts/05_optimize_metaprompt.py` — MetaPromptOptimizerの動作(1回のLLM呼び出しで構造改善)
- `prompts/06_optimize_gepa.py` — GepaPromptOptimizerの動作(反復最適化、max_metric_callsによるコスト制御)
- `prompts/07_deploy_lifecycle.py` — 段階的デプロイ(staging→production)、ロールバック、タグによるガバナンス
- `prompts/08_model_config.py` — model_configでモデルパラメータをプロンプトと一緒に保存する意義
- `prompts/09_structured_output.py` — Pydanticモデルによる出力スキーマ定義、beta.chat.completions.parseの使い方
