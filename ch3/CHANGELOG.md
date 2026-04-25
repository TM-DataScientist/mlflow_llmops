# CHANGELOG

## 2026-04-25

### `scraped_data.json` を中間生成物として扱うよう変更

MLflow公式ドキュメントのクロール結果をリポジトリに含めないよう、`ch3/scraped_data.json` を削除し、今後は `.gitignore` で除外するようにした。

また、`ch3/scripts/web_ingest.py` は他章と同様に、Milvusへの取り込み後に `scraped_data.json` を削除する挙動に揃えた。

## 2026-04-23 (4)

### ch7/serving/ 各スクリプトへの詳細な日本語コメント追加

`ch7/serving/` ディレクトリの全スクリプトに、動作がわかるよう詳細な日本語コメントを追加した。

| ファイル | 主な追加コメント |
|---|---|
| `agent.py` | @invokeデコレータの役割、Responses APIメッセージ形式の解説、ステートレスサービングの説明、リクエスト処理の各ステップ |
| `log_model.py` | models-from-codeパターンの説明、championエイリアスの意味、log_model()とレジストリ自動登録の関係、verify_model()のThread/configurable解説 |
| `model_code.py` | models-from-codeパターンの詳細説明、set_model()の役割、LangGraphのCompiledGraphについて |
| `start_server.py` | AgentServerとFastAPI/uvicornの関係、appをモジュールレベルで公開する理由、app_import_stringによるマルチワーカーサポートの説明 |
| `eval_serving.py` | 各スコアラー（RelevanceToQuery/Safety/Guidelines）の評価内容、asyncio.run()による非同期/同期ブリッジの説明、EVAL_DATASETの入力形式解説 |

## 2026-04-23 (3)

### ch3 と ch7 の `make ingest` 動作比較

両章の `scripts/web_ingest.py` を比較した。コアロジック（Scrapyクロール → HTMLクリーニング → チャンク分割 → OpenAI Embeddings → Milvus保存）は同一。差異は以下の2点：

| 項目 | ch3 | ch7 |
|------|-----|-----|
| DBの保存先 | `PROJECT_ROOT` 基準の絶対パス（スクリプト位置から `ch3/data/milvus.db`） | カレントディレクトリ基準の相対パス（`./data/milvus.db`） |
| `scraped_data.json` | 取り込み後もファイルを残す（`os.remove` をコメントアウト） | 取り込み後に削除する |

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
