"""6.2節: プロンプトの登録

QAエージェントのシステムプロンプトをPrompt Registryに登録する。
第4章でコード内にハードコードされていたプロンプトを、レジストリに移行する最初のステップ。

実行: make register
前提: MLflow Tracking Serverが起動していること (uv run mlflow server --port 5000)
"""

import mlflow

# MLflowのトラッキングサーバーに接続する
mlflow.set_tracking_uri("http://localhost:5000")

# 第4章のQAエージェントで使用していたシステムプロンプト
# コード内にハードコードされていたものをPrompt Registryに移行する
initial_prompt = """
あなたはMLflowに関する質問に答える専門アシスタントです。
ユーザーの質問に対して、必要に応じてドキュメント検索やWeb検索を使用して、
正確で詳細な回答を提供してください。

回答の際は以下の点に注意してください：
- 公式ドキュメントに基づいた正確な情報を提供する
- 必要に応じてコード例を含める
- 情報源を明記する
"""

# プロンプトをPrompt Registryに登録する(初回登録でversion 1が作成される)
# tags引数で作成者・用途・言語などのメタデータを付与できる
prompt = mlflow.genai.register_prompt(
    name="qa-agent-system-prompt",
    template=initial_prompt,
    commit_message="QAエージェントの初期プロンプト",
    tags={
        "author": "alice@example.com",
        "task": "qa",
        "language": "ja",
    },
)

print(f"Registered: {prompt.name} (version {prompt.version})")

# テンプレート変数のデモ(原稿の補足説明)
# {{ }} で囲んだ変数はprompt.format(変数名=値)で実行時に埋め込める
print(f"\nテンプレート変数の例:")
print(f"  二重中括弧 {{{{ }}}} で変数を定義し、prompt.format()で埋め込みます")
