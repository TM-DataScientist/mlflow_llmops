"""
第8章 - 8.1 本番トレーシングの基本設定

本番環境でのトレーシング設定と、メタデータの追加方法を確認します。

【このスクリプトで学ぶこと】
  1. 非同期ログによる低レイテンシートレーシング設定
  2. トレースへのユーザーID・セッションIDなどのメタデータ付与
  3. filter_stringを使ったトレース検索・絞り込み

実行方法:
  make tracing
  または
  uv run python monitoring/01_tracing_setup.py
"""

from dotenv import load_dotenv

# .envファイルからOPENAI_API_KEYなどの環境変数を読み込む
load_dotenv()

import mlflow
import os
import uuid
from openai import OpenAI

# === 1. 本番向け設定 ===

# 非同期ログ記録を有効化（本番環境推奨）
# "true" に設定すると、トレース書き込みをバックグラウンドスレッドで実行するため
# LLM呼び出しのレイテンシーへの影響を最小化できる。
# 同期ログ（デフォルト）では LLM呼び出し完了後にトレース書き込みが完了するまで
# 次の処理が始まらないのに対し、非同期では即座に次の処理に進む。
os.environ["MLFLOW_ENABLE_ASYNC_TRACE_LOGGING"] = "true"

# OpenTelemetryのサービス名を設定する
# この名前はトレースのメタデータとして記録され、複数サービスのトレースを
# 混在させた場合に "qa-agent" からのトレースを識別するために使用される。
os.environ["OTEL_SERVICE_NAME"] = "qa-agent"

# 実験（エクスペリメント）を設定する
# 同じ実験名を使い続けることで、トレースが1箇所に集約され分析しやすくなる
mlflow.set_experiment("ch8-monitoring-quickstart")

# OpenAI APIの自動トレーシングを有効化する
# autolog()を呼ぶだけで、以降のOpenAI API呼び出しが自動的にトレースされる。
# プロンプト内容・レスポンス・トークン数・レイテンシーなどが記録される。
mlflow.openai.autolog()

client = OpenAI()


# === 2. メタデータ付きのLLM呼び出し ===

# @mlflow.trace デコレータを付けることで、この関数全体が1つのトレースとして記録される
# 関数の入力引数・戻り値・実行時間などが自動的にスパンとして記録される
@mlflow.trace
def handle_request(message: str, user_id: str, session_id: str) -> str:
    """メタデータを付与したリクエスト処理。

    本番環境では、ユーザーID・セッションIDを付与することで
    問題発生時のデバッグ（どのユーザーのどのセッションで問題が起きたか）や
    ユーザー別の利用傾向分析が可能になる。

    Args:
        message: ユーザーからの質問テキスト
        user_id: ユーザー識別子（例: "user-1"）
        session_id: セッション識別子（UUID）
    """
    # 現在実行中のトレースにタグ（メタデータ）を追加する
    # タグは MLflow UI で確認でき、filter_string での検索条件にも使える
    mlflow.update_current_trace(
        tags={
            # 規約: "mlflow.trace.user" はユーザー識別子の標準タグキー
            "mlflow.trace.user": user_id,
            # 同一セッション内の複数リクエストをグループ化するためのID
            "mlflow.trace.session": session_id,
            # リクエスト個別の追跡ID（ログとの紐付けなどに使用）
            "mlflow.trace.request_id": str(uuid.uuid4()),
            # 環境タグ: 開発/本番の区別に使用（本番では "production" に変更する）
            "environment": "development",
        }
    )

    # OpenAI APIを呼び出してLLMからの回答を取得する
    # autolog()が有効なので、このAPI呼び出しは自動的に子スパンとして記録される
    response = client.chat.completions.create(
        model="gpt-5-nano-2025-08-07",
        messages=[
            {
                "role": "system",
                # システムプロンプト: エージェントの役割と回答スタイルを定義する
                "content": "あなたはMLflowの専門家です。簡潔に回答してください。",
            },
            {"role": "user", "content": message},
        ],
    )
    # choicesリストの先頭要素からLLMの回答テキストを取り出す
    return response.choices[0].message.content


# === 3. 複数リクエストを実行 ===

# テスト用の質問リスト（実際の本番環境では外部からリクエストが来る）
questions = [
    "MLflow Tracingとは何ですか?",
    "MLflowでモデルをサーブする方法は?",
    "MLflow Evaluateの主な機能は?",
]

print("=== トレース生成 ===")

# 同一セッションとして扱うためにセッションIDを事前に生成しておく
session_id = str(uuid.uuid4())

for i, q in enumerate(questions):
    # 2人のユーザーを交互にシミュレートする（i % 2 → 0 or 1、+1 → 1 or 2）
    user_id = f"user-{(i % 2) + 1}"
    answer = handle_request(q, user_id=user_id, session_id=session_id)
    print(f"\nQ: {q}")
    # 長い回答は先頭100文字だけ表示する
    print(f"A: {answer[:100]}...")

# 非同期ログが有効な場合、バッファに溜まったトレースを明示的に送信する
# これがないと、スクリプト終了時に未送信のトレースが失われる可能性がある
mlflow.flush_trace_async_logging()


# === 4. トレース検索 ===
print("\n=== トレース検索 ===")

# アクティブな実験から直近10件のトレースを取得する
# max_resultsを指定しないと大量のデータが返る可能性があるため、適切に制限する
all_traces = mlflow.search_traces(max_results=10)
print(f"総トレース数: {len(all_traces)}")

# filter_stringでタグを条件に絞り込む
# バッククォートは、ピリオドを含むタグキーをエスケープするために必要
user1_traces = mlflow.search_traces(
    filter_string="tags.`mlflow.trace.user` = 'user-1'",
)
print(f"user-1のトレース数: {len(user1_traces)}")

print("\n✅ MLflow UI でトレースの詳細を確認してください。")
print("   各トレースに user, session, request_id タグが設定されています。")
