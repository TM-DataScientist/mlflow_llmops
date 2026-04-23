"""
第8章 - 8.3 フィードバック収集

トレースに対してフィードバックを記録し、検索する方法を確認します。

【このスクリプトで学ぶこと】
  1. mlflow.log_feedback() でユーザーフィードバックをトレースに付与する方法
  2. フィードバックの種類（bool値、数値スコア）と AssessmentSource の使い方
  3. トレースからフィードバックを取得して確認する方法

【フィードバック収集の目的】
  本番システムでユーザーが「役に立った/役に立たなかった」などのシグナルを
  提供することで、LLMの回答品質を継続的に評価・改善できる。
  フィードバックはトレースに Assessment として紐付けられ、
  MLflow UIの [Assessments] タブで確認できる。

実行方法:
  make feedback
  または
  uv run python monitoring/03_feedback.py
"""

from dotenv import load_dotenv

# .envからOPENAI_API_KEYを読み込む
load_dotenv()

import mlflow
from openai import OpenAI
# フィードバックの評価元情報を記述するためのMLflowエンティティをインポート
from mlflow.entities import AssessmentSource, AssessmentSourceType

# 実験名を設定する（他の監視スクリプトと共通の実験に集約する）
mlflow.set_experiment("ch8-monitoring-quickstart")
# OpenAI API呼び出しの自動トレーシングを有効化する
mlflow.openai.autolog()

client = OpenAI()

# === 1. LLM呼び出し(フィードバック対象のトレースを生成) ===
print("=== フィードバック対象のトレース生成 ===\n")

# テスト用の質問とその期待されるフィードバックを定義する
# thumbs_up: サムズアップ(True)かサムズダウン(False)か
# rating: 1〜5のスコア評価
# comment: フィードバックの根拠コメント
questions_and_feedback = [
    {
        "question": "MLflowでモデルをデプロイする方法を教えてください。",
        "thumbs_up": True,
        "rating": 5,
        "comment": "正確で分かりやすい回答でした",
    },
    {
        "question": "MLflowとKubeflowの違いは?",
        "thumbs_up": True,
        "rating": 3,
        "comment": "概要は良いが、もう少し具体例が欲しい",
    },
    {
        "question": "量子コンピューティングについて教えてください。",
        "thumbs_up": False,
        "rating": 1,
        "comment": "MLflowとは無関係な質問に回答してしまっている",
    },
]

for item in questions_and_feedback:
    # LLM API呼び出し: autolog()により自動的にトレースが生成される
    response = client.chat.completions.create(
        model="gpt-5-nano-2025-08-07",
        messages=[
            {"role": "system", "content": "あなたはMLflowの専門家です。"},
            {"role": "user", "content": item["question"]},
        ],
    )

    # 直前のLLM呼び出しに対応するトレースIDを取得する
    trace_id = mlflow.get_last_active_trace_id()
    answer = response.choices[0].message.content

    # 質問と回答の先頭80文字を表示する
    print(f"Q: {item['question']}")
    print(f"A: {answer[:80]}...")

    # === 2. フィードバックを記録 ===

    # フィードバック①: サムズアップ/ダウン（bool値）
    # log_feedback() はトレースに Assessment を紐付けるAPIで、
    # "thumbs_up"や"rating"などのname引数で評価の種類を識別する
    mlflow.log_feedback(
        trace_id=trace_id,
        # フィードバックの名前（キー）: 同一トレースに複数のfeedbackを付与できる
        name="user_feedback",
        # bool値で「役に立った(True)」「役に立たなかった(False)」を記録する
        value=item["thumbs_up"],
        # フィードバックの提供元情報を記述する
        source=AssessmentSource(
            # source_type: 人間によるフィードバックか、LLMによるジャッジかを区別する
            # HUMAN = 人間が評価, LLM_JUDGE = LLMが自動評価
            source_type=AssessmentSourceType.HUMAN,
            # source_id: どのユーザー・システムがフィードバックを提供したかを識別する
            source_id="demo-user-1",
        ),
        # rationale: フィードバックの根拠・理由を自由記述で記録する
        rationale=item["comment"],
    )

    # フィードバック②: 数値スコア (1〜5)
    # 同じトレースに対して複数種類のフィードバックを付与できる
    mlflow.log_feedback(
        trace_id=trace_id,
        name="rating",
        # 数値（整数・浮動小数点数）でスコアを記録する
        value=item["rating"],
        source=AssessmentSource(
            source_type=AssessmentSourceType.HUMAN,
            source_id="demo-user-1",
        ),
        # ratingにはコメント不要なので rationale は省略
    )

    # 絵文字でフィードバック結果を視覚的に表示する
    thumbs = "\U0001f44d" if item["thumbs_up"] else "\U0001f44e"
    print(f"   -> feedback: {thumbs}, rating: {item['rating']}/5")
    print()

# === 3. フィードバック付きトレースの確認 ===
print("=== フィードバック確認 ===\n")

# 直近10件のトレースを取得する
traces = mlflow.search_traces(max_results=10)

# head(3)で先頭3件のみ処理する（iterrows()でDataFrameを1行ずつ処理）
for _, row in traces.head(3).iterrows():
    # トレースIDを使ってトレースの詳細情報を取得する
    trace = mlflow.get_trace(trace_id=row["trace_id"])

    # type="feedback" で Assessmentの中からfeedback型のみを取得する
    # （他にも type="expectation" など複数タイプが存在する）
    assessments = trace.search_assessments(type="feedback")

    if assessments:
        # トレースIDは長いため先頭16文字のみ表示する
        print(f"Trace: {row['trace_id'][:16]}...")
        for a in assessments:
            # a.name: フィードバック名, a.feedback.value: 評価値, a.source.source_id: 評価者ID
            print(f"  {a.name}: {a.feedback.value} (by {a.source.source_id})")
        print()

print(
    "✅ MLflow UIの各トレースで [Assessments] タブからフィードバックを確認できます。"
)
