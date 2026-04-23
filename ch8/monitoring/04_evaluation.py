"""
第8章 - 8.3/8.4 品質評価パイプライン

蓄積されたトレースに対して、LLM-as-a-Judgeで品質評価を実行します。
5章で開発時に使ったmlflow.genai.evaluate()を、本番トレースに適用する形です。

【このスクリプトで学ぶこと】
  1. 本番トレースをそのままevaluate()の入力として使う方法
  2. 組み込みスコアラー（Safety, RelevanceToQuery, Guidelines）の使い方
  3. 評価結果のメトリクスサマリーの確認方法

【開発時評価 vs 本番トレース評価の違い】
  - 開発時（第5章）: predict_fn に評価データを渡してLLMを新たに呼び出す
  - 本番評価（この章）: 既存のトレース（実際のユーザーリクエスト）を直接評価する
    → コストを追加せず、本番での実際の挙動を評価できる

実行方法:
  make eval
  または
  uv run python monitoring/04_evaluation.py

前提:
  先に make tracing, make cost, make feedback を実行して
  トレースを蓄積しておいてください。
"""

from dotenv import load_dotenv

# .envからOPENAI_API_KEYを読み込む（LLM-as-a-Judgeに使用）
load_dotenv()

import mlflow
from mlflow.genai.scorers import (
    Safety,          # 安全性スコアラー: 有害・不適切なコンテンツを検出する
    RelevanceToQuery,  # 関連性スコアラー: 回答が質問に関連しているか評価する
    Guidelines,      # ガイドラインスコアラー: カスタムのガイドラインへの準拠を評価する
)

# 実験名を設定する（他の監視スクリプトと共通の実験を使用する）
mlflow.set_experiment("ch8-monitoring-quickstart")

# === 1. 本番トレースの取得 ===
print("=== 本番トレースの取得 ===\n")

# 直近20件のトレースを取得する
# 実際の本番環境では、日次・週次バッチで評価を実行することが多い
traces = mlflow.search_traces(max_results=20)

print(f"評価対象トレース数: {len(traces)}")

if len(traces) == 0:
    print("⚠️ トレースが見つかりません。先に make tracing を実行してください。")
    exit(1)

# === 2. スコアラーで評価 ===
print("\n=== LLM-as-a-Judge 評価実行 ===\n")

# evaluate() にトレースのDataFrameを直接渡すことで、
# 既存の本番トレースを再評価できる（新たなLLM呼び出しは発生しない）
results = mlflow.genai.evaluate(
    # data: DataFrameまたはトレースのリストを渡す
    # 本番トレースを渡すことで、実際のユーザーリクエストに対する品質を評価できる
    data=traces,
    scorers=[
        # RelevanceToQuery: 回答が質問のトピックに関連しているか（1=関連、0=無関係）
        RelevanceToQuery(),
        # Safety: 回答に有害・暴力的・差別的なコンテンツが含まれていないか（1=安全、0=問題あり）
        Safety(),
        # Guidelines: カスタムルールへの準拠度を評価する
        # LLMをジャッジとして使い、ガイドラインに従っているか yes/no で判定する
        Guidelines(
            name="helpfulness",
            # 評価基準を自然言語で記述する（LLMがこのガイドラインに基づいて評価する）
            guidelines="回答はユーザーの質問に対して具体的で実用的な情報を提供している必要があります。",
        ),
    ],
)

# === 3. 結果の確認 ===
print("=== 評価結果サマリー ===\n")

# metrics はスコアラー名をキー、平均スコアを値とするdict
# 例: {"relevance_to_query/score": 0.85, "safety/score": 1.0, "helpfulness/score": 0.72}
for metric_name, value in results.metrics.items():
    print(f"  {metric_name}: {value:.3f}")

print(f"\n✅ 評価完了。MLflow UI のエクスペリメント画面で詳細を確認できます。")
print("   Evaluation タブに各トレースのスコアが記録されています。")
