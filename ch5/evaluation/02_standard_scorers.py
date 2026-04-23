"""5.4.4節: 標準の評価指標(ToolCallCorrectness, Correctness)を個別にテストする。

既存のトレースを使って標準スコアラーを試すデモスクリプト。
01_vibe_check.pyで生成されたトレースを自動取得して評価します。

【標準スコアラーとは】
  MLflowが提供するビルトインの評価指標。LLMをジャッジとして使い、
  回答の品質を自動的に数値（0〜1）で評価する。

  - ToolCallCorrectness: エージェントが期待されるツールを呼び出したかを評価する
    （例: doc_search を呼び出すべき質問で、実際に doc_search が使われたか）
  - Correctness: エージェントの回答が期待される回答の要点をカバーしているかを評価する

【個別テストの目的】
  スコアラーを全件評価（05_run_evaluation.py）に組み込む前に、
  動作を確認するための単体テストとして実行する。

実行: make test-standard
前提: make vibe-check でトレースが生成済みであること
"""

import sys

from dotenv import load_dotenv

# .envファイルからOPENAI_API_KEYを読み込む（LLM-as-a-Judgeに必要）
load_dotenv()

import mlflow
from mlflow.genai.scorers import Correctness, ToolCallCorrectness

# MLflow接続設定（エージェントを使わずスコアラーのみテストするため明示的に設定する）
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("MLflow QAエージェント")


def get_latest_traces(experiment_name: str = "MLflow QAエージェント", max_results: int = 5):
    """最新のトレースを取得するヘルパー関数。

    実験名からexperiment_idを取得し、タイムスタンプ降順で最新のトレースを返す。

    Args:
        experiment_name: 検索対象の実験名
        max_results: 取得する最大件数

    Returns:
        トレースのリスト（新しい順）。実験が存在しない場合は空リスト。
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return []
    traces = mlflow.search_traces(
        # locationsに実験IDを指定して対象実験を絞り込む
        locations=[experiment.experiment_id],
        max_results=max_results,
        # timestamp DESCで最新のトレースから順に取得する
        order_by=["timestamp DESC"],
        # リスト形式で返す（DataFrameではなく）
        return_type="list",
    )
    return traces


def test_tool_call_correctness(trace):
    """ToolCallCorrectnessスコアラーを単体でテストする。

    エージェントが期待したツール（doc_search）を呼び出したかを評価する。
    """
    print("--- ToolCallCorrectness ---")

    scorer = ToolCallCorrectness()
    # 期待するツール呼び出しを定義する（nameでツール名を指定する）
    expected_tools = [{"name": "doc_search"}]

    result = scorer(
        trace=trace,
        # expectationsにexpected_tool_callsを渡す
        expectations={"expected_tool_calls": expected_tools},
    )

    print(f"  name: {result.name}")
    print(f"  value: {result.value}")  # 1.0=成功, 0.0=失敗
    # rationale: LLMジャッジの判定理由（ToolCallCorrectnessにはない場合もある）
    if hasattr(result, "rationale") and result.rationale:
        print(f"  rationale: {result.rationale}")
    print()


def test_correctness(trace):
    """Correctnessスコアラーを単体でテストする。

    エージェントの回答が期待される回答の要点をカバーしているかを評価する。
    LLMをジャッジとして使い、yes/noまたはスコアで判定する。
    """
    print("--- Correctness ---")

    scorer = Correctness()
    # テスト対象の質問に対する期待される回答を定義する
    expected_response = (
        "MLflowトレーシングは、LangChain、LangGraph、LlamaIndex、"
        "OpenAI SDK、Anthropic SDK、AWS Bedrock SDKなどの"
        "主要なフレームワークに対応しています。"
    )

    result = scorer(
        trace=trace,
        # expectationsにexpected_responseを渡す
        expectations={"expected_response": expected_response},
    )

    print(f"  name: {result.name}")
    print(f"  value: {result.value}")  # 1.0=要点をカバー, 0.0=カバー不足
    # rationale: LLMジャッジが「なぜこのスコアを付けたか」の説明テキスト
    if hasattr(result, "rationale") and result.rationale:
        print(f"  rationale: {result.rationale}")
    print()


def main():
    """標準スコアラーの個別テストを実行する。"""
    print("=" * 60)
    print("5.4.4節: 標準スコアラーの個別テスト")
    print("=" * 60)

    # 最新トレースを取得する
    try:
        traces = get_latest_traces(max_results=3)
    except Exception as e:
        print(f"\nMLflow Tracking Serverに接続できません: {e}")
        print("  'uv run mlflow server --port 5000' を実行してください。")
        sys.exit(1)

    if not traces:
        print("\nトレースが見つかりません。")
        print("  先に 'make vibe-check' を実行してトレースを生成してください。")
        sys.exit(1)

    # 最新のトレースを1件使ってテストする
    trace = traces[0]
    print(f"\nトレースID: {trace.info.trace_id}")
    print(f"トレース数: {len(traces)}件取得\n")

    # 各スコアラーを個別にテストする
    test_tool_call_correctness(trace)
    test_correctness(trace)

    print("=" * 60)
    print("標準スコアラーのテスト完了!")
    print("=" * 60)


if __name__ == "__main__":
    main()
