"""5.5-5.6節: 評価データセットを定義し、mlflow.genai.evaluate()で自動評価を実行する。

評価データセットの定義 → predict_fnの実装 → evaluate()の実行 → 結果確認
までを1つのスクリプトにまとめています。

【mlflow.genai.evaluate()の仕組み】
  1. data（評価データセット）の各行について predict_fn を呼び出してLLMの回答を取得する
  2. 各スコアラーが（入力・出力・期待値）を見て品質スコアを付ける
  3. 全件の評価結果とスコアをMLflow Runとして記録する
  4. MLflow UIのEvaluationタブで結果を確認できる

【predict_fn と評価データセットの対応】
  evaluate(data=eval_dataset, predict_fn=predict_fn) を呼ぶと、
  各 data[i]["inputs"]["question"] が predict_fn(question=...) に渡される。
  inputs のキー名と predict_fn の引数名が一致している必要がある。

実行: make eval
前提: make ingest でドキュメントが取り込み済みであること
"""

import sys

from dotenv import load_dotenv

# .envファイルからOPENAI_API_KEYを読み込む
load_dotenv()

import mlflow

from agents.langgraph import LangGraphAgent
from agents.thread import Thread
# 全評価スコアラーをインポートする（scorers.pyで定義している）
from evaluation.scorers import (
    correctness,           # 回答が期待される内容をカバーしているか（LLM-as-a-Judge）
    tool_call_correctness, # 期待されるツールが呼び出されたか
    contains_code_block,   # 回答にコードブロックが含まれているか（ルールベース）
    has_reference_link,    # 参照リンクが含まれているか（Guidelinesベース）
    appropriate_katakana,  # カタカナ表記が適切か（Guidelinesベース）
)

# --- 5.5節: 評価データセットの定義 ---
# 各エントリは inputs（エージェントへの入力）と expectations（期待値）を持つ辞書
# expectations はスコアラーが参照する期待値（expected_response, expected_tool_calls等）
eval_dataset = [
    {
        "inputs": {
            "question": "MLflowトレーシングはどのフレームワークに対応していますか？"
        },
        "expectations": {
            # LLMジャッジが回答の正確さを判定するための参照回答
            "expected_response": (
                "MLflowトレーシングは、LangChain、LangGraph、LlamaIndex、"
                "OpenAI SDK、Anthropic SDK、AWS Bedrock SDKなどの"
                "主要なフレームワークに対応しています。"
            ),
            # ToolCallCorrectnessがチェックするべきツール呼び出し一覧
            "expected_tool_calls": [{"name": "doc_search"}],
        },
    },
    {
        "inputs": {
            "question": "LangGraphエージェントのトークン使用量を追跡するにはどうすればよいですか？"
        },
        "expectations": {
            "expected_response": (
                "LangGraphエージェントのトークン使用量をMLflowで可視化するには、"
                "MLflowのトレーシング機能が利用できます。"
                "mlflow.langchain.autolog() APIをコードに追加することで、"
                "エージェントを実行する度にトレースが生成され、"
                "呼び出しごとのトークンの使用量が記録されます。"
            ),
            "expected_tool_calls": [{"name": "doc_search"}],
        },
    },
    {
        "inputs": {
            "question": "実験管理はどのように始めれば良いですか？"
        },
        "expectations": {
            "expected_response": (
                "MLflowで実験管理を始めるには、mlflow.set_experiment()で"
                "実験を作成し、mlflow.start_run()でランを開始します。"
                "パラメータやメトリクスをログし、MLflow UIで結果を確認できます。"
            ),
            "expected_tool_calls": [{"name": "doc_search"}],
        },
    },
    {
        "inputs": {
            "question": "MLflowとは何ですか？"
        },
        "expectations": {
            "expected_response": (
                "MLflowは機械学習のライフサイクル全体を管理するための"
                "オープンソースプラットフォームです。"
                "実験管理、モデル管理、デプロイメントなどの機能を提供しています。"
            ),
            "expected_tool_calls": [{"name": "doc_search"}],
        },
    },
    {
        "inputs": {
            "question": "MLflowと競合するツールは何ですか？"
        },
        "expectations": {
            "expected_response": (
                "MLflowの競合ツールとしては、Weights & Biases (W&B)、"
                "Neptune.ai、Comet MLなどがあります。"
            ),
            "expected_tool_calls": [{"name": "doc_search"}],
        },
    },
]


def main():
    """評価データセットを使ってmlflow.genai.evaluate()を実行する。"""
    print("=" * 60)
    print("5.5-5.6節: 自動評価の実行")
    print("=" * 60)

    # --- エージェントの初期化 ---
    try:
        agent = LangGraphAgent()
    except ConnectionError:
        print("\nMLflow Tracking Serverに接続できません。")
        print("  'uv run mlflow server --port 5000' を実行してください。")
        sys.exit(1)

    # --- 5.6節: predict_fnの定義 ---
    def predict_fn(question: str) -> str:
        """評価用のpredict関数。データセットのinputs["question"]に対応する。

        evaluate()はdata[i]["inputs"]のキーをpredictfnの引数として展開する。
        例: inputs={"question": "..."} → predict_fn(question="...")
        """
        # 各呼び出しで新しいThreadを作成してステートレスに評価する
        return agent.process_query(query=question, thread=Thread())

    # --- evaluate()の実行 ---
    print(f"\n評価データセット: {len(eval_dataset)}件")
    print("スコアラー: correctness, tool_call_correctness, "
          "contains_code_block, has_reference_link, appropriate_katakana")
    print("\n評価を開始します...\n")

    try:
        # evaluate()が内部で:
        # 1. predict_fn を全データセットで呼び出してLLM回答を収集
        # 2. 全スコアラーで各回答を評価
        # 3. 結果をMLflow Runとして記録
        eval_results = mlflow.genai.evaluate(
            data=eval_dataset,
            predict_fn=predict_fn,
            scorers=[
                correctness,
                tool_call_correctness,
                contains_code_block,
                has_reference_link,
                appropriate_katakana,
            ],
        )
    except Exception as e:
        print(f"\n評価中にエラーが発生しました: {e}")
        sys.exit(1)

    # --- 結果の表示 ---
    print("\n" + "=" * 60)
    print("評価完了!")
    print("=" * 60)
    print(f"\nMLflow UI で詳細を確認してください。")
    print(f"  http://localhost:5000")


if __name__ == "__main__":
    main()
