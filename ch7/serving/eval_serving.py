"""サービング環境での評価スクリプト（7.3節）。

Agent Serverと同じ@invoke関数をin-processで呼び出し、
第5章と同じ評価フレームワークを適用します。

【評価の実施方法】
  通常の評価（第5章）との違いは、predict_fnに本番サービングと同じ
  @invoke関数を使用する点です。これにより、サービング環境特有の
  前処理・後処理も含めたエンドツーエンドの品質を評価できます。

注意: Agent Serverを停止してから実行してください。
    エージェント初期化時にMilvusデータベースを開くため、
    Agent Serverが起動中だとファイルロックが競合します。

使用方法:
    make eval
    # または
    uv run python -m serving.eval_serving
"""

import asyncio
import os

import dotenv

# .envファイルからAPIキーなどの環境変数を読み込む
dotenv.load_dotenv()

import mlflow
# get_invoke_function: @invokeで登録されたエンドポイントハンドラを取得するユーティリティ
from mlflow.genai.agent_server import get_invoke_function
# LLM-as-a-Judgeスコアラー（GPTなどのLLMが評価者となって品質を自動採点する）
from mlflow.genai.scorers import RelevanceToQuery, Safety, Guidelines
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse

# @invokeデコレータの登録に必要（このインポートによりhandle_requestが登録される）
import serving.agent  # noqa: F401

# --- MLflow設定 ---
# 評価結果はMLflow Runとして別の実験に記録する
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("QAエージェント - サービング評価")


# 評価データセット（第5章のデータセットと共通の形式）
# 各サンプルは inputs（エージェントへの入力）と expected_response（期待する回答）を持つ
# inputsのネスト構造はhandle_request()が受け取るResponses API形式に対応している
EVAL_DATASET = [
    {
        "inputs": {
            # handle_request()の引数requestに相当する辞書
            "request": {
                "input": [
                    {
                        "role": "user",
                        "content": "MLflow Tracingとは何ですか?",
                    }
                ]
            }
        },
        # expected_responseはスコアラーが回答の適切さを判定する際の参照として使用する
        "expected_response": (
            "MLflow Tracingは、LLMアプリケーションの実行フローを可視化するための"
            "機能です。プロンプト、検索結果、ツール呼び出し、モデルの応答を"
            "記録し、デバッグや品質改善に活用できます。"
        ),
    },
    {
        "inputs": {
            "request": {
                "input": [
                    {
                        "role": "user",
                        "content": "MLflowでプロンプトをバージョン管理する方法は?",
                    }
                ]
            }
        },
        "expected_response": (
            "MLflowのPrompt Registryを使うことで、プロンプトの"
            "バージョン管理とエイリアスによるライフサイクル管理が可能です。"
        ),
    },
    {
        "inputs": {
            "request": {
                "input": [
                    {
                        "role": "user",
                        "content": "MLflowの評価機能でLLMの品質をどう測定しますか?",
                    }
                ]
            }
        },
        "expected_response": (
            "mlflow.genai.evaluate()を使用し、LLM-as-a-Judgeスコアラーで"
            "関連性、安全性、正確性などの品質指標を自動的に評価できます。"
        ),
    },
]


def sync_invoke_fn(request: dict) -> ResponsesAgentResponse:
    """Agent Serverの@invoke関数を同期的に呼び出すラッパー。

    mlflow.genai.evaluate()はpredict_fnに同期関数（synchronous function）を要求するが、
    handle_request()はasync関数として定義されているため、このラッパーで同期化する。

    asyncio.run()を使って新しいイベントループを作成し、
    非同期関数を同期的に実行することで両者を橋渡しする。

    Args:
        request: Responses API形式のリクエスト辞書
                 （EVAL_DATASETのinputs["request"]に対応）

    Returns:
        handle_request()が生成したResponses API形式のレスポンス
    """
    # get_invoke_function()でモジュールに登録された@invoke関数を取得する
    invoke_fn = get_invoke_function()
    # asyncio.run()で非同期関数を同期的に実行する
    return asyncio.run(invoke_fn(ResponsesAgentRequest(**request)))


def main():
    """サービング環境のエージェントを評価する。

    【使用するスコアラー】
      - RelevanceToQuery: 回答がユーザーの質問に関連しているかを評価（0〜1）
      - Safety: 有害なコンテンツが含まれていないかを評価（0〜1）
      - Guidelines(uses_sources): ドキュメントに基づく具体的情報が含まれるかを評価

    評価結果はMLflow UIの "QAエージェント - サービング評価" 実験で確認できる。
    """
    print("=" * 50)
    print("サービング環境での評価（第7章）")
    print("=" * 50)

    # 第5章と同じスコアラーを使用してサービング環境の品質を評価する
    scorers = [
        # ユーザーの質問に対して回答が適切に関連しているかをLLMが判定する
        RelevanceToQuery(),
        # 回答に有害なコンテンツ（暴力・差別・違法情報など）が含まれないかを評価する
        Safety(),
        # カスタムガイドラインに基づいて回答の品質を評価する
        # ここではMLflow公式ドキュメントに基づく具体的な情報が含まれているかを確認する
        Guidelines(
            name="uses_sources",
            guidelines=(
                "回答にはMLflow公式ドキュメントや検索結果に基づく"
                "具体的な情報を含む必要があります。"
            ),
        ),
    ]

    print(f"評価データセット: {len(EVAL_DATASET)} 件")
    print(f"スコアラー: {', '.join(s.name for s in scorers)}")
    print()

    # mlflow.genai.evaluate()で全データセットを評価する
    # predict_fnにはsync_invoke_fn（サービングと同じエンドポイントハンドラ）を渡す
    # 評価結果は自動的にMLflow Runとして記録される
    results = mlflow.genai.evaluate(
        data=EVAL_DATASET,
        predict_fn=sync_invoke_fn,
        scorers=scorers,
    )

    # 集計されたスコアを表示する（各スコアラーの平均値）
    print("\n--- 評価結果 ---")
    for metric_name, value in results.metrics.items():
        print(f"  {metric_name}: {value:.3f}")

    print(f"\n評価完了: {len(EVAL_DATASET)} 件")
    print("詳細はMLflow UIで確認できます:")
    print(f"  {TRACKING_URI}")


if __name__ == "__main__":
    main()
