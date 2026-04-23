"""Agent Server用エージェント定義（7.3節）。

第4章のLangGraphAgentをAgent Serverで公開するためのラッパーです。
@invoke デコレータでResponses APIエンドポイントとして登録します。

【このファイルの役割】
  - handle_request() を @invoke() で装飾することで、
    AgentServer（start_server.py）がこの関数を /invocations エンドポイントに紐付ける
  - Responses API形式（OpenAI互換）のJSONをLangGraphAgentの呼び出し形式に変換する

変更点（第4章からの差分）:
- システムプロンプトをプロンプトレジストリから取得（6章との連携）
- Responses API形式でリクエスト/レスポンスを処理
- Agent Serverの自動トレーシングを活用
"""

import os
import uuid

import dotenv

# .envファイルからAPIキーなどの環境変数を読み込む（import前に実行する必要がある）
dotenv.load_dotenv()

import mlflow
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
# @invokeデコレータ: 関数をAgent Serverのエンドポイントハンドラとして登録する
from mlflow.genai.agent_server import invoke
# Responses API形式のリクエスト/レスポンス型定義（OpenAI Responses API互換）
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse

# 第4章のエージェントを再利用（LangGraph + RAGツール）
from agents.langgraph.agent import LangGraphAgent
# Thread: 会話履歴を管理するオブジェクト（thread_idでLangGraphの状態を識別）
from agents.thread import Thread

# --- MLflow設定 ---
# 環境変数が未設定の場合はローカルのMLflowサーバーを使用
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("QAエージェント - サービング")

# エージェントのインスタンスをモジュールロード時に1回だけ初期化する
# （リクエストのたびに初期化するとパフォーマンスが悪化するため）
agent = LangGraphAgent()


def _load_system_prompt() -> str:
    """プロンプトレジストリからシステムプロンプトを取得する。

    第6章でMLflowプロンプトレジストリに登録したシステムプロンプトを使用する。
    "@production" エイリアスが指す最新の本番用プロンプトを動的に取得することで、
    コードを変更せずにプロンプトのみ更新できる。

    レジストリが利用できない場合は、第4章のデフォルトプロンプトにフォールバックする。
    """
    try:
        # プロンプトレジストリからproductionエイリアスのプロンプトを取得
        # URI形式: "prompts:/{プロンプト名}@{エイリアス}"
        prompt = mlflow.genai.load_prompt("prompts:/qa-agent-system-prompt@production")
        return prompt.template
    except Exception:
        # プロンプトレジストリが未設定または接続失敗の場合は第4章のデフォルトを使用
        from agents.langgraph.agent import SYSTEM_PROMPT

        return SYSTEM_PROMPT


@invoke()
async def handle_request(request) -> ResponsesAgentResponse:
    """QAエージェントへのリクエストを処理するエンドポイントハンドラ。

    Responses APIのメッセージ形式（OpenAI互換）を受け取り、
    第4章のLangGraphAgentで処理して、Responses API形式で返す。

    【処理の流れ】
      1. dictをResponsesAgentRequestオブジェクトに変換（必要な場合）
      2. inputメッセージリストからuserロールのテキストを抽出
      3. LangGraphAgentに渡して回答を生成
      4. ResponsesAgentResponse形式に変換して返す

    Args:
        request: Responses API形式のリクエスト（dictまたはResponsesAgentRequest）
                 形式例: {"input": [{"role": "user", "content": "質問"}]}

    Returns:
        Responses API形式のレスポンス
    """
    # JSONとして受信したdictをPydanticモデルに変換して型安全に扱う
    if isinstance(request, dict):
        request = ResponsesAgentRequest(**request)

    # Responses APIのinputメッセージリストを走査してuserロールのテキストを抽出する
    # inputは複数のメッセージを含むリスト（マルチターン対応の設計）
    user_message = None
    for msg in request.input:
        # Pydanticモデルの場合は辞書に変換してから処理する
        msg_dict = msg.model_dump() if hasattr(msg, "model_dump") else msg
        if msg_dict.get("role") == "user":
            content = msg_dict.get("content", "")
            if isinstance(content, list):
                # contentがリスト形式の場合（マルチモーダルメッセージなど）は
                # type=="input_text" のテキスト要素だけを結合する
                user_message = " ".join(
                    item.get("text", "")
                    for item in content
                    if isinstance(item, dict) and item.get("type") == "input_text"
                )
            else:
                # contentが文字列の場合はそのまま使用する
                user_message = content

    # ユーザーメッセージが取得できなかった場合はガイダンスメッセージを返す
    if not user_message:
        return ResponsesAgentResponse(
            output=[
                {
                    # Responses APIではメッセージごとにユニークなIDが必要
                    "id": f"msg_{uuid.uuid4().hex[:24]}",
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "質問を入力してください。",
                        }
                    ],
                }
            ]
        )

    # LangGraphAgentでユーザーの質問を処理する
    # リクエストごとに新しいThreadを作成することでステートレスなサービングを実現する
    # （会話履歴は保持しない設計 = 各リクエストが独立した会話として処理される）
    thread = Thread()
    response_text = agent.process_query(user_message, thread)

    # Responses API形式でレスポンスを構築して返す
    # output はメッセージオブジェクトのリスト形式
    return ResponsesAgentResponse(
        output=[
            {
                # レスポンスメッセージにもユニークなIDを付与する
                "id": f"msg_{uuid.uuid4().hex[:24]}",
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": response_text,
                    }
                ],
            }
        ]
    )
