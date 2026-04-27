# ---- このセルの見どころ ----
# スーパーバイザー型マルチエージェントアプリをPythonファイルとして書き出します。
# research/outline/writer/reviewの4エージェントを順番に呼び、スーパーバイザーが次の担当を決めます。
# 最後にMLflow ResponsesAgent互換のインターフェースでラップし、標準的なpredict入力で呼び出せるようにします。

"""
スーパーバイザー型マルチエージェントによる技術レポート作成アプリケーションです。

エージェントの役割:
  - リサーチエージェント: テーマについて調査し、ポイントを箇条書きで整理
  - 構成エージェント: 見出し構成と各見出しの要点を決定
  - ライティングエージェント: 構成に沿って本文を執筆
  - レビューエージェント: レポート案をチェックし、必要なら修正を提案

スーパーバイザー:
  - 全体を調整し、各エージェントに順番に仕事を振り分ける

MLflow ResponseAgent:
  - システム全体をResponseAgentでラッピングし、標準的なインターフェースで公開します。
"""

from __future__ import annotations

from typing import TypedDict, Literal, Annotated
import functools

import mlflow
from mlflow.entities import SpanType

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import create_react_agent
from langgraph.graph.message import add_messages


# ==========
# 事前準備
# ==========

# LLM（全エージェント共通）
# GPT-5系の一部モデルはtemperatureの明示指定を受け付けないため、デフォルト値を使います。
llm = ChatOpenAI(model="gpt-5-nano-2025-08-07")

# LangChain/MLflowの自動ロギングを有効化
# LLM呼び出しやエージェント実行を自動的にMLflowに記録
mlflow.langchain.autolog()


# ==========
# 状態の定義
# ==========

class AgentState(TypedDict):
    """
    エージェント間で受け渡す情報をまとめた共有状態です。

    この状態はワークフロー全体を通じて維持され、
    各エージェントが必要な情報を読み取り、自分の出力を書き込みます。
    """
    topic: str              # レポートのテーマ
    research_notes: str     # リサーチ結果（箇条書き）
    outline: str            # レポート構成（見出しと要点）
    draft: str              # レポート本文（初稿）
    review_comments: str    # レビューコメント
    final_report: str       # 最終レポート（修正済み）
    next_agent: str         # スーパーバイザーが決定する次のエージェント名


# ==========
# 各エージェントノードの実装
# ==========

def create_agent_node(agent_name: str, system_prompt: str):
    """
    エージェントノードを作成するファクトリー関数です。

    このパターンにより、コードの重複を避け、
    各エージェントの設定を一元管理できます。

    Args:
        agent_name: エージェントの識別名
        system_prompt: エージェントの役割と指示を定義するプロンプト

    Returns:
        エージェントノード関数
    """
    def agent_node(state: AgentState) -> AgentState:
        """エージェントの処理を実行し、状態を更新します。"""
        # 状態から必要な情報を取り出す
        topic = state.get("topic", "")
        research = state.get("research_notes", "")
        outline = state.get("outline", "")
        draft = state.get("draft", "")

        # エージェントごとに異なる入力を構築
        # 各エージェントは前段階の出力を入力として受け取る
        if agent_name == "research_agent":
            user_content = f"テーマ: {topic}"
        elif agent_name == "outline_agent":
            user_content = f"テーマ: {topic}\n\nリサーチ結果:\n{research}"
        elif agent_name == "writer_agent":
            user_content = f"テーマ: {topic}\n\n構成案:\n{outline}"
        elif agent_name == "review_agent":
            user_content = f"レポートドラフト:\n{draft}"
        else:
            user_content = ""

        # LLMを呼び出し
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content),
        ]
        response = llm.invoke(messages)
        result = response.content

        # 結果を状態に保存
        # 各エージェントは自分の担当フィールドのみを更新
        if agent_name == "research_agent":
            state["research_notes"] = result
        elif agent_name == "outline_agent":
            state["outline"] = result
        elif agent_name == "writer_agent":
            state["draft"] = result
        elif agent_name == "review_agent":
            # レビュー結果を解析して、コメントと最終レポートに分割
            if "【修正後レポート案】" in result:
                comments, final = result.split("【修正後レポート案】", maxsplit=1)
                state["review_comments"] = comments.strip()
                state["final_report"] = final.strip()
            else:
                state["review_comments"] = result
                state["final_report"] = draft  # 修正不要の場合は原稿をそのまま使用

        # トレースにプレビューを記録（後で分析しやすくするため）
        mlflow.update_current_trace(tags={
            f"{agent_name}_preview": result[:100],  # 最初の100文字のみ
        })

        return state

    return agent_node


# 4つの専門エージェントを作成
# 各エージェントは明確な役割と具体的な出力形式を持つ

research_node = create_agent_node(
    "research_agent",
    "あなたは技術リサーチ担当です。テーマについて重要なポイントを箇条書きで5〜7個挙げてください。"
)

outline_node = create_agent_node(
    "outline_agent",
    "あなたは技術レポートの構成を考える担当です。リサーチメモをもとに、見出し構成（3〜5個）と各見出しの要点を番号付きで出力してください。"
)

writer_node = create_agent_node(
    "writer_agent",
    "あなたは技術レポートの執筆担当です。構成案に従って、各見出しごとに2〜4文程度で本文を書いてください。専門用語は平易な言葉で説明してください。"
)

review_node = create_agent_node(
    "review_agent",
    "あなたは技術レポートのレビュー担当です。技術的な正確さ、構成のわかりやすさ、文体をチェックし、【コメント】と【修正後レポート案】の形式で出力してください。"
)


# ==========
# スーパーバイザーノード
# ==========

def supervisor_node(state: AgentState) -> AgentState:
    """
    【スーパーバイザー: ワークフローの制御塔】

    現在の状態を確認し、次にどのエージェントを呼ぶかを決定します。

    判断ロジック（順次処理）:
    1. リサーチ結果がない → research_agent
    2. 構成案がない → outline_agent
    3. 本文がない → writer_agent
    4. 最終レポートがない → review_agent
    5. すべて完了 → FINISH
    """
    # 状態を見て、次に呼ぶべきエージェントを判断
    if not state.get("research_notes"):
        next_agent = "research_agent"
    elif not state.get("outline"):
        next_agent = "outline_agent"
    elif not state.get("draft"):
        next_agent = "writer_agent"
    elif not state.get("final_report"):
        next_agent = "review_agent"
    else:
        next_agent = "FINISH"

    state["next_agent"] = next_agent

    # トレースに判断結果を記録
    mlflow.update_current_trace(tags={
        "next_agent": next_agent,
    })

    return state


# ==========
# グラフの構築
# ==========

def build_graph():
    """
    スーパーバイザー型のマルチエージェントグラフを構築します。

    グラフ構造:
    - 中央にスーパーバイザーを配置
    - 各エージェントはスーパーバイザーから呼び出され、完了後にスーパーバイザーに戻る
    - スーパーバイザーが次のエージェントを決定（条件分岐）
    """
    workflow = StateGraph(AgentState)

    # 5つのノードを追加（スーパーバイザー + 4つの専門エージェント）
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("research_agent", research_node)
    workflow.add_node("outline_agent", outline_node)
    workflow.add_node("writer_agent", writer_node)
    workflow.add_node("review_agent", review_node)

    # 開始点: まずスーパーバイザーから開始
    workflow.add_edge(START, "supervisor")

    # スーパーバイザーの判断に基づいて分岐
    def route_supervisor(state: AgentState) -> Literal["research_agent", "outline_agent", "writer_agent", "review_agent", "__end__"]:
        """スーパーバイザーの決定に基づいて次のノードを返す"""
        next_agent = state.get("next_agent", "FINISH")
        if next_agent == "FINISH":
            return END
        return next_agent

    workflow.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "research_agent": "research_agent",
            "outline_agent": "outline_agent",
            "writer_agent": "writer_agent",
            "review_agent": "review_agent",
            END: END,
        },
    )

    # 各エージェント実行後は、必ずスーパーバイザーに戻る（固定エッジ）
    # これにより、スーパーバイザーが次のステップを制御できる
    for agent in ["research_agent", "outline_agent", "writer_agent", "review_agent"]:
        workflow.add_edge(agent, "supervisor")

    return workflow.compile()


# グラフを構築
graph = build_graph()


# ==========
# ResponseAgentでラッピング
# ==========

from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    output_to_responses_items_stream,
    to_chat_completions_input,
)
from typing import Generator

class MultiAgentResponsesAgent(ResponsesAgent):
    """
    マルチエージェントシステムをResponsesAgentでラッピングします。

    ResponsesAgentとは？
    - MLflowが提供する標準的なエージェントインターフェース
    - OpenAI互換のAPI形式でサービング可能
    - ストリーミングと非ストリーミングの両方に対応

    メリット:
    - 既存のチャットアプリケーションとの統合が容易
    - REST API、Python、CLI等、複数の方法で呼び出し可能
    - MLflowの評価・モニタリング機能と統合
    """

    def __init__(self, graph):
        self.graph = graph

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        """
        非ストリーミング版の予測メソッド。
        ResponsesAgentRequestを受け取り、マルチエージェントを実行します。

        Args:
            request: OpenAI互換の入力リクエスト

        Returns:
            ResponsesAgentResponse: 最終レポートを含むレスポンス
        """
        # ストリーミング版を実行して、完了イベントだけを集める
        outputs = [
            event.item
            for event in self.predict_stream(request)
            if event.type == "response.output_item.done"
        ]

        return ResponsesAgentResponse(
            output=outputs,
            custom_outputs=request.custom_inputs
        )

    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """
        ストリーミング版の予測メソッド。
        各エージェントの出力をリアルタイムで返すことも可能ですが、
        ここでは最終結果のみを返す実装としています。

        Args:
            request: OpenAI互換の入力リクエスト

        Yields:
            ResponsesAgentStreamEvent: ストリーミングイベント
        """
        # RequestをChatCompletions形式に変換
        messages = to_chat_completions_input([i.model_dump() for i in request.input])

        # ユーザーの質問からトピックを抽出
        topic = messages[-1]["content"] if messages else ""

        # グラフに渡す初期状態を作成
        initial_state = AgentState(
            topic=topic,
            research_notes="",
            outline="",
            draft="",
            review_comments="",
            final_report="",
            next_agent="",
        )

        # グラフを実行（すべてのエージェントが順次実行される）
        final_state = self.graph.invoke(initial_state)

        # 最終レポートを出力として返す
        final_report = final_state.get("final_report", "")

        yield ResponsesAgentStreamEvent(
            type="response.output_item.done",
            item=self.create_text_output_item(
                text=final_report,
                id="final_report",
            )
        )


# エージェントをインスタンス化してMLflowに登録
agent = MultiAgentResponsesAgent(graph)
mlflow.models.set_model(agent)
