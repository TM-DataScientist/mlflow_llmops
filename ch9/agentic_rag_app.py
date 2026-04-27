# ---- このセルの見どころ ----
# Agentic RAGアプリ本体をPythonファイルとして書き出します。
# LangGraphでrouter/retrieve/check/rewrite/answerの5ノードを接続し、質問に応じて検索有無や再検索を判断します。
# MLflow Tracingのタグにはルーティング結果や品質判定を残し、後から失敗経路を分析しやすくします。

"""
最小構成のエージェント型RAGアプリケーションの例です。

・LangGraphで次の5ノードを持つグラフを作成します:
  - router: 質問を見て「検索すべきかどうか」を決める
  - retrieve: Chromaから関連文書を検索する
  - check: 検索結果が十分かどうかを判定する
  - rewrite: 質問を少し言い換える
  - answer: コンテキスト＋質問から最終回答を生成する

・MLflow Tracingで、各ノードの処理をスパンとして記録します。
"""

from __future__ import annotations

from typing import Literal, Dict, Any

import mlflow
from mlflow.entities import SpanType

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, MessagesState, START, END

from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages


# ==========
# 事前準備
# ==========

# LangChainのLLM
# GPT-5系の一部モデルはtemperatureの明示指定を受け付けないため、デフォルト値を使います。
llm = ChatOpenAI(model="gpt-5-nano-2025-08-07")

# Chromaベクトルストア（前のステップで作成したものを読み込み）
# build_vectorstore()で保存したときと同じ埋め込みモデルを指定します。
PERSIST_DIR = "./chroma_store"
EMBEDDING_MODEL = "text-embedding-3-small"
vectordb = Chroma(
    embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL),
    persist_directory=PERSIST_DIR,
)
retriever = vectordb.as_retriever()

# LangChain/MLflowの自動ロギングを有効化
# LLM呼び出しやチェーン実行を自動的にMLflowに記録
mlflow.langchain.autolog()


# ==========
# 検索関数
# ==========

def retrieve_docs(query: str) -> str:
    """
    質問文から関連文書を検索し、テキストを1つの文字列として返します。

    Args:
        query: 検索クエリ（ユーザーの質問）

    Returns:
        検索結果の文書を改行で連結した文字列
    """
    docs = retriever.invoke(query)
    return "\n\n".join([d.page_content for d in docs])


# ==========
# カスタム状態の定義
# ==========

class AgenticRAGState(TypedDict):
    """
    LangGraphのワークフロー全体で共有される状態

    Attributes:
        messages: 会話履歴（質問、中間メッセージ、回答など）
        route: routerノードの判定結果（'rag' or 'llm_only'）
        context: 検索結果のテキスト
        check_result: checkノードの判定結果（'answer' or 'rewrite'）
    """
    messages: Annotated[list, add_messages]  # 自動的にメッセージを追加
    route: str
    context: str
    check_result: str


# ==========
# ノード定義
# ==========

def router_node(state: AgenticRAGState) -> Dict[str, Any]:
    """
    【ノード1: ルーター】
    質問を見て「検索すべきかどうか」を判定するノードです。

    判定ロジック:
    - 社内文書に関する質問 → 'rag'（検索が必要）
    - 一般常識や外部知識の質問 → 'llm_only'（LLMの知識のみで回答）
    """
    question = state["messages"][-1].content

    prompt = (
        "次の質問に答えるために、社内の技術文書を検索した方がよいかを判定してください。\n"
        "検索した方がよい場合は 'rag'、不要な場合は 'llm_only' とだけ答えてください。\n\n"
        f"質問: {question}"
    )
    res = llm.invoke([HumanMessage(content=prompt)])
    decision = res.content.strip().lower()

    # トレースにルート情報をタグとして残す（後で分析しやすくするため）
    mlflow.update_current_trace(tags={"route_decision": decision})

    return {"messages": [AIMessage(content=f"[route={decision}]")], "route": decision}


def retrieve_node(state: AgenticRAGState) -> Dict[str, Any]:
    """
    【ノード2: 検索】
    Chromaベクトルデータベースから関連文書を検索するノードです。

    処理内容:
    1. ユーザーの質問（最初のメッセージ）を取得
    2. ベクトル検索を実行
    3. 検索結果をstateのcontextに格納
    """
    question = state["messages"][0].content
    context = retrieve_docs(question)

    # 検索結果をstateに追加
    return {"messages": [AIMessage(content=context)], "context": context}


def check_node(state: AgenticRAGState) -> Dict[str, Any]:
    """
    【ノード3: 品質チェック】
    検索結果のテキスト（context）が質問に十分関連しているかどうかを判定します。

    判定基準:
    - 関連性が高い → 'answer'（そのまま回答生成へ）
    - 関連性が低い → 'rewrite'（質問を改善して再検索）
    """
    question = state["messages"][0].content
    context = state.get("context", "")

    prompt = (
        "あなたは、検索で得られた文書が質問に関連しているかどうかを判定する役割です。\n"
        "関連していれば 'yes'、ほとんど関係なければ 'no' とだけ回答してください。\n\n"
        f"質問: {question}\n\n"
        f"検索結果: {context[:2000]}\n"  # 長すぎる場合は最初の2000文字のみ使用
    )
    res = llm.invoke([HumanMessage(content=prompt)])
    score = res.content.strip().lower()

    decision = "answer" if score == "yes" else "rewrite"

    # トレースに判定結果をタグとして保存
    mlflow.update_current_trace(tags={"check_decision": decision})

    # 判定結果をstateに保存
    return {"check_result": decision}


def rewrite_node(state: AgenticRAGState) -> Dict[str, Any]:
    """
    【ノード4: 質問の改善】
    質問を検索に適した形に言い換えるノードです。

    目的:
    - 曖昧な表現を具体化
    - 検索に適したキーワードを含める
    - 元の意図は保持
    """
    question = state["messages"][0].content
    prompt = (
        "次の質問文を、検索に適した形になるように言い換えてください。\n"
        "ただし、意味や意図は変えないでください。\n\n"
        f"元の質問: {question}"
    )
    res = llm.invoke([HumanMessage(content=prompt)])

    # 新しい質問をMessagesStateに追加して、再度routerに戻します
    return {"messages": [HumanMessage(content=res.content)]}


def answer_node(state: AgenticRAGState) -> Dict[str, Any]:
    """
    【ノード5: 回答生成】
    検索結果のコンテキストと質問を使って最終回答を生成するノードです。
    contextがない場合は、LLMの知識のみで回答します。

    2つのモード:
    1. RAGモード: 検索結果に基づいて回答（ハルシネーション防止）
    2. LLMモード: LLMの内部知識のみで回答
    """
    question = state["messages"][0].content
    context = state.get("context", "")

    if context:
        # RAGルート: コンテキストに基づいて回答
        prompt = (
            "以下のコンテキストに基づいて質問に答えてください。\n"
            "コンテキストに書かれていないことは推測せず、「わかりません」と答えてください。\n"
            "3文以内で、簡潔でわかりやすい日本語で答えてください。\n\n"
            f"質問: {question}\n\n"
            f"コンテキスト:\n{context}"
        )
    else:
        # LLM単体ルート: LLMの知識で直接回答
        prompt = (
            "以下の質問に、あなたの知識に基づいて答えてください。\n"
            "3文以内で、簡潔でわかりやすい日本語で答えてください。\n\n"
            f"質問: {question}"
        )

    res = llm.invoke([HumanMessage(content=prompt)])
    return {"messages": [res]}


# ==========
# グラフ定義
# ==========

def build_agentic_rag_graph():
    """
    LangGraphでエージェント型RAGのワークフローグラフを組み立てます。

    グラフ構造:
    - ノード: 処理単位（router, retrieve, check, rewrite, answer）
    - エッジ: ノード間の遷移（固定エッジと条件付きエッジ）
    - 条件分岐: routerとcheckの判定結果に応じて次のノードを動的に決定
    """
    workflow = StateGraph(AgenticRAGState)

    # 5つのノードを登録
    workflow.add_node("router", router_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("check", check_node)
    workflow.add_node("rewrite", rewrite_node)
    workflow.add_node("answer", answer_node)

    # 開始点: まずrouterノードから開始
    workflow.add_edge(START, "router")

    # routerの判定結果に応じて分岐
    def route_decision(state: AgenticRAGState) -> Literal["retrieve", "answer"]:
        """routerの判定結果に基づいて次のノードを決める"""
        route = state.get("route", "rag")
        return "retrieve" if route == "rag" else "answer"

    workflow.add_conditional_edges(
        "router",
        route_decision,
        {
            "retrieve": "retrieve",  # 検索が必要 → retrieveノードへ
            "answer": "answer",      # 検索不要 → answerノードへ
        },
    )

    # retrieve後は必ずcheckに進む（固定エッジ）
    workflow.add_edge("retrieve", "check")

    # checkの判定結果に応じて分岐
    def check_decision(state: AgenticRAGState) -> Literal["answer", "rewrite"]:
        """checkの判定結果に基づいて次のノードを決める"""
        return state.get("check_result", "answer")

    workflow.add_conditional_edges(
        "check",
        check_decision,
        {
            "answer": "answer",    # 品質OK → 回答生成へ
            "rewrite": "rewrite",  # 品質不足 → 質問改善へ
        },
    )

    # rewrite後は再度routerへ戻る（再試行ループ）
    workflow.add_edge("rewrite", "router")

    # answer後は終了
    workflow.add_edge("answer", END)

    # グラフをコンパイルして実行可能な状態にする
    return workflow.compile()


# グラフを構築
graph = build_agentic_rag_graph()

# MLflowのModels from Codeパターンで登録できるようにする
mlflow.models.set_model(graph)
