"""LangGraphを使用したエージェント実装。

このモジュールは、LangGraphフレームワークを使用してMLflowに関する質問に
回答するエージェントを実装しています。ツール呼び出しとメモリ管理を
サポートしています。

【LangGraphとは】
  LangGraphは、LLMアプリケーションを有向グラフとしてモデル化するフレームワーク。
  「エージェント（LLM）→ ツール（関数）→ エージェント → ...」という
  ループを状態機械として表現し、複雑な推論ループを実装できる。

【エージェントのアーキテクチャ】
  ユーザー入力
    └→ [agent ノード] LLMが回答を生成
          ├→ ツール呼び出しあり → [tools ノード] ツールを実行 → [agent ノード] に戻る
          └→ ツール呼び出しなし → [END] 処理終了、回答を返す
"""
from typing import List, Optional
import os

import mlflow

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
# MemorySaver: 会話履歴をメモリ（インプロセス）に保存するチェックポインタ
# thread_idをキーにして、複数の会話セッションを同時管理できる
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
# MessagesState: グラフのステートとしてメッセージリストを管理するビルトイン型
from langgraph.graph.message import MessagesState
# ToolNode: ツール（関数）呼び出しを担当するビルトインノード
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI

from agents.thread import Message, Thread
# 利用可能なツールをインポートする（ツールが使えない環境では None になる場合がある）
from .tools import doc_search, web_search, open_url

# MLflowトレーシングの設定
# Tracking Serverへの接続先を指定する（環境変数 MLFLOW_TRACKING_URI でも設定可）
mlflow.set_tracking_uri("http://localhost:5000")
# トレースを集約する実験（エクスペリメント）の名前
mlflow.set_experiment("MLflow QAエージェント")
# LangChain/LangGraph の全呼び出しを自動トレースする
# これにより、LLM呼び出し・ツール呼び出しが自動でスパンとして記録される
mlflow.langchain.autolog()

# システムプロンプト: エージェントの役割と振る舞いを定義する
# このプロンプトは会話の最初のメッセージとして必ずLLMに渡される
SYSTEM_PROMPT = """あなたはMLflowに関する質問に答える専門アシスタントです。
MLflowは機械学習のライフサイクルを管理するためのオープンソースプラットフォームです。

あなたの責務:
- MLflowの機能、API、ベストプラクティスに関する質問に回答する
- MLflowの概念を説明し、問題のトラブルシューティングを支援する
- 適切なリソースやドキュメントへユーザーを案内する

利用可能なツールを使用して、正確で最新の情報を取得してください。
ツールから取得した情報を提供する際は、必ずURLを含む引用を記載してください。
"""


class LangGraphAgent:
    """MLflowに関する質問に回答するLangGraphエージェント。

    このクラスは、リポジトリ内のすべてのエージェント実装が提供すべき
    インターフェースを提供します。CLIとAPIは、基盤となるフレームワークに
    関係なく、このクラスを使用してエージェントと対話します。
    """

    def __init__(self):
        """エージェントを初期化する。

        ツールのリストを構築し、LangGraphのグラフをコンパイルする。
        この処理はサーバー起動時に1回だけ実行される（リクエストごとではない）。
        """
        # None のツール（環境変数未設定などで使えないツール）を除外してリストを構築する
        self.tools = [t for t in [doc_search, web_search, open_url] if t is not None]
        # LangGraphの実行グラフを構築・コンパイルする
        self.executor = self._build_graph()

        print(f"LangGraphエージェントを初期化しました（ツール数: {len(self.tools)}）:")
        for tool in self.tools:
            print(f"  - {tool.name}")

    def _tools_condition(self, messages: List[BaseMessage]) -> str:
        """ツールを呼び出すか終了するかを判定する条件関数。

        LangGraphのadd_conditional_edgesに渡すルーター関数。
        LLMの最新メッセージにツール呼び出し（tool_calls）が含まれていれば
        'tools' ノードへ、なければ 'end'（グラフ終了）へルーティングする。

        Args:
            messages: グラフのステート内のメッセージリスト

        Returns:
            次の遷移先ノード名: "tools" または "end"
        """
        if messages:
            last = messages[-1]
            # AIMessage に tool_calls 属性があり、かつ空でない場合はツール呼び出しあり
            if isinstance(last, AIMessage) and getattr(last, "tool_calls", []):
                return "tools"
        return "end"

    def _build_graph(self):
        """LangGraphエージェントのグラフを構築する。

        グラフの構造:
          - "agent" ノード: LLMを呼び出して回答またはツール呼び出しを生成する
          - "tools" ノード: ToolNodeがツールを実行し、結果をステートに追加する
          - 条件付きエッジ: _tools_condition()の戻り値で次のノードを決定する

        Returns:
            MemorySaverによる会話履歴管理機能を持つコンパイル済みグラフ
        """
        # LLMモデルを初期化する（環境変数 LLM_MODEL で切り替え可能）
        model = ChatOpenAI(model=os.environ.get("LLM_MODEL", "gpt-5-nano-2025-08-07"))

        # ツールがある場合はモデルにツールをバインドする
        # bind_tools() により、LLMはツールの定義を知り、必要に応じてtool_callsを生成できる
        model_with_tools = model.bind_tools(self.tools) if self.tools else model

        def call_model(state: MessagesState):
            """グラフの "agent" ノード: モデルを呼び出して応答を生成する。

            MessagesState のメッセージ履歴全体を LLM に渡し、
            LLMのレスポンス（AIMessage）をステートに追加して返す。
            """
            response = model_with_tools.invoke(state["messages"])
            # ステートの更新: リストで返すとMessagesStateが既存メッセージに追記してくれる
            return {"messages": [response]}

        # StateGraph: グラフの構造を定義するビルダー
        # MessagesState: ステートとしてメッセージリストを管理するビルトイン型
        graph = StateGraph(MessagesState)

        # "agent" ノードを追加する（call_model関数をノードとして登録）
        graph.add_node("agent", call_model)

        if self.tools:
            # ツールがある場合のグラフ構造を構築する

            # ToolNode: tool_callsに応じてツール関数を実行するビルトインノード
            # ツールの実行結果をToolMessageとしてステートに追加する
            tool_node = ToolNode(self.tools)
            graph.add_node("tools", tool_node)

            # 条件付きエッジ: "agent"ノードの出力に応じて次のノードを動的に決定する
            # _tools_condition()が "tools" を返せば tools ノードへ、"end" なら終了
            graph.add_conditional_edges(
                "agent",
                lambda state: self._tools_condition(state["messages"]),
                {"tools": "tools", "end": END},
            )

            # ツール実行後はエージェントに戻る（ループを形成する）
            # ツール結果をもとにLLMがさらに推論・回答生成を行う
            graph.add_edge("tools", "agent")
        else:
            # ツールがない場合は "agent" ノードから直接終了する
            graph.add_edge("agent", END)

        # グラフのエントリーポイント（最初に実行するノード）を設定する
        graph.set_entry_point("agent")

        # MemorySaver: thread_idをキーにして会話ステートをインメモリで保持する
        # 同じthread_idを渡し続けることで複数ターンの会話文脈が維持される
        return graph.compile(checkpointer=MemorySaver())

    def _extract_last_ai_message(self, messages) -> Optional[AIMessage]:
        """メッセージリストの末尾から最初のAIMessageを取り出す。

        ツール実行結果（ToolMessage）の後にAIMessageが来るため、
        末尾から逆順に探索して最初に見つかったAIMessageを返す。

        Args:
            messages: グラフ実行後のメッセージリスト

        Returns:
            最後のAIMessage、見つからない場合はNone
        """
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                return msg
        return None

    @mlflow.trace
    def process_query(self, query: str, thread: Thread) -> str:
        """ユーザーのクエリをエージェントで処理して回答を返す。

        これは、リポジトリ内のすべてのエージェント実装が提供すべき
        統一インターフェースです。CLIとAPIは、基盤となるフレームワーク
        （LangGraph、Pydantic AI、OpenAI Agents SDKなど）に関係なく、
        このメソッドを使用してエージェントと対話します。

        【処理の流れ】
          1. 初回会話であればシステムプロンプトをメッセージに追加する
          2. ユーザーメッセージをメッセージに追加する
          3. グラフを実行してLLMとツールが連携して回答を生成する
          4. 最後のAIMessageを取り出して文字列として返す

        Args:
            query: ユーザーの入力メッセージ
            thread: コンテキストを維持するための会話スレッド
                    thread.idがLangGraphのthread_idとして使われる

        Returns:
            エージェントの回答（文字列）
        """
        # トレースにタグを追加（MLflow UIでの検索・フィルタリングに使用）
        mlflow.update_current_trace(
            tags={
                "environment": "development",
                "agent_version": "v1.0",
            }
        )

        print(f"クエリを処理中: {query}")

        # グラフに渡すメッセージリスト（MemorySaverで既存履歴とマージされる）
        incoming_messages = []

        # 新しい会話（メッセージ履歴が空）の場合のみシステムプロンプトを追加する
        # 2回目以降のターンでは、MemorySaverが前の会話状態を保持しているため不要
        if not thread.messages:
            incoming_messages.append(SystemMessage(content=SYSTEM_PROMPT))
            thread.messages.append(Message(role="system", content=SYSTEM_PROMPT))

        # ユーザーのメッセージをグラフへの入力とThreadの履歴両方に追加する
        incoming_messages.append(HumanMessage(content=query))
        thread.messages.append(Message(role="user", content=query))

        # グラフを実行する
        # configのthread_idにより、MemorySaverが前ターンの会話状態を引き継ぐ
        config = {"configurable": {"thread_id": thread.id}}
        result_state = self.executor.invoke({"messages": incoming_messages}, config=config)

        # 実行結果のメッセージリストから最後のAIメッセージを取り出す
        ai_message = self._extract_last_ai_message(result_state["messages"])
        if not ai_message:
            raise RuntimeError("エージェントからの回答がありませんでした。")

        # 回答をスレッドの履歴に保存し、文字列として返す
        thread.messages.append(Message(role="assistant", content=ai_message.content))
        return ai_message.content
