"""Milvusベクトルストアを使用したドキュメント検索ツール。

このモジュールは、事前にインジェストされたMLflowドキュメントから
関連情報を検索するツールを提供します。

【ch3版との違い】
  ch3は MilvusClient を直接使って低レベルの検索を行うが、
  このモジュールは LangChain の Milvus 統合（langchain_milvus）を使い、
  LangChainのリトリーバーインターフェースで検索する。
  また、接続エラー時のリトライロジックを実装している。

【ベクトル検索の仕組み】
  1. クエリテキストをOpenAI Embeddingsで数値ベクトルに変換する
  2. Milvusデータベースに保存済みのドキュメントベクトルと類似度を計算する
  3. コサイン類似度の高い上位k件のドキュメントを返す

【ファイルの依存関係】
  - Milvusデータベース: scripts/web_ingest.py で事前に作成が必要（make ingest）
  - OpenAI Embeddings API: OPENAI_API_KEY 環境変数が必要
"""
import logging
from pathlib import Path

from langchain.tools import tool
from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings
# connections: Milvusの接続管理モジュール（リセット処理に使用）
from pymilvus import connections

import os

# このモジュールのロガーを取得する
logger = logging.getLogger(__name__)

# langchain_milvus が AsyncMilvusClient 初期化時に出す警告を抑制する
# 同期処理のみ使用するため、非同期クライアントの警告は無視してよい
logging.getLogger("langchain_milvus").setLevel(logging.ERROR)

# データベースファイルのパス（カレントディレクトリ相対パス）
# make ingest で data/milvus.db が生成される
DB_PATH = Path("data") / "milvus.db"


@tool
def doc_search(query: str) -> str:
    """MLflowドキュメントから関連情報を検索する。

    MLflowの機能、API、ガイド、ベストプラクティスに関する
    詳細情報を見つけるために使用します。

    Args:
        query: 検索クエリ（自然言語で入力する）

    Returns:
        検索結果のテキスト、またはエラーメッセージ
    """
    # データベースファイルが存在しない場合は使用不可メッセージを返す
    if not DB_PATH.exists():
        return "ドキュメント検索は利用できません。先に 'make ingest' を実行してください。"

    # リトライロジック: 接続エラーが発生した場合に最大2回まで再試行する
    # Milvusのソケット接続が切れた際に自動で再接続するための仕組み
    max_retries = 2
    for attempt in range(max_retries):
        try:
            # Milvusリトリーバーを取得する（DBファイルの存在確認込み）
            retriever = _get_retriever()
            if retriever is None:
                return "ドキュメント検索は利用できません。先に 'make ingest' を実行してください。"

            # LangChainリトリーバーインターフェースでベクトル検索を実行する
            # invoke(query) → クエリをベクトル化 → Milvus類似検索 → Documentリストを返す
            docs = retriever.invoke(query)

            # Document オブジェクトの page_content（本文テキスト）を結合して返す
            return "\n\n".join([doc.page_content for doc in docs])

        except Exception as e:
            # 接続エラーの場合は接続をリセットしてリトライする
            if "Connection refused" in str(e) or "connect" in str(e).lower():
                logger.warning(f"Milvus接続に失敗しました（試行 {attempt + 1}回目）、接続をリセット中...")
                _reset_milvus_connection()
                if attempt == max_retries - 1:
                    # 最大リトライ回数を超えた場合はユーザーに通知する
                    return "接続の問題により、ドキュメント検索が一時的に利用できません。もう一度お試しください。"
            else:
                # 接続エラー以外の例外は上位に再送出する
                raise


def _reset_milvus_connection():
    """Milvusの接続をリセットして古いソケットをクリアする。

    非同期接続が残っている場合に disconnect() で全接続を切断する。
    次のリトライで新しい接続が確立される。
    """
    try:
        # 全ての接続エイリアスを列挙して切断する
        for alias in list(connections.list_connections()):
            connections.disconnect(alias[0])
    except Exception:
        # 切断中のエラーは無視する（既に切断済みの場合など）
        pass


def _get_retriever():
    """Milvusデータベースからリトリーバーを取得する。

    LangChainのMilvusベクトルストアに接続し、
    類似検索用のリトリーバーオブジェクトを作成して返す。

    Returns:
        LangChainリトリーバー、またはDBが存在しない場合はNone
    """
    if not DB_PATH.exists():
        return None

    # 埋め込みモデルを初期化する（環境変数 EMBEDDING_MODEL で切り替え可能）
    embeddings = OpenAIEmbeddings(model=os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small"))

    # LangChainのMilvusラッパーでベクトルストアに接続する
    # uri に .db ファイルを指定することで Milvus Lite（サーバーレス）として動作する
    vectorstore = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": str(DB_PATH)},
        collection_name="mlflow_docs",   # インジェスト時に作成したコレクション名
    )

    # as_retriever() でLangChainリトリーバーインターフェースに変換する
    # search_kwargs={"k": 5} で上位5件を返すよう指定する
    return vectorstore.as_retriever(search_kwargs={"k": 5})
