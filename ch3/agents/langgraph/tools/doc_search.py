"""Milvusベクトルストアを使用したドキュメント検索ツール。

このモジュールは、事前にインジェストされたMLflowドキュメントから
関連情報を検索するツールを提供します。

【ベクトル検索の仕組み】
  1. クエリテキストをOpenAI Embeddingsで数値ベクトル（埋め込みベクトル）に変換する
  2. Milvusデータベースに保存済みのドキュメントベクトルと類似度を計算する
  3. コサイン類似度の高い上位k件のドキュメントを返す

【ファイルの依存関係】
  - Milvusデータベース: scripts/web_ingest.py で事前に作成が必要（make ingest）
  - OpenAI Embeddings API: OPENAI_API_KEY 環境変数が必要
"""
from functools import lru_cache
import logging
from pathlib import Path
import os

from langchain.tools import tool
from langchain_openai import OpenAIEmbeddings
from pymilvus import MilvusClient

# このモジュールのロガーを取得する（ルートロガーではなくモジュール固有のロガー）
logger = logging.getLogger(__name__)

# プロジェクトルートを起点にデータベースパスを構築する
# __file__の3階層上（tools → langgraph → agents → ch3）がプロジェクトルート
PROJECT_ROOT = Path(__file__).resolve().parents[3]
# MilvusのローカルDBファイルのパス（LiteモードでサーバーレスDBとして使用）
DB_PATH = PROJECT_ROOT / "data" / "milvus.db"


@tool
def doc_search(query: str) -> str:
    """MLflowドキュメントから関連情報を検索する。

    MLflowの機能、API、ガイド、ベストプラクティスに関する
    詳細情報を見つけるために使用します。

    Args:
        query: 検索クエリ（自然言語で入力する）

    Returns:
        検索結果のテキスト（title, URL を含む）、またはエラーメッセージ
    """
    # データベースファイルが存在しない場合は使用不可メッセージを返す
    # （サーバー起動前に make ingest を実行する必要がある）
    if not DB_PATH.exists():
        return "ドキュメント検索は利用できません。先に 'make ingest' を実行してください。"

    try:
        # Milvusに対してベクトル検索を実行する（上位5件を取得）
        results = _search_documents(query, k=5)
    except Exception:
        # 接続エラーや検索エラーをキャッチしてユーザーに通知する
        logger.exception("Milvus検索に失敗しました")
        return "ドキュメント検索中にエラーが発生しました。少し待ってから再度お試しください。"

    if not results:
        return "関連するドキュメントが見つかりませんでした。"

    # 検索結果をテキスト形式に整形する
    formatted_results = []
    for row in results:
        # Milvusの検索結果は entity キーにフィールド値が格納されている
        entity = row.get("entity", {})
        text = entity.get("text", "").strip()    # ドキュメントの本文テキスト
        title = entity.get("title", "").strip()  # ページタイトル
        url = entity.get("url", "").strip()      # ドキュメントのURL

        # 空でないフィールドのみを結合して1件の結果テキストを構築する
        parts = [text]
        if title:
            parts.append(f"Title: {title}")
        if url:
            parts.append(f"URL: {url}")
        formatted_results.append("\n".join(part for part in parts if part))

    # 複数の結果を空行区切りで結合して返す
    return "\n\n".join(formatted_results)


@lru_cache(maxsize=1)
def _get_embeddings() -> OpenAIEmbeddings:
    """OpenAI Embeddingsクライアントをシングルトンとして取得する。

    lru_cache(maxsize=1) により、初回呼び出し時のみインスタンスを生成し、
    以降はキャッシュから返すことでAPIクライアントの重複生成を防ぐ。
    """
    return OpenAIEmbeddings(model=os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small"))


@lru_cache(maxsize=1)
def _get_client() -> MilvusClient:
    """MilvusClientをシングルトンとして取得する。

    ローカルのMilvus Liteファイル（.db）に直接接続する。
    サーバーレスで動作するため、別途Milvusサーバーの起動は不要。
    """
    return MilvusClient(uri=str(DB_PATH.resolve()))


def _search_documents(query: str, k: int) -> list[dict]:
    """クエリをベクトル化してMilvusで類似検索を実行する。

    Args:
        query: 検索クエリテキスト
        k: 取得する上位件数

    Returns:
        Milvusの検索結果リスト（各要素に entity フィールドが含まれる）
    """
    # クエリテキストをOpenAI Embeddings APIで数値ベクトルに変換する
    query_vector = _get_embeddings().embed_query(query)

    # Milvusでコサイン類似度による近似最近傍検索を実行する
    return _get_client().search(
        collection_name="mlflow_docs",   # インジェスト時に作成したコレクション名
        data=[query_vector],             # 検索クエリベクトル（リスト形式で渡す）
        limit=k,                         # 上位k件を返す
        output_fields=["text", "title", "url"],  # 返すフィールド名
    )[0]  # search()は[[result,...]]の形式で返るため[0]で最初の結果リストを取り出す
