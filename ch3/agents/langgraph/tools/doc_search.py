"""Milvusベクトルストアを使用したドキュメント検索ツール。

このモジュールは、事前にインジェストされたMLflowドキュメントから
関連情報を検索するツールを提供します。
"""
from functools import lru_cache
import logging
from pathlib import Path
import os

from langchain.tools import tool
from langchain_openai import OpenAIEmbeddings
from pymilvus import MilvusClient

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DB_PATH = PROJECT_ROOT / "data" / "milvus.db"


@tool
def doc_search(query: str) -> str:
    """MLflowドキュメントから関連情報を検索する。

    MLflowの機能、API、ガイド、ベストプラクティスに関する
    詳細情報を見つけるために使用します。

    Args:
        query: 検索クエリ

    Returns:
        検索結果のテキスト、またはエラーメッセージ
    """
    # raise ValueError("ドキュメントでーたベースに接続できません")
    if not DB_PATH.exists():
        return "ドキュメント検索は利用できません。先に 'make ingest' を実行してください。"

    try:
        results = _search_documents(query, k=5)
    except Exception:
        logger.exception("Milvus検索に失敗しました")
        return "ドキュメント検索中にエラーが発生しました。少し待ってから再度お試しください。"

    if not results:
        return "関連するドキュメントが見つかりませんでした。"

    formatted_results = []
    for row in results:
        entity = row.get("entity", {})
        text = entity.get("text", "").strip()
        title = entity.get("title", "").strip()
        url = entity.get("url", "").strip()
        parts = [text]
        if title:
            parts.append(f"Title: {title}")
        if url:
            parts.append(f"URL: {url}")
        formatted_results.append("\n".join(part for part in parts if part))
    return "\n\n".join(formatted_results)


@lru_cache(maxsize=1)
def _get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small"))


@lru_cache(maxsize=1)
def _get_client() -> MilvusClient:
    return MilvusClient(uri=str(DB_PATH.resolve()))


def _search_documents(query: str, k: int) -> list[dict]:
    query_vector = _get_embeddings().embed_query(query)
    return _get_client().search(
        collection_name="mlflow_docs",
        data=[query_vector],
        limit=k,
        output_fields=["text", "title", "url"],
    )[0]
