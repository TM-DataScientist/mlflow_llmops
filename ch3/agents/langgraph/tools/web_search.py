"""Exa APIを使用したWeb検索ツール。

このモジュールは、最新のWeb情報を検索するツールを提供します。
ドキュメントにない最新情報が必要な場合に使用します。

【Exa APIとは】
  Exa（旧Metaphor）は、意味的な検索（セマンティック検索）に特化した
  検索APIサービス。キーワードマッチではなく、クエリの意図・文脈を理解して
  関連性の高いWebページを返す。LLMアプリケーションとの相性が良い。

【ツール選択の指針（エージェントへのヒント）】
  - まず doc_search でMLflow公式ドキュメントを検索する
  - 公式ドキュメントにない最新情報（リリースノート、コミュニティ記事等）が
    必要な場合のみ web_search を使用する
"""
import json
import os

from langchain.tools import tool


@tool
def web_search(query: str) -> str:
    """Webから最新の情報を検索する。

    MLflow、Databricks、または関連トピックについて、
    ドキュメントにない最新情報が必要な場合に使用します。

    Args:
        query: 検索クエリ（自然言語で入力する）

    Returns:
        検索結果のJSON文字列（title, url, content, highlights を含む）
    """
    # Exa APIクライアントをインポートする（遅延インポートで起動時の依存を軽減）
    # exa_py がインストールされていない場合もインポートエラーを遅延させられる
    from exa_py import Exa

    # APIキーを環境変数から取得する（.envファイルで設定する）
    api_key = os.environ.get("EXA_API_KEY")
    if not api_key:
        # APIキーが設定されていない場合はエラーメッセージをJSON形式で返す
        return json.dumps({"error": "EXA_API_KEYが環境変数に設定されていません"})

    try:
        # Exaクライアントを初期化する
        exa = Exa(api_key=api_key)

        # 検索を実行する（コンテンツとハイライト付き）
        # search_and_contents(): 検索結果に加えてページの本文テキストも取得する
        response = exa.search_and_contents(
            query=query,
            type="auto",           # 検索タイプを自動選択（neural/keywordを自動で切り替え）
            num_results=5,         # 上位5件を取得
            text={"max_characters": 1000},    # 本文テキストは最大1000文字に制限
            highlights={"num_sentences": 3},  # クエリに関連する3文をハイライトとして抽出
        )

        # 検索結果を整形する
        results = []
        for result in response.results:
            entry = {
                "title": result.title,  # ページタイトル
                "url": result.url,      # ページURL
            }
            # テキストコンテンツが取得できた場合のみ追加する
            if hasattr(result, "text") and result.text:
                entry["content"] = result.text
            # ハイライト（関連文章の抜粋）が取得できた場合のみ追加する
            if hasattr(result, "highlights") and result.highlights:
                entry["highlights"] = result.highlights
            results.append(entry)

        # 結果をJSON文字列にシリアライズして返す
        # indent=2 で読みやすい形式にする（LLMのコンテキストとして使用するため）
        return json.dumps({"results": results, "total": len(results)}, indent=2)

    except Exception as e:
        # 検索エラーをキャッチしてJSON形式で返す
        return json.dumps({"error": str(e)})
