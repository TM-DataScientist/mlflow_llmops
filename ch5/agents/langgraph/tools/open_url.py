"""システムブラウザでURLを開くツール。

このモジュールは、ユーザーのデフォルトブラウザで
URLを開くツールを提供します。

【使用場面】
  エージェントが「このURLを確認してください」と回答する際に、
  実際にブラウザを起動してページを開く。ユーザーが手動でURLをコピー&
  ペーストする手間を省く。

【OS対応】
  - macOS: open コマンドを使用
  - Windows: start コマンド（shell=True）を使用
  - Linux: xdg-open コマンドを使用（デスクトップ環境が必要）
"""
import json
import subprocess
import sys

from langchain.tools import tool


@tool
def open_url(url: str) -> str:
    """システムのデフォルトブラウザでURLを開く。

    ドキュメントページ、GitHubリンク、その他のWeb URLを
    ユーザーが閲覧できるように開きます。

    Args:
        url: 開くURL（http:// または https:// で始まる完全なURL）

    Returns:
        成功/失敗を示すJSON文字列
    """
    try:
        # sys.platform でOSを判定し、対応するブラウザ起動コマンドを選択する
        if sys.platform == "darwin":
            # macOS: open コマンドでデフォルトブラウザを起動する
            subprocess.run(["open", url], check=True)
        elif sys.platform == "win32":
            # Windows: start コマンドでデフォルトブラウザを起動する
            # shell=True が必要（start は Windowsのシェル組み込みコマンドのため）
            subprocess.run(["start", url], shell=True, check=True)
        else:
            # Linux: xdg-open コマンドでデフォルトブラウザを起動する
            # GNOMEやKDEなどのデスクトップ環境が必要
            subprocess.run(["xdg-open", url], check=True)

        # 成功した場合はURLと共に成功メッセージを返す
        return json.dumps({"success": True, "message": f"URLを開きました: {url}"})

    except Exception as e:
        # subprocess.run が check=True でも失敗した場合（コマンドが存在しない等）
        return json.dumps({"success": False, "error": str(e)})
