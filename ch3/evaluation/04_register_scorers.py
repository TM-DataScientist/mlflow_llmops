"""5.4.6節: 評価指標をMLflowに登録してバージョン管理する。

GuidelinesスコアラーをMLflowのJudgesタブに登録し、
get_scorerで取得するデモスクリプト。

【スコアラー登録のメリット】
  スコアラーをMLflowに登録することで:
  - 複数のスクリプトやチームメンバーが同じスコアラーを再利用できる
  - スコアラーの変更履歴をバージョン管理できる
  - MLflow UIのJudgesタブで一元管理できる

【登録できるスコアラー】
  Guidelinesスコアラーのみ登録可能。
  ルールベース（@scorer）スコアラーはコード依存のため登録不可。

実行: make register
前提: MLflow Tracking Serverが起動していること
"""

import sys

from dotenv import load_dotenv

# .envファイルから環境変数を読み込む
load_dotenv()

import mlflow

# 登録対象のスコアラーをインポートする（scorers.pyで定義している）
from evaluation.scorers import has_reference_link, appropriate_katakana

# MLflow接続設定
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("MLflow QAエージェント")


def main():
    """スコアラーをMLflowに登録する。"""
    print("=" * 60)
    print("5.4.6節: スコアラーの登録")
    print("=" * 60)

    try:
        # has_reference_link スコアラーをMLflowに登録する
        # register()を呼ぶと新しいバージョンとして登録される（初回はv1）
        print("\n--- has_reference_link の登録 ---")
        has_reference_link.register()
        print("  登録完了")

        # appropriate_katakana スコアラーをMLflowに登録する
        print("\n--- appropriate_katakana の登録 ---")
        appropriate_katakana.register()
        print("  登録完了")

        # 登録済みスコアラーの取得テスト
        # get_scorer(name="...") でMLflowから取得できることを確認する
        # versionを省略すると最新バージョンを取得する。
        # 特定バージョンの指定も可能: get_scorer(name="...", version=1)
        print("\n--- 登録済みスコアラーの取得テスト ---")
        loaded_scorer = mlflow.genai.get_scorer(name="has_reference_link")
        print(f"  取得成功: {loaded_scorer.name}")

        loaded_scorer2 = mlflow.genai.get_scorer(name="appropriate_katakana")
        print(f"  取得成功: {loaded_scorer2.name}")

    except ConnectionError:
        print("\nMLflow Tracking Serverに接続できません。")
        print("  'uv run mlflow server --port 5000' を実行してください。")
        sys.exit(1)
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        sys.exit(1)

    print()
    print("=" * 60)
    print("スコアラーの登録完了!")
    print("MLflow UI (http://localhost:5000) のJudgesタブで")
    print("登録されたスコアラーを確認してください。")
    print("=" * 60)


if __name__ == "__main__":
    main()
