"""QAエージェントのモデル記録スクリプト（7.2節）。

第4章で構築したLangGraphエージェントをMLflowに記録し、
モデルレジストリに登録します。

【スクリプトの全体フロー】
  1. log_agent()  : モデルをMLflow Runに記録し、レジストリに自動登録する
  2. set_champion_alias() : 登録済みモデルに "champion" エイリアスを付与する
  3. verify_model() : "models:/qa-agent@champion" でロードし、動作確認する

使用方法:
    make log-model
    # または
    uv run python -m serving.log_model
"""

import os
from pathlib import Path

import dotenv

# .envファイルからAPIキーなどの環境変数を読み込む
dotenv.load_dotenv()

import mlflow
from mlflow import MlflowClient


# --- MLflow設定 ---
# 環境変数が未設定の場合はローカルのMLflowサーバーを使用
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "QAエージェント - サービング"
# モデルレジストリに登録するモデル名（エイリアスでの参照に使用）
REGISTERED_MODEL_NAME = "qa-agent"

# models-from-codeパターン用のモデル定義ファイルへの絶対パス
# このファイルがMLflowのアーティファクトとして保存され、
# ロード時にコードが再実行されてモデルが再構築される
MODEL_CODE_PATH = str(Path(__file__).parent / "model_code.py")


def log_agent():
    """QAエージェントをMLflowに記録する。

    MLflow v3ではmodels-from-codeパターンを使用し、
    モデル定義コードのパスを指定して記録します。
    log_model()のregistered_model_nameを指定することで、
    Run記録と同時にモデルレジストリへの登録も行われます。

    Returns:
        (model_info, run): モデル情報とMLflow Runオブジェクトのタプル
    """
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="qa-agent-serving") as run:
        # models-from-codeパターンでエージェントを記録する
        # lc_model にファイルパスを渡すことで、コードファイル自体がアーティファクトになる
        # (モデルオブジェクトをpickleするのではなく、Pythonコードを保存する方式)
        model_info = mlflow.langchain.log_model(
            lc_model=MODEL_CODE_PATH,
            name="qa-agent",                                # アーティファクト名（Run内のパス）
            registered_model_name=REGISTERED_MODEL_NAME,    # レジストリに自動登録
        )

        # モデルのメタデータをRunのタグとして記録する
        # MLflow UIでフィルタリングや検索に使用できる
        mlflow.set_tags(
            {
                "agent_type": "langgraph",
                "tools": "doc_search,web_search,open_url",
                "chapter": "7",
                "base_chapter": "4",
            }
        )

        print(f"モデルを記録しました:")
        print(f"  Run ID: {run.info.run_id}")
        print(f"  Model URI: {model_info.model_uri}")
        print(f"  Registered Model: {REGISTERED_MODEL_NAME}")

        return model_info, run


def set_champion_alias(model_info):
    """log_model()で自動登録されたモデルにchampionエイリアスを設定する。

    【エイリアスの意味】
      MLflowモデルレジストリではバージョン番号（v1, v2, ...）の他に
      "champion", "challenger" などの意味のある名前（エイリアス）で
      モデルを参照できる。コードで "models:/qa-agent@champion" と書くことで
      バージョン番号を意識せずに本番モデルを参照できる。

    Args:
        model_info: mlflow.langchain.log_model()の戻り値
    """
    client = MlflowClient(tracking_uri=TRACKING_URI)

    # log_model(registered_model_name=...)で自動登録されたバージョン番号を取得
    # 通常は1から始まり、登録のたびにインクリメントされる
    model_version = model_info.registered_model_version

    # 指定したバージョンに "champion" エイリアスを付与する
    # 既存のchampionエイリアスがあれば上書きされる
    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias="champion",
        version=model_version,
    )

    print(f"championエイリアスを設定しました:")
    print(f"  名前: {REGISTERED_MODEL_NAME}")
    print(f"  バージョン: {model_version}")
    print(f"  エイリアス: champion")


def verify_model():
    """記録したモデルをロードして動作確認する。

    "models:/qa-agent@champion" のURI形式でレジストリからロードし、
    テストクエリを送信して正常に回答が返るかを確認する。
    """
    print("\n--- モデルの動作確認 ---")

    # エイリアスを使ってレジストリからモデルをロードする
    # models-from-codeの場合、ここでmodel_code.pyが再実行されてモデルが再構築される
    model_uri = f"models:/{REGISTERED_MODEL_NAME}@champion"
    print(f"モデルをロード中: {model_uri}")

    loaded_model = mlflow.langchain.load_model(model_uri)

    # LangChain形式のinputでテストクエリを送信する
    test_query = "MLflow Tracingの主な機能を教えてください"
    print(f"テストクエリ: {test_query}")

    result = loaded_model.invoke(
        {
            "messages": [
                {"role": "user", "content": test_query}
            ]
        },
        # thread_idはLangGraphのチェックポイント（会話状態）を識別するキー
        config={"configurable": {"thread_id": "verify-test"}},
    )

    # LangGraphの戻り値はmessagesリストなので、最後のAIメッセージを取得する
    for msg in reversed(result["messages"]):
        if hasattr(msg, "content") and msg.type == "ai" and msg.content:
            print(f"回答: {msg.content[:200]}...")
            break

    print("動作確認: OK")


def main():
    """メインの実行フロー。

    log_agent → set_champion_alias → verify_model の順に実行する。
    """
    print("=" * 50)
    print("QAエージェントのモデル記録（第7章）")
    print("=" * 50)

    # 1. エージェントをMLflowに記録・レジストリに自動登録
    model_info, run = log_agent()

    # 2. 登録済みモデルにchampionエイリアスを設定
    set_champion_alias(model_info)

    # 3. ロードして動作確認
    verify_model()

    print("\n完了! 次のコマンドでサービングを開始できます:")
    print("  make serve")


if __name__ == "__main__":
    main()
