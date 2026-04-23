"""Agent Server起動スクリプト（7.3節）。

QAエージェントをResponses APIエンドポイントとしてホストします。
MLflow Agent ServerはFastAPI/uvicornベースのASGIアプリケーションを
自動生成し、/invocationsエンドポイントでエージェントを公開します。

使用方法:
    make serve
    # または
    uv run python serving/start_server.py --reload --port 5005

テスト:
    curl -X POST http://localhost:5005/invocations \
        -H "Content-Type: application/json" \
        -d '{"input": [{"role": "user", "content": "MLflow Tracingとは何ですか?"}]}'
"""

# agent.py内の@invokeデコレータを登録するためにインポートが必要
# このインポートによってhandle_request関数が@invoke()で装飾され、
# AgentServerに認識されるエンドポイントハンドラとして登録される
import serving.agent  # noqa: F401

from mlflow.genai.agent_server import (
    AgentServer,
    setup_mlflow_git_based_version_tracking,
)

# AgentServerインスタンスを作成する
# "ResponsesAgent"はエージェントのタイプ名で、Responses API形式を使うことを示す
# 内部でFastAPIアプリケーションが構築される
agent_server = AgentServer("ResponsesAgent")

# FastAPI/ASGIアプリケーションオブジェクトを取得する
# uvicornなどの外部ASGIサーバーから参照するために、モジュールレベルで公開する
# (app_import_stringで "serving.start_server:app" と指定するため必須)
app = agent_server.app

# Gitコミットとトレースを紐付け（任意）
# リポジトリのルートで実行した場合、トレースにコミットハッシュが記録される
# これによりMLflow UIで「このトレースはどのコミットで生成されたか」が追跡可能になる
setup_mlflow_git_based_version_tracking()


def main():
    # app_import_stringを指定することで複数uvicornワーカーをサポート
    # 文字列形式で指定することでワーカープロセスが各自でappをインポートできる
    # (オブジェクトを直接渡す場合はマルチプロセスで共有できない)
    agent_server.run(app_import_string="serving.start_server:app")


if __name__ == "__main__":
    main()
