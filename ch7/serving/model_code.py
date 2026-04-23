"""models-from-code用のモデル定義ファイル（7.2節）。

MLflow v3のLangChain統合では、モデルオブジェクトを直接渡すのではなく、
このようなコードファイルで定義してmlflow.models.set_model()で登録します。

【models-from-codeパターンとは】
  従来の「pickle/cloudpickle」シリアライズではなく、Pythonコードそのものを
  アーティファクトとして保存し、ロード時にコードを再実行してモデルを再構築する方式。
  依存関係の変更や環境の違いによるデシリアライズエラーを回避できる利点がある。

参考: https://mlflow.org/docs/latest/ml/model/models-from-code/
"""

import mlflow

# 第4章で構築したLangGraphベースのQAエージェントをインポート
from agents.langgraph.agent import LangGraphAgent

# エージェントのインスタンスを作成する
# ここで初期化されるexecutorはLangGraphのCompiledGraphオブジェクト
agent = LangGraphAgent()

# MLflowにモデルとして登録する
# set_model()を呼ぶことで、このファイルがmlflow.langchain.log_model()から
# 参照されたときにMLflowが「これがモデル本体だ」と認識する
mlflow.models.set_model(agent.executor)
