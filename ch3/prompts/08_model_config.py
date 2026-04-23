"""6.2.6節: モデルパラメータの紐付け

プロンプトと共にモデル名やパラメータを保存し、再現性を高める。

【model_configとは】
  プロンプトのテンプレートだけでなく、推論時に使用するモデル名・温度・
  最大トークン数などのパラメータも一緒に保存する機能。
  プロンプトをロードすると model_config も一緒に取り出せるため、
  「どのモデルをどの設定で使ったか」が完全に再現可能になる。

【再現性が重要な理由】
  プロンプトとモデル設定が別管理だと、後から「このバージョンのプロンプトを
  どのモデルで使ったか」がわからなくなる。model_configに記録することで
  実験の再現性が向上する。

実行: make model-config
前提: MLflow Tracking Serverが起動していること
"""

import mlflow

# MLflowのトラッキングサーバーに接続する
mlflow.set_tracking_uri("http://localhost:5000")

# model_config引数でモデル名・パラメータをプロンプトと一緒に保存する
# こうすることでプロンプトとモデル設定をセットで管理・再現できる
prompt = mlflow.genai.register_prompt(
    name="qa-prompt",
    template="以下の質問に答えて下さい: {{question}}",
    model_config={
        # 使用するLLMモデル名（このモデル名でAPIを呼び出すことが前提）
        "model_name": "gpt-5-nano-2025-08-07",
        # 生成の多様性を制御する温度パラメータ（0=確定的、1=ランダム）
        "temperature": 0.7,
        # 1回の応答で生成できる最大トークン数
        "max_tokens": 1000,
        # nucleus sampling（上位確率のトークンのみを使う）の閾値
        "top_p": 0.9,
    },
    commit_message="モデルパラメータを追加",
)

# @latestエイリアスで最新バージョンのプロンプトとmodel_configを取得する
loaded = mlflow.genai.load_prompt("prompts:/qa-prompt@latest")

# model_configはdictとして取得できる
print(f"モデル: {loaded.model_config['model_name']}")
print(f"温度: {loaded.model_config['temperature']}")
