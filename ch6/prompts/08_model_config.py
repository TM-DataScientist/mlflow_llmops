"""6.2.6節: モデルパラメータの紐付け

プロンプトと共にモデル名やパラメータを保存し、再現性を高める。

実行: make model-config
前提: MLflow Tracking Serverが起動していること
"""

import mlflow

mlflow.set_tracking_uri("http://localhost:5000")

prompt = mlflow.genai.register_prompt(
    name="qa-prompt",
    template="以下の質問に答えて下さい: {{question}}",
    model_config={
        "model_name": "gpt-4",
        "temperature": 0.7,
        "max_tokens": 1000,
        "top_p": 0.9,
    },
    commit_message="モデルパラメータを追加",
)

# プロンプトとモデルパラメータをロード
loaded = mlflow.genai.load_prompt("prompts:/qa-prompt@latest")
print(f"モデル: {loaded.model_config['model_name']}")
print(f"温度: {loaded.model_config['temperature']}")
