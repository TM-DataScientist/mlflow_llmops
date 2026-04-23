"""6.2.7節: 構造化出力(Structured Output)の利用

response_formatパラメータで期待される出力形式を定義し、
OpenAI APIで構造化出力を取得する。

【構造化出力とは】
  LLMの出力を自由なテキストではなく、あらかじめ定義したスキーマ（型）に
  従ったJSON形式で取得する機能。OpenAI APIのresponse_formatパラメータで
  Pydanticモデルを渡すことで実現できる。

【メリット】
  - 後処理が容易（JSONパース不要、型安全にアクセスできる）
  - 必須フィールドの欠落を防げる（スキーマ違反はAPIがエラーを返す）
  - 複雑な情報抽出タスクに適している

【Prompt Registryとの連携】
  response_format引数にPydanticモデルを渡してプロンプトを登録することで、
  「このプロンプトはQAResponse形式で回答する」という設計情報も一緒に管理できる。

実行: make structured
前提: OPENAI_API_KEYが設定されていること
"""

from typing import List

import mlflow
import openai
from dotenv import load_dotenv
from pydantic import BaseModel

# .envからOPENAI_API_KEYを読み込む
load_dotenv()

# MLflowのトラッキングサーバーに接続する
mlflow.set_tracking_uri("http://localhost:5000")


# 期待する出力形式をPydanticモデルで定義する
# Pydanticのフィールド定義が、OpenAI APIへのJSONスキーマとして自動変換される
class QAResponse(BaseModel):
    """QAエージェントの回答スキーマ。

    Attributes:
        answer: 回答本文（文字列）
        confidence: モデルの確信度（0.0〜1.0の浮動小数点数）
        sources: 情報源のリスト（URL等）
    """
    answer: str
    confidence: float
    sources: List[str]


# response_format引数にPydanticモデルを渡すことで、
# 期待する出力スキーマをプロンプトと一緒にPrompt Registryに保存する。
# ロード時にもresponse_formatが復元されるため、スキーマを別途管理する必要がない。
prompt = mlflow.genai.register_prompt(
    name="qa-prompt",
    template="次の質問に回答して下さい: {{question}}",
    # Pydanticモデルクラスを渡すと、JSONスキーマに変換されて保存される
    response_format=QAResponse,
    commit_message="構造化出力を追加",
)
print(f"プロンプト '{prompt.name}' (version {prompt.version}) を登録しました")

# Prompt Registryから最新バージョンをロードする
loaded = mlflow.genai.load_prompt("prompts:/qa-prompt@latest")

# beta.chat.completions.parse を使うとOpenAI APIが構造化出力を返す。
# 通常の chat.completions.create とは異なり、parsedフィールドに
# Pydanticモデルのインスタンスが格納されている。
response = openai.OpenAI().beta.chat.completions.parse(
    model="gpt-5-nano-2025-08-07",
    messages=[
        {
            "role": "user",
            # loaded.format(question=...) でテンプレート変数を埋め込む
            "content": loaded.format(question="MLflowとは何ですか?"),
        }
    ],
    # response_format に Pydantic クラスを渡してスキーマを指定する
    response_format=QAResponse,
)

# parsed フィールドに QAResponse インスタンスが格納されている
# （テキストではなく型付きオブジェクトとして取得できる）
result = response.choices[0].message.parsed
print(f"\n構造化出力の結果:")
print(f"  回答: {result.answer}")
print(f"  確信度: {result.confidence}")
print(f"  ソース: {result.sources}")
