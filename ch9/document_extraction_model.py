# ---- このセルの見どころ ----
# このセルはPythonファイルを書き出します。MLflowのModels from Codeでは、このファイル自体がモデル定義になります。
# 処理の流れは、Prompt Registryからテンプレートを読む -> 入力文書を埋め込む -> OpenAIにJSON抽出させる -> DataFrameで返す、です。
# @mlflow.traceを付けた関数はMLflow Tracing上でスパンとして記録され、プロンプト整形やLLM呼び出しを後から追跡できます。

import json
from typing import Any, Dict, Optional

import pandas as pd
import mlflow
from mlflow.pyfunc import PythonModel
from mlflow.models import set_model
from mlflow.entities import SpanType

from openai import OpenAI

# デフォルト設定
DEFAULT_PROMPT_URI = "prompts:/document-extraction-system/1"
DEFAULT_LLM_MODEL = "gpt-5-nano-2025-08-07"

# OpenAI APIの呼び出しを自動的にMLflowに記録
mlflow.openai.autolog()

def _load_prompt(prompt_uri: str):
    """
    Prompt Registryからプロンプトをロードする関数
    URI形式: prompts://<プロンプト名>/<バージョン>
    """
    return mlflow.genai.load_prompt(prompt_uri)

@mlflow.trace(span_type=SpanType.TOOL)
def _render_prompt(prompt, text: str) -> str:
    """
    プロンプトテンプレートに実際のテキストを埋め込む関数
    {{text}}プレースホルダーを入力テキストで置換します
    """
    try:
        return prompt.format(text=text)
    except Exception:
        # フォールバック: format()が使えない場合は文字列置換
        tmpl = getattr(prompt, "template", None)
        if isinstance(tmpl, str):
            return tmpl.replace("{{text}}", text)
        return str(prompt).replace("{{text}}", text)

@mlflow.trace(span_type=SpanType.LLM)
def _call_llm_return_json(*, client, prompt_text: str, model: str, max_tokens: int) -> Dict[str, Any]:
    """
    OpenAI APIを呼び出し、JSON形式で結果を返す関数

    Args:
        client: OpenAIクライアント
        prompt_text: 生成済みのプロンプト全文
        model: 使用するLLMモデル名
        max_tokens: 最大トークン数
        # GPT-5系の一部モデルはtemperatureを指定できないため、APIには渡しません。

    Returns:
        抽出された情報を含む辞書
    """
    # GPT-5系の一部モデルはtemperatureの明示指定を受け付けないため、
    # デフォルト値を使い、temperatureパラメータは送信しません。
    res = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Return ONLY valid JSON (json_object)."},
            {"role": "user", "content": prompt_text},
        ],
        max_completion_tokens=max_tokens,
        response_format={"type": "json_object"},  # JSON形式を強制
    )
    content = res.choices[0].message.content
    return json.loads(content)


class DocumentExtractionModel(PythonModel):
    """
    ビジネス文書から情報を抽出するカスタムMLflowモデル

    入力形式: pandas.DataFrame
        - 必須カラム: 'text' (抽出対象のテキスト)
        - オプションカラム: 'model' (使用するLLMモデル名)

    出力形式: pandas.DataFrame
        - 抽出されたJSON項目が各カラムとして返される
    """

    def load_context(self, context):
        """
        モデルロード時に1回だけ実行される初期化メソッド
        設定の読み込み、クライアントの初期化、プロンプトのロードを行います
        """
        # model_configから設定を取得
        self.cfg = getattr(context, "model_config", {}) or {}

        # OpenAIクライアントを初期化（環境変数からAPIキーを取得）
        self.client = OpenAI()

        # 設定値を取得（デフォルト値を指定）
        self.prompt_uri = self.cfg.get("prompt_uri", DEFAULT_PROMPT_URI)
        self.default_model = self.cfg.get("default_model", DEFAULT_LLM_MODEL)
        self.max_tokens = int(self.cfg.get("max_tokens", 1024))

        # Prompt Registryからプロンプトをロード
        self.prompt = _load_prompt(self.prompt_uri)

    @mlflow.trace(span_type=SpanType.CHAIN)
    def predict(self, context, model_input, params=None):
        """
        推論メソッド - 入力テキストから情報を抽出します

        処理フロー:
        1. 入力の検証とDataFrame化
        2. 各行に対してループ処理
        3. プロンプトのレンダリング
        4. LLM呼び出し
        5. 結果の収集とDataFrame化
        """
        # 入力がDataFrameでない場合は変換
        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input)

        # 'text'カラムの存在を確認
        if "text" not in model_input.columns:
            raise ValueError("Input must contain column 'text'.")

        rows = []
        # 各行を処理
        for _, row in model_input.iterrows():
            text = str(row.get("text", ""))
            model = str(row.get("model", self.default_model))

            # プロンプトに実際のテキストを埋め込み
            prompt_text = _render_prompt(self.prompt, text)

            # LLMを呼び出して情報抽出
            extracted = _call_llm_return_json(
                client=self.client,
                prompt_text=prompt_text,
                model=model,
                max_tokens=self.max_tokens,
            )
            rows.append(extracted)

        return pd.DataFrame(rows)


# ★Models from Codeの重要なポイント★
# set_model()を呼び出して、このファイル全体をMLflowモデルとして認識させます
app = DocumentExtractionModel()
set_model(app)
