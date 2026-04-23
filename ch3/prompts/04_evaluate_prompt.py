"""6.3節: 改善したプロンプトで評価を実行する

Prompt Registryに登録したプロンプトのバージョンごとに評価を実行し、
改善の効果を定量的に比較する。

原稿ではLangGraphAgentを使っているが、ch6は独立構成のためOpenAI APIで直接呼び出す。
エージェント統合版の評価はch5のサンプルコードを参照。

実行: make eval
前提: 02_version_update.pyを実行済み、OPENAI_API_KEYが設定されていること
"""

import mlflow
import openai
from dotenv import load_dotenv
from mlflow.genai.scorers import scorer

# .envファイルからOPENAI_API_KEYなどの環境変数を読み込む
load_dotenv()

# MLflowのトラッキングサーバーと実験名を設定する
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("プロンプト評価")
# OpenAI APIの呼び出しを自動でトレースし、どのプロンプトバージョンを使ったか記録する
mlflow.openai.autolog()

# 評価に使う質問・期待回答のデータセットをインポートする
from data.eval_dataset import EVAL_DATA


def create_predict_fn(prompt_version: str):
    """指定バージョンのプロンプトで予測関数を作成する。"""

    # @mlflow.traceでこの関数の入出力をMLflowのトレースとして記録する
    @mlflow.trace
    def predict_fn(question: str) -> str:
        # Prompt Registryから指定バージョンのプロンプトを取得する
        prompt = mlflow.genai.load_prompt(
            f"prompts:/qa-agent-system-prompt/{prompt_version}"
        )
        # 取得したプロンプトをシステムメッセージとしてOpenAI APIに渡して回答を生成する
        completion = openai.OpenAI().chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": prompt.template},
                {"role": "user", "content": question},
            ],
        )
        return completion.choices[0].message.content

    return predict_fn


# @scorerデコレータでMLflowの評価スコアラーとして登録する
@scorer
def answer_quality(inputs, outputs, expectations):
    """回答が期待される内容をカバーしているか評価する。"""
    expected = expectations.get("expected_answer", "")
    # LLMをジャッジとして使い、回答が期待内容を満たしているか yes/no で判定させる
    response = openai.OpenAI().chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": (
                    "以下の「回答」が「期待される回答」の要点をカバーしているか評価してください。\n\n"
                    f"質問: {inputs.get('question', '')}\n"
                    f"回答: {outputs}\n"
                    f"期待される回答: {expected}\n\n"
                    "要点がカバーされている場合は 'yes'、不十分な場合は 'no' のみ返してください。"
                ),
            }
        ],
    )
    # "yes"なら1(合格)、"no"なら0(不合格)としてスコアを返す
    judgment = response.choices[0].message.content.strip().lower()
    return judgment == "yes"


# バージョン1のプロンプトで全評価データを実行し、スコアをMLflowに記録する
print("=== バージョン1の評価 ===")
results_v1 = mlflow.genai.evaluate(
    data=EVAL_DATA,
    predict_fn=create_predict_fn("1"),
    scorers=[answer_quality],
)
print(f"バージョン1: {results_v1.metrics}")

# バージョン2のプロンプトで同じ評価データを実行し、改善効果を比較する
print("\n=== バージョン2の評価 ===")
results_v2 = mlflow.genai.evaluate(
    data=EVAL_DATA,
    predict_fn=create_predict_fn("2"),
    scorers=[answer_quality],
)
print(f"バージョン2: {results_v2.metrics}")

print("\nMLflow UI (http://localhost:5000) の Evaluation タブで結果を比較してください。")
