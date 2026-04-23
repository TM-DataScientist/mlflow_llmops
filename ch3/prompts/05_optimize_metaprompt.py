"""6.3.4節: MetaPromptによるプロンプトの構造改善

MetaPromptOptimizerは少量のデータ(またはゼロショット)でも動作し、
プロンプトの構造を改善する。実行は高速で、LLM呼び出しは1回程度。

実行: make optimize-meta
前提: 01_register_prompt.pyを実行済み、OPENAI_API_KEYが設定されていること
"""

import mlflow
import openai
from dotenv import load_dotenv
from mlflow.genai.optimize import MetaPromptOptimizer
from mlflow.genai.scorers import scorer

# .envからOPENAI_API_KEYを読み込む
load_dotenv()

# MLflowのトラッキングサーバーと実験名を設定する
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("プロンプト最適化 - MetaPrompt")

# 評価に使う質問・期待回答のデータセットをインポートする
from data.eval_dataset import EVAL_DATA


def predict_fn(question: str) -> str:
    # Prompt Registryから最新バージョンのプロンプトをロードして回答を生成する
    prompt = mlflow.genai.load_prompt("prompts:/qa-agent-system-prompt@latest")
    completion = openai.OpenAI().chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": prompt.template},
            {"role": "user", "content": question},
        ],
    )
    return completion.choices[0].message.content


# LLMをジャッジとして使い、回答が期待内容をカバーしているかをyes/noで判定するスコアラー
@scorer
def answer_quality(inputs, outputs, expectations):
    expected = expectations.get("expected_answer", "")
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
    return response.choices[0].message.content.strip().lower() == "yes"


print("MetaPromptによるプロンプト最適化を開始します...")
print("(LLM呼び出し1回程度で高速に完了します)\n")

# MetaPromptOptimizerでプロンプトを最適化する
# reflection_modelがプロンプトの構造を分析し、改善案を1回のLLM呼び出しで生成する
# 最適化結果は自動的にPrompt Registryに新バージョンとして登録される
result = mlflow.genai.optimize_prompts(
    predict_fn=predict_fn,
    train_data=EVAL_DATA,
    prompt_uris=["prompts:/qa-agent-system-prompt@latest"],
    optimizer=MetaPromptOptimizer(
        reflection_model="openai:/gpt-4o",
    ),
    scorers=[answer_quality],
)

if result.initial_eval_score is not None:
    print(f"初期スコア: {result.initial_eval_score:.3f}")
if result.final_eval_score is not None:
    print(f"最適化後スコア: {result.final_eval_score:.3f}")

# 最適化されたプロンプトはPrompt Registryに新バージョンとして自動登録されている
optimized = result.optimized_prompts[0]
print(f"\n最適化されたプロンプト: {optimized.name} (version {optimized.version})")
print(f"テンプレート(先頭200文字):\n{optimized.template[:200]}...")
print("\nMLflow UI のPromptsタブでバージョン比較を確認してください。")
