"""6.3.4節: GEPAによるプロンプトの反復最適化

GEPA(Gradient-free Evolutionary Prompt Alignment)はリフレクションを使って
プロンプトを反復的に改善するオプティマイザ。MetaPromptより時間・コストがかかるが、
より高い品質が期待できる。

注意: LLMを多数回呼び出すため、実行コストが高く時間がかかる。
      max_metric_callsで呼び出し回数を制限すること。

実行: make optimize-gepa
前提: 01_register_prompt.pyを実行済み、OPENAI_API_KEYが設定されていること
"""

import time

import mlflow
import openai
from dotenv import load_dotenv
from mlflow.genai.optimize import GepaPromptOptimizer
from mlflow.genai.scorers import scorer

# .envからOPENAI_API_KEYを読み込む
load_dotenv()

# MLflowのトラッキングサーバーと実験名を設定する
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("プロンプト最適化 - GEPA")

# 評価に使う質問・期待回答のデータセットをインポートする
from data.eval_dataset import EVAL_DATA


def predict_fn(question: str) -> str:
    # Prompt Registryから最新バージョンのプロンプトをロードして回答を生成する
    prompt = mlflow.genai.load_prompt("prompts:/qa-agent-system-prompt@latest")
    completion = openai.OpenAI().chat.completions.create(
        model="gpt-5-nano-2025-08-07",
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
        model="gpt-5-nano-2025-08-07",
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


print("GEPAによるプロンプト最適化を開始します...")
print(f"学習データ: {len(EVAL_DATA)}件")
print("(LLMを多数回呼び出すため、完了まで数分かかる場合があります)\n")

# 所要時間を計測するために開始時刻を記録する
start_time = time.time()

# GepaPromptOptimizerでプロンプトを反復最適化する
# GEPAはリフレクションを使ってスコアが向上するまでプロンプトを繰り返し書き直す
# max_metric_callsでLLM呼び出し上限を設定し、コストを制御する
result = mlflow.genai.optimize_prompts(
    predict_fn=predict_fn,
    train_data=EVAL_DATA,
    prompt_uris=["prompts:/qa-agent-system-prompt@latest"],
    optimizer=GepaPromptOptimizer(
        reflection_model="openai:/gpt-5-nano-2025-08-07",
        max_metric_calls=10,
    ),
    scorers=[answer_quality],
)

elapsed = int(time.time() - start_time)
m, s = divmod(elapsed, 60)

print(f"\n完了しました。(所要時間: {m}分{s}秒)")
if result.initial_eval_score is not None:
    print(f"初期スコア: {result.initial_eval_score:.3f}")
if result.final_eval_score is not None:
    print(f"最終スコア: {result.final_eval_score:.3f}")

# 最適化されたプロンプトはPrompt Registryに新バージョンとして自動登録されている
optimized = result.optimized_prompts[0]
print(f"\n最適化されたプロンプト: {optimized.name} (version {optimized.version})")
print(f"テンプレート(先頭200文字):\n{optimized.template[:200]}...")
print("\nMLflow UIのPromptsタブでバージョン比較を確認してください。")
