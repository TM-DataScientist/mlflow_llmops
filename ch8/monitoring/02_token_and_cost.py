"""
第8章 - 8.2 トークン使用量とコストの可視化

自動トレーシングによるトークン使用量の追跡と、
コスト計算の基本的な流れを確認します。

【このスクリプトで学ぶこと】
  1. autolog()で自動的に記録されるトークン使用量の取得方法
  2. トレースのタグにコスト情報を付与する方法
  3. タグを条件にトレースを検索してコストを集計する方法

実行方法:
  make cost
  または
  uv run python monitoring/02_token_and_cost.py
"""

from dotenv import load_dotenv

# .envからOPENAI_API_KEYを読み込む
load_dotenv()

import mlflow
from mlflow import MlflowClient
from openai import OpenAI

# cost_calculator.py のコスト計算関数をインポートする
# 同ディレクトリに配置されているユーティリティモジュール
from cost_calculator import calculate_cost

# 実験名を設定する（01_tracing_setup.pyと共通の実験に集約する）
mlflow.set_experiment("ch8-monitoring-quickstart")

# OpenAI API呼び出しの自動トレーシングを有効化する
# トークン使用量（input_tokens, output_tokens）が自動的にトレースに記録される
mlflow.openai.autolog()

client = OpenAI()

# MlflowClientは完了済みトレースへのタグ書き込みに使用する
# mlflow.set_tag() はアクティブなRunに対して使うものだが、
# トレースにタグを付ける場合は MlflowClient.set_trace_tag() を使う
mlflow_client = MlflowClient()

# === 1. 異なるモデルでLLM呼び出し ===
# (モデル名, プロンプト) のタプルリスト
# 同じモデルを複数回呼び出すことでトークン使用量の分布を確認できる
models_and_prompts = [
    ("gpt-5-nano-2025-08-07", "Pythonのデコレータを簡潔に説明してください。"),
    ("gpt-5-nano-2025-08-07", "MLflowのトレーシング機能について、主な利点を3つ挙げてください。"),
    (
        "gpt-5-nano-2025-08-07",
        "RAGシステムの品質評価で重要な指標は何ですか?詳しく説明してください。",
    ),
]

print("=== トークン使用量とコスト ===\n")
total_cost = 0.0

for model, prompt in models_and_prompts:
    # LLM API呼び出し: autolog()により自動的にトレースが記録される
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )

    # 直前のAPI呼び出しに対応するトレースIDを取得する
    # autolog()はLLM呼び出し完了時に自動でトレースを作成し、そのIDを記録する
    trace_id = mlflow.get_last_active_trace_id()

    # トレースIDを使ってトレースオブジェクトを取得する
    trace = mlflow.get_trace(trace_id=trace_id)

    # トレースのトークン使用量情報を取得する
    # token_usage は {"input_tokens": int, "output_tokens": int, "total_tokens": int} の形式
    usage = trace.info.token_usage

    # 各トークン数を取得（キーが存在しない場合は0をデフォルト値として使用）
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    total_tokens = usage.get("total_tokens", 0)

    # cost_calculator.pyのcalculate_cost()でトークン数からAPIコストを計算する
    # モデル名・入力トークン数・出力トークン数から料金テーブルを参照して計算
    cost = calculate_cost(model, input_tokens, output_tokens)
    total_cost += cost

    # コスト情報をトレースのタグとして記録する
    # 完了済みトレースへのタグ書き込みはMlflowClient.set_trace_tag()を使う
    # （mlflow.set_tag()はアクティブなRunに対してのみ使えるため使用不可）
    mlflow_client.set_trace_tag(trace_id, "cost.total_usd", f"{cost:.6f}")
    # コストの計算に使用したモデル名も記録しておく（検索・集計時のキーとして使用）
    mlflow_client.set_trace_tag(trace_id, "cost.model", model)

    # 結果を表示する（プロンプトは先頭40文字のみ表示）
    print(f"プロンプト: {prompt[:40]}...")
    print(f"  モデル: {model}")
    print(f"  入力トークン: {input_tokens}")
    print(f"  出力トークン: {output_tokens}")
    print(f"  合計トークン: {total_tokens}")
    print(f"  コスト: ${cost:.6f}")
    print()

print(f"--- 合計コスト: ${total_cost:.6f} ---")

# === 2. トレースを検索してコスト集計 ===
print("\n=== コスト集計(トレース検索) ===")

# cost.total_usd タグが付いているトレースだけを絞り込んで取得する
# filter_string で != '' を使うことで、タグが存在するトレースのみが返される
traces = mlflow.search_traces(
    filter_string="tags.`cost.total_usd` != ''",
    max_results=100,
)

if len(traces) > 0:
    # search_traces()の戻り値はPandas DataFrameで、"tags"列はdictを含む
    # apply()でrows全体にラムダ関数を適用し、cost.total_usd の値を浮動小数点数に変換する
    costs = traces["tags"].apply(lambda t: float(t.get("cost.total_usd", 0)))
    print(f"  トレース数: {len(costs)}")
    print(f"  合計コスト: ${costs.sum():.6f}")
    print(f"  平均コスト: ${costs.mean():.6f}")
    print(f"  最大コスト: ${costs.max():.6f}")
else:
    print("  コスト付きトレースが見つかりませんでした。")

print("\n✅ コスト分析完了。MLflow UIの各トレースで cost.* タグを確認できます。")
