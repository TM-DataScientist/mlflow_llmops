"""6.3.2節: 本番環境へのデプロイ

段階的デプロイ、ロールバック、タグによるガバナンスのデモ。
エイリアスの切り替えだけでコード変更なしにデプロイ・ロールバックが可能。

実行: make deploy
前提: 02_version_update.pyを実行済みであること
"""

import mlflow

# MLflowのトラッキングサーバーに接続する
mlflow.set_tracking_uri("http://localhost:5000")

print("=== 段階的デプロイ ===\n")

# ステップ1: 新バージョンをまずstagingに向けて限定テストする
mlflow.genai.set_prompt_alias(
    "qa-agent-system-prompt",
    alias="staging",
    version=2,
)
staging = mlflow.genai.load_prompt("prompts:/qa-agent-system-prompt@staging")
print(f"1. stagingにバージョン{staging.version}をデプロイ")

# ステップ2: stagingで問題がなければエイリアスを書き換えてproductionに昇格する
# コードの変更や再デプロイは不要で、エイリアスの向き先を変えるだけで完了する
mlflow.genai.set_prompt_alias(
    "qa-agent-system-prompt",
    alias="production",
    version=2,
)
prod = mlflow.genai.load_prompt("prompts:/qa-agent-system-prompt@production")
print(f"2. productionにバージョン{prod.version}を昇格")

print("\n=== ロールバック ===\n")

# 問題発生時はエイリアスを前バージョンに戻すだけで即座にロールバックできる
mlflow.genai.set_prompt_alias(
    "qa-agent-system-prompt",
    alias="production",
    version=1,
)
rollback = mlflow.genai.load_prompt("prompts:/qa-agent-system-prompt@production")
print(f"3. productionをバージョン{rollback.version}にロールバック")
print("   コード変更・再デプロイ不要で即座に反映されます。")

print("\n=== タグによるガバナンス ===\n")

# tagsにステータスや担当者を記録することでレビューフローを管理できる
# 登録時にtags引数で設定し、承認後のタグ変更はMLflow UIから行う
new_prompt = mlflow.genai.register_prompt(
    name="qa-agent-system-prompt",
    template="(レビュー用のプロンプト)",
    commit_message="レビュー待ちバージョン",
    tags={"status": "pending_review", "author": "dev@example.com"},
)
print(f"4. バージョン{new_prompt.version}をレビュー待ちで登録(tags引数で設定)")
print("   承認後のタグ更新はMLflow UIから行えます。")

# デモ後にproductionを安定バージョンに戻す
mlflow.genai.set_prompt_alias(
    "qa-agent-system-prompt",
    alias="production",
    version=2,
)
print(f"\n最終状態: productionをバージョン2に復元しました")
