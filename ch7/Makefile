.PHONY: install ingest log-model serve test-request eval test-gateway gateway-legacy clean help

help:
	@echo "MLflow Sample Agent - サービング（第7章）"
	@echo ""
	@echo "使用方法:"
	@echo "  make install        依存関係をインストールする"
	@echo "  make ingest         Milvusにドキュメントを取り込む"
	@echo "  make log-model      QAエージェントをMLflowに記録する"
	@echo "  make serve          Agent Serverを起動する"
	@echo "  make test-request   サービングエンドポイントにテストリクエストを送信する"
	@echo "  make eval           サービング中のエージェントを評価する"
	@echo "  make test-gateway   AI Gateway経由でテストリクエストを送信する(新方式)"
	@echo "  make gateway-legacy AI Gatewayを起動する(Legacy方式)"
	@echo "  make clean          生成されたファイルを削除する"

install:
	uv sync

ingest:
	uv run python scripts/web_ingest.py

log-model:
	uv run python -m serving.log_model

serve:
	uv run python serving/start_server.py --reload --port 5005

test-request:
	curl -s -X POST http://localhost:5005/invocations \
		-H "Content-Type: application/json" \
		-d '{"input": [{"role": "user", "content": "MLflow Tracingとは何ですか?"}]}' | python3 -m json.tool --no-ensure-ascii

eval:
	uv run python -m serving.eval_serving

# 新AI Gateway（Tracking Server統合型）のテスト
# 事前にMLflow UI (http://localhost:5000/#/gateway) でエンドポイントを作成してください
test-gateway:
	@echo "AI Gateway経由でテストリクエストを送信します..."
	@echo "※ 事前にMLflow UI (http://localhost:5000/#/gateway) で"
	@echo "  qa-agent-llmエンドポイントを作成してください"
	@echo ""
	curl -s -X POST http://localhost:5000/gateway/qa-agent-llm/mlflow/invocations \
		-H "Content-Type: application/json" \
		-d '{"messages": [{"role": "user", "content": "hello"}], "max_tokens": 50}' | python3 -m json.tool --no-ensure-ascii

# Legacy方式: 別プロセスとしてAI Gatewayを起動（YAML設定ファイル使用）
# 新方式（Tracking Server統合型）ではこのターゲットは不要です
gateway-legacy:
	set -a && . ./.env && set +a && uv run mlflow gateway start --config-path gateway/gateway_config.yaml --port 5010

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
