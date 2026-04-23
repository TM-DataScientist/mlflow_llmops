"""評価用データセット

6.3節と6.5節で使用するQAエージェントの評価データ。
MLflowに関する質問と期待される回答のペア。

【データ形式】
  各エントリは以下の構造を持つ辞書:
  {
    "inputs": {"question": "質問テキスト"},
    "expectations": {"expected_answer": "期待する回答テキスト"}
  }

  - inputs.question: predict_fn の引数として渡される質問テキスト
  - expectations.expected_answer: スコアラーが回答の品質を判定する際に参照する期待回答
    （完全一致ではなく、要点のカバレッジを評価するために使われる）

【評価データ設計の指針】
  - 実際のユーザーが質問しそうな多様なカテゴリをカバーする
  - expected_answerは完璧な模範解答ではなく、「最低限この情報が含まれるべき」という指針
  - 難しすぎず簡単すぎない質問で、エージェントの改善効果が観察できるものを選ぶ
"""

# MLflowの主要機能を5つのカテゴリでカバーする評価データセット
EVAL_DATA = [
    {
        "inputs": {
            "question": "MLflow Tracingとは何ですか?",
        },
        "expectations": {
            # 「何ができるか」「どんな情報が記録されるか」「何に使えるか」が含まれれば合格
            "expected_answer": "MLflow TracingはLLMアプリケーションの実行フローを可視化する機能です。各ステップの入出力、レイテンシ、トークン使用量を記録し、デバッグや性能分析に活用できます。",
        },
    },
    {
        "inputs": {
            "question": "MLflowでプロンプトをバージョン管理する方法を教えてください。",
        },
        "expectations": {
            # 「Prompt Registry」「register_prompt()」「エイリアス」が含まれれば合格
            "expected_answer": "MLflow Prompt Registryを使ってプロンプトをバージョン管理できます。mlflow.genai.register_prompt()で登録し、エイリアス(@production, @latestなど)で環境ごとに使い分けられます。",
        },
    },
    {
        "inputs": {
            "question": "MLflow Evaluateの主な機能は何ですか?",
        },
        "expectations": {
            # 「mlflow.genai.evaluate()」「スコアラー」「品質評価」が含まれれば合格
            "expected_answer": "mlflow.genai.evaluate()はLLMアプリケーションの品質を定量評価する機能です。組み込みスコアラー(Correctness, Safety等)やカスタムスコアラーを使って、正確性・安全性・関連性などを自動評価できます。",
        },
    },
    {
        "inputs": {
            "question": "MLflowのautolog機能について説明してください。",
        },
        "expectations": {
            # 「自動トレース」「コード変更なし」「有効化方法」が含まれれば合格
            "expected_answer": "autologはLangChain、OpenAIなどのフレームワークと統合し、コード変更なしで自動的にトレースを記録する機能です。mlflow.langchain.autolog()のように有効化します。",
        },
    },
    {
        "inputs": {
            "question": "MLflow Model Registryとは何ですか?",
        },
        "expectations": {
            # 「バージョン管理」「エイリアス」「ライフサイクル管理」が含まれれば合格
            "expected_answer": "Model Registryはモデルのバージョン管理とライフサイクル管理を行う機能です。モデルの登録、エイリアス(champion/challengerなど)による管理、ステージ遷移ができます。",
        },
    },
]
