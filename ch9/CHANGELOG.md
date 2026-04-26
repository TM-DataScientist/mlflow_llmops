# CHANGE LOG

## 2026-04-26

- `9.2.ipynb`: `OpenAIEmbeddings`の埋め込みモデルを`text-embedding-3-small`に明示しました。
  - 理由: モデル名を省略するとLangChainのデフォルト値に依存するため、教材としての再現性が下がります。ベクトルDB作成時と検索時で同じ埋め込みモデルを使う必要があることもNotebookに追記しました。
  - 根拠:
    - OpenAI Embeddings guide: https://platform.openai.com/docs/guides/embeddings
    - LangChain OpenAIEmbeddings API reference: https://api.python.langchain.com/en/latest/openai/embeddings/langchain_openai.embeddings.base.OpenAIEmbeddings.html
