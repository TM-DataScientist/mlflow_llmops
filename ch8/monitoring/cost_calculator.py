"""
コスト計算ユーティリティ

トークン使用量からAPI費用を計算します。
料金は頻繁に更新されるため、最新の公式料金を確認してください。

参考:
  - OpenAI: https://openai.com/api/pricing/
  - Anthropic: https://www.anthropic.com/pricing
"""

# 料金テーブル (per 1K tokens, USD)
# 最終更新: 2025年1月
# 注意: これらの値はスクリプト実行時の参照用であり、
#       最新料金は各プロバイダーの公式サイトで確認すること
MODEL_PRICING: dict[str, dict[str, float]] = {
    # OpenAI モデルの料金設定
    # input: プロンプト（入力）トークン1K個あたりのコスト(USD)
    # output: 補完（出力）トークン1K個あたりのコスト(USD)
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4o-2024-11-20": {"input": 0.0025, "output": 0.01},
    "gpt-4.1": {"input": 0.002, "output": 0.008},
    "gpt-4.1-mini": {"input": 0.0004, "output": 0.0016},
    "gpt-4.1-nano": {"input": 0.0001, "output": 0.0004},
    # サンプルコードで使用するモデル（gpt-4.1-nanoと同等の料金を設定）
    "gpt-5-nano-2025-08-07": {"input": 0.0001, "output": 0.0004},
    "o3-mini": {"input": 0.0011, "output": 0.0044},
    # Anthropic モデルの料金設定
    "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
    "claude-3-5-haiku-20241022": {"input": 0.0008, "output": 0.004},
}

# モデル名のエイリアスマッピング
# APIが返すモデル名（バージョン付き）を、料金テーブルのキー（バージョンなし）に変換する。
# OpenAIは "gpt-4o-mini-2024-07-18" のようなバージョン付きの名前を返すことがあるため、
# 料金テーブルに存在しない名前でも正しくコストを計算できるよう正規化する。
MODEL_ALIASES: dict[str, str] = {
    "gpt-4o-mini-2024-07-18": "gpt-4o-mini",
    "gpt-4o-2024-08-06": "gpt-4o",
    "claude-3-5-sonnet": "claude-sonnet-4-20250514",
    "claude-3-5-haiku": "claude-3-5-haiku-20241022",
    "claude-3.5-sonnet": "claude-sonnet-4-20250514",
    "claude-3.5-haiku": "claude-3-5-haiku-20241022",
}


def resolve_model_name(model: str) -> str:
    """モデル名のエイリアスを解決する。

    バージョン付きモデル名（例: "gpt-4o-mini-2024-07-18"）を
    料金テーブルのキー（例: "gpt-4o-mini"）に変換する。
    エイリアスが見つからない場合はそのまま返す。

    Args:
        model: 解決前のモデル名（APIが返したままの名前）

    Returns:
        料金テーブルのキーに対応するモデル名
    """
    return MODEL_ALIASES.get(model, model)


def calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
) -> float:
    """トークン使用量からAPIコスト(USD)を計算する。

    入力トークンと出力トークンの料金を別々に計算し合算する。
    キャッシュ済みトークンは通常料金の50%で計算される（プロバイダーによる）。

    Args:
        model: モデル名（バージョン付きも可）
        input_tokens: 合計入力トークン数（キャッシュ済みトークンを含む）
        output_tokens: 出力トークン数
        cached_input_tokens: キャッシュ済み入力トークン数（デフォルト: 0）
                             入力トークン全体のうち、キャッシュから取得した分

    Returns:
        推定APIコスト(USD)。料金テーブルにないモデルの場合は0.0を返す。
    """
    # エイリアスを解決して料金テーブルのキーに対応するモデル名を取得する
    resolved = resolve_model_name(model)
    pricing = MODEL_PRICING.get(resolved)
    if pricing is None:
        # 料金テーブルに存在しないモデルの場合はコスト0を返す（計算不能）
        return 0.0

    # キャッシュ入力トークンは通常料金の50%割引で計算する
    # 通常トークン = 全入力 - キャッシュ済み入力
    regular_input = input_tokens - cached_input_tokens
    # キャッシュ済みトークンのコスト（50%割引）
    cached_cost = (cached_input_tokens / 1000) * pricing["input"] * 0.5
    # 通常トークンのコスト（割引なし）
    regular_cost = (regular_input / 1000) * pricing["input"]
    # 出力トークンのコスト
    output_cost = (output_tokens / 1000) * pricing["output"]

    return regular_cost + cached_cost + output_cost


def format_cost_report(
    model: str, input_tokens: int, output_tokens: int, cost: float
) -> str:
    """コストレポートの文字列を生成する。

    Args:
        model: モデル名
        input_tokens: 入力トークン数
        output_tokens: 出力トークン数
        cost: 計算済みコスト(USD)

    Returns:
        整形されたコストレポート文字列
    """
    return (
        f"Model: {model}\n"
        f"  Input:  {input_tokens:>8,} tokens\n"  # カンマ区切り・右詰め8桁
        f"  Output: {output_tokens:>8,} tokens\n"
        f"  Cost:   ${cost:.6f}"  # 小数点6桁まで表示（マイクロドル単位）
    )
