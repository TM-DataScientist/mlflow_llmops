"""会話スレッド管理モジュール。

LangGraphエージェントの会話状態（メッセージ履歴）を管理するデータクラスを定義する。
CLIやAPIサーバーからエージェントを呼び出す際に、同一スレッド内の会話文脈を
維持するために使用される。

【設計方針】
  - Threadはシリアライズ・DBへの永続化を行わないインメモリ管理に特化している。
  - スレッドIDはUUIDで自動生成されるため、呼び出し側がIDを意識する必要がない。
  - 複数の会話を同時並行で管理する場合は、それぞれ別のThreadインスタンスを作成する。

【LangGraphとの連携】
  agent.py では thread.id を LangGraph の thread_id として使用する。
  LangGraph は thread_id を使って MemorySaver（チェックポインタ）内の
  会話ステートを識別・保存するため、同じ thread_id を渡し続けることで
  複数ターンにわたる会話文脈が保持される。
"""
from dataclasses import dataclass, field
from typing import List
import uuid


@dataclass
class Message:
    """1つの会話メッセージを表すデータクラス。

    OpenAI Chat Completions API のメッセージ形式に対応している。

    Attributes:
        role: メッセージの送信者役割。以下のいずれかを取る:
            - "system"    : エージェントの振る舞いを定義するシステムプロンプト
            - "user"      : エンドユーザーの発言
            - "assistant" : エージェント（LLM）の回答
        content: メッセージの本文テキスト。
    """
    role: str
    content: str


@dataclass
class Thread:
    """1つの会話セッションを表すデータクラス。

    Attributes:
        id: スレッドの一意識別子（UUID v4）。
            LangGraph の MemorySaver がこの ID でチェックポイントを管理する。
            デフォルトで自動生成されるため、通常は明示的に指定しなくてよい。
        messages: この会話セッション内の全メッセージ履歴リスト。
            最初にSystemMessageが追加され、以降はuserとassistantのメッセージが
            交互に追加される。
    """
    # uuid.uuid4() で毎回異なるUUIDを生成。field(default_factory=...) を使うことで
    # dataclassのデフォルト引数として関数（callable）を指定できる。
    # lambda でラップするのは、uuid.uuid4() を呼び出し時ではなくインスタンス生成時に
    # 評価させるため。
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # メッセージ履歴の初期値は空リスト。field(default_factory=list) を使うことで
    # クラス変数として [] を共有する「ミュータブルデフォルト引数の罠」を回避できる。
    messages: List[Message] = field(default_factory=list)
