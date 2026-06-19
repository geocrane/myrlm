"""Бейзлайн «всё в контекст»: кладём весь текст в один промпт и спрашиваем.

Это привычный способ. На сверхдлинных входах он либо не влезает в окно модели,
либо страдает от context rot — что и должно проявиться в сравнении.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from rlm.config import load_config
from rlm.llm_client import LLMClient, count_tokens

NAIVE_SYSTEM_PROMPT = (
    "Ответь на вопрос пользователя, опираясь ТОЛЬКО на приведённый ниже документ. "
    "Если ответа в документе нет — так и скажи. Отвечай кратко."
)


@dataclass
class BaselineResult:
    answer: str
    elapsed: float
    method: str
    usage: dict[str, Any] = field(default_factory=dict)
    note: str = ""


def run_naive(
    question: str,
    context: str,
    *,
    client: LLMClient | None = None,
    config: dict[str, Any] | None = None,
    context_token_budget: int | None = None,
) -> BaselineResult:
    """Один прямой вызов модели со всем контекстом в промпте."""
    config = config or load_config()
    client = client or LLMClient(config)
    # Бюджет согласован с окном модели (config.naive.context_token_budget).
    if context_token_budget is None:
        context_token_budget = config.get("naive", {}).get("context_token_budget", 29000)

    started = time.time()
    note = ""
    ctx = context
    approx_tokens = count_tokens(context)
    if approx_tokens > context_token_budget:
        # Честно отмечаем усечение под бюджет окна (символьная пропорция).
        keep = int(len(context) * context_token_budget / approx_tokens)
        ctx = context[:keep]
        note = f"контекст усечён до ~{context_token_budget} токенов (исходно ~{approx_tokens})"

    messages = [
        {"role": "system", "content": NAIVE_SYSTEM_PROMPT},
        {"role": "user", "content": f"ДОКУМЕНТ:\n{ctx}\n\nВОПРОС: {question}"},
    ]
    answer = client.chat(messages, role="naive")

    return BaselineResult(
        answer=answer,
        elapsed=round(time.time() - started, 2),
        method="naive",
        usage=client.usage.snapshot(),
        note=note,
    )
