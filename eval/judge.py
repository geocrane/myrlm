"""Оценка ответов: точное совпадение + LLM-судья на спорных случаях.

Синтетические задачи имеют короткий каноничный ответ (код/город/число), поэтому
сначала пробуем дешёвую нормализованную проверку на вхождение, а к судье
прибегаем только когда строгая проверка не сработала.
"""

from __future__ import annotations

import re
from typing import Any

from rlm.llm_client import LLMClient

JUDGE_SYSTEM_PROMPT = (
    "Ты — строгий проверяющий. Тебе дают вопрос, эталонный ответ и ответ модели. "
    "Скажи, верен ли ответ модели по существу. Ответь РОВНО одним словом: YES или NO."
)


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def exact_match(prediction: str, gold: str) -> bool:
    """Нормализованная проверка вхождения эталона в ответ модели."""
    pred = _normalize(prediction)
    gold_n = _normalize(gold)
    if not gold_n:
        return False
    # Для коротких ответов (код/город/число) ищем как отдельный «токен».
    pattern = r"(?<![\w\d])" + re.escape(gold_n) + r"(?![\w\d])"
    return re.search(pattern, pred) is not None


# Номер договора вида D-1042 (с опциональным дефисом). Для сравнения множеств.
_CONTRACT_RE = re.compile(r"D-?(\d{3,4})", re.IGNORECASE)


def _contract_set(text: str) -> set[str]:
    return {m.group(1) for m in _CONTRACT_RE.finditer(text)}


def set_match(prediction: str, gold: str) -> bool:
    """Сравнение множеств номеров договоров (порядок и формат «D-» не важны).

    Эталон всегда в формате D-XXXX. Если в ответе модели префикса «D-» нет вовсе
    (модель вернула голые числа, напр. список ['2795', ...]) — откатываемся на
    извлечение отдельных 3–4-значных чисел.
    """
    gold_set = _contract_set(gold)
    if not gold_set:
        return False
    pred_set = _contract_set(prediction)
    if not pred_set:
        pred_set = set(re.findall(r"\b\d{3,4}\b", prediction))
    return pred_set == gold_set


def all_match(prediction: str, gold: str) -> bool:
    """Все части эталона (через '|') должны присутствовать в ответе (для join)."""
    pred = _normalize(prediction)
    parts = [p for p in (_normalize(x) for x in gold.split("|")) if p]
    if not parts:
        return False
    return all(p in pred for p in parts)


def judge_with_llm(question: str, gold: str, prediction: str, client: LLMClient) -> bool:
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"ВОПРОС: {question}\nЭТАЛОН: {gold}\nОТВЕТ МОДЕЛИ: {prediction}\n\n"
                "Верен ли ответ модели? YES или NO."
            ),
        },
    ]
    verdict = client.chat(messages, role="judge", temperature=0.0).strip().upper()
    return verdict.startswith("YES")


def evaluate(
    question: str,
    gold: str,
    prediction: str,
    *,
    answer_kind: str = "exact",
    judge_client: LLMClient | None = None,
) -> dict[str, Any]:
    """Вернуть вердикт и каким методом он получен."""
    if answer_kind == "set":
        return {"correct": set_match(prediction, gold), "method": "set"}
    if answer_kind == "all":
        return {"correct": all_match(prediction, gold), "method": "all"}
    if exact_match(prediction, gold):
        return {"correct": True, "method": "exact"}
    if answer_kind == "judge" and judge_client is not None:
        ok = judge_with_llm(question, gold, prediction, judge_client)
        return {"correct": ok, "method": "judge"}
    return {"correct": False, "method": "exact"}
