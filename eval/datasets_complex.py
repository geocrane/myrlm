"""Сложный многодокументный датасет «договоры».

Контекст = набор карточек договоров (поля: №, контрагент, категория, сумма, дата,
ответственный, статус), разбавленных шумом и добитых до целевой длины. В отличие от
простого датасета (`datasets.py`, единый текст), здесь данные структурированы и
требуют РЕАЛЬНОЙ обработки: фильтрация по диапазону сумм, объединение полей (join),
отбор документов с нарушением — по явному флагу и по вычислимому правилу.

Это профиль RLM: всё решается кодом (распарсить карточки, отфильтровать, посчитать).
RAG (одноразовый top-k) и naive (предел окна) тут должны заметно проигрывать.

Типы задач:
  - range_count      — сколько договоров на сумму в диапазоне [X, Y];
  - join_lookup      — контрагент И ответственный по конкретному договору;
  - violation_flag   — номера договоров со статусом «нарушение» (явный флаг);
  - violation_rule   — номера договоров, где сумма > лимита их категории (правило).

Детерминированы по seed. Возвращают тот же `Task`, что и `datasets.py`, поэтому
стенд и Excel-отчёт работают без изменений.
"""

from __future__ import annotations

import random

from .datasets import Task, _FILLERS, _PEOPLE

# Контрагенты и категории с лимитами (₽) для правила нарушений.
_COUNTERPARTIES = [
    "Вектор", "Гранит", "Сфера", "Орион", "Кедр", "Меридиан",
    "Аксиома", "Темп", "Полюс", "Ритм", "Атлант", "Зенит",
]
_CATEGORY_LIMITS = {
    "Поставка": 500_000,
    "Услуги": 300_000,
    "Аренда": 1_000_000,
    "Подряд": 750_000,
}
_CATEGORIES = list(_CATEGORY_LIMITS)
_NEUTRAL_STATUSES = ["действует", "исполнен", "расторгнут", "на согласовании"]

N_CONTRACTS = 24  # сколько реальных карточек в каждой задаче (остальное — шум)


def _render_card(c: dict) -> str:
    return (
        f"=== ДОГОВОР №{c['num']} ===\n"
        f"Контрагент: ООО «{c['cp']}»\n"
        f"Категория: {c['category']}\n"
        f"Сумма: {c['amount']} ₽\n"
        f"Дата: {c['date']}\n"
        f"Ответственный: {c['resp']}\n"
        f"Статус: {c['status']}"
    )


def _gen_contracts(rng: random.Random, n: int, *, with_flag: bool) -> list[dict]:
    """Сгенерировать n уникальных карточек договоров."""
    nums = set()
    cards = []
    for _ in range(n):
        while True:
            num = f"D-{rng.randint(1000, 9999)}"
            if num not in nums:
                nums.add(num)
                break
        cat = rng.choice(_CATEGORIES)
        amount = rng.randint(50, 1500) * 1000          # 50k..1.5M, кратно 1000
        status = "нарушение" if (with_flag and rng.random() < 0.3) else rng.choice(_NEUTRAL_STATUSES)
        cards.append({
            "num": num,
            "cp": rng.choice(_COUNTERPARTIES),
            "category": cat,
            "amount": amount,
            "date": f"2024-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
            "resp": rng.choice(_PEOPLE),
            "status": status,
        })
    return cards


def _regulation_block() -> str:
    lines = ["=== РЕГЛАМЕНТ ЛИМИТОВ ==="]
    for cat, lim in _CATEGORY_LIMITS.items():
        lines.append(f"Лимит по категории «{cat}» — {lim} ₽.")
    return "\n".join(lines)


def _assemble(rng: random.Random, blocks: list[str], target_chars: int) -> str:
    """Перемешать карточки/блоки с шумом-наполнителем до целевой длины."""
    pieces = list(blocks)
    while sum(len(p) for p in pieces) + len(pieces) < target_chars:
        pieces.append(rng.choice(_FILLERS))
    rng.shuffle(pieces)
    return "\n\n".join(pieces)


def _make_range_count(rng, target_chars, idx) -> Task:
    cards = _gen_contracts(rng, N_CONTRACTS, with_flag=False)
    amounts = sorted(c["amount"] for c in cards)
    # Границы выбираем так, чтобы в диапазон гарантированно попало несколько договоров.
    lo = amounts[rng.randint(0, len(amounts) // 3)]
    hi = amounts[rng.randint(2 * len(amounts) // 3, len(amounts) - 1)]
    gold = sum(1 for c in cards if lo <= c["amount"] <= hi)
    ctx = _assemble(rng, [_render_card(c) for c in cards], target_chars)
    return Task(
        id=f"range_{idx}",
        type="range_count",
        question=f"Сколько договоров заключено на сумму от {lo} до {hi} ₽ включительно? Ответь одним числом.",
        answer=str(gold),
        context=ctx,
        char_len=len(ctx),
        answer_kind="exact",
    )


def _make_join_lookup(rng, target_chars, idx) -> Task:
    cards = _gen_contracts(rng, N_CONTRACTS, with_flag=False)
    target = rng.choice(cards)
    ctx = _assemble(rng, [_render_card(c) for c in cards], target_chars)
    return Task(
        id=f"join_{idx}",
        type="join_lookup",
        question=(
            f"Назови контрагента и ответственного по договору №{target['num']}. "
            "Укажи оба значения."
        ),
        # gold: два значения через | для проверки «оба присутствуют» (answer_kind=all).
        answer=f"{target['cp']} | {target['resp']}",
        context=ctx,
        char_len=len(ctx),
        answer_kind="all",
    )


def _make_violation_flag(rng, target_chars, idx) -> Task:
    cards = _gen_contracts(rng, N_CONTRACTS, with_flag=True)
    violators = [c["num"] for c in cards if c["status"] == "нарушение"]
    if len(violators) < 2:  # гарантируем минимум для осмысленной задачи
        for c in rng.sample(cards, 2):
            c["status"] = "нарушение"
        violators = [c["num"] for c in cards if c["status"] == "нарушение"]
    ctx = _assemble(rng, [_render_card(c) for c in cards], target_chars)
    return Task(
        id=f"viol_flag_{idx}",
        type="violation_flag",
        question=(
            "Перечисли номера всех договоров (вида D-XXXX) со статусом «нарушение». "
            "Перечисли через запятую."
        ),
        answer=", ".join(sorted(violators)),
        context=ctx,
        char_len=len(ctx),
        answer_kind="set",
    )


def _make_violation_rule(rng, target_chars, idx) -> Task:
    cards = _gen_contracts(rng, N_CONTRACTS, with_flag=False)
    violators = [c["num"] for c in cards if c["amount"] > _CATEGORY_LIMITS[c["category"]]]
    if len(violators) < 2:  # подкрутим пару карточек, чтобы нарушение точно было
        for c in rng.sample(cards, 2):
            c["amount"] = _CATEGORY_LIMITS[c["category"]] + rng.randint(1, 300) * 1000
        violators = [c["num"] for c in cards if c["amount"] > _CATEGORY_LIMITS[c["category"]]]
    blocks = [_render_card(c) for c in cards] + [_regulation_block()]
    ctx = _assemble(rng, blocks, target_chars)
    return Task(
        id=f"viol_rule_{idx}",
        type="violation_rule",
        question=(
            "По регламенту договор нарушает лимит, если его сумма превышает лимит его "
            "категории. Перечисли номера всех договоров (вида D-XXXX), нарушающих лимит. "
            "Перечисли через запятую."
        ),
        answer=", ".join(sorted(violators)),
        context=ctx,
        char_len=len(ctx),
        answer_kind="set",
    )


_MAKERS = {
    "range_count": _make_range_count,
    "join_lookup": _make_join_lookup,
    "violation_flag": _make_violation_flag,
    "violation_rule": _make_violation_rule,
}


def generate_complex_tasks(
    context_lengths: list[int],
    tasks_per_type: int,
    seed: int = 42,
) -> list[Task]:
    rng = random.Random(seed)
    tasks: list[Task] = []
    counter = 0
    for length in context_lengths:
        for ttype, maker in _MAKERS.items():
            for _ in range(tasks_per_type):
                counter += 1
                tasks.append(maker(rng, length, counter))
    return tasks
