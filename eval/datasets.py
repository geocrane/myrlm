"""Генератор синтетических long-context задач + загрузчик.

Создаёт длинный «стог» distractor-текста, в который вшиты факты-«иголки». Три типа
задач, перекликающиеся со статьёй RLM:
  - single_hop  — найти один факт в длинном тексте (needle-in-haystack);
  - multi_hop   — объединить 2+ факта, разнесённых по тексту;
  - aggregation — посчитать/перечислить по многим записям (аналог OOLONG, где
                  RLM особенно выигрывает над «всё в окно»).

Задачи детерминированы по seed, что важно для честного сравнения подходов.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, asdict
from typing import Any

# Словарь вымышленного мира — чтобы факты нельзя было «угадать» из претрейна.
_CITIES = ["Велдрис", "Кантара", "Морибонд", "Эльдория", "Сунгрейв", "Тарн", "Озмир", "Калленхейм"]
_PEOPLE = ["Орен", "Сильва", "Тадеуш", "Брина", "Касиус", "Лиора", "Вестон", "Мирабель", "Дрейк", "Нола"]
_DEPTS = ["Логистика", "Алхимия", "Картография", "Дозор", "Архив", "Кузня"]
_ENTITIES = ["Гильдия", "Башня", "Караван", "Обсерватория", "Порт", "Цитадель"]
_PROJECTS = ["Аврора", "Гелиос", "Обол", "Зенит", "Кобальт", "Мираж"]

# Нейтральные предложения-наполнители (distractors).
_FILLERS = [
    "Торговцы солью пересекали степь до первых заморозков.",
    "На рассвете колокола звонили трижды, как велит старый устав.",
    "Реки в этих краях мелеют к середине лета.",
    "Караванщики предпочитали северный тракт из-за разбойников на южном.",
    "Старые карты часто путали два одноимённых перевала.",
    "Ремесленники жаловались на нехватку медной проволоки.",
    "Погода менялась стремительно, и навигаторы сверялись со звёздами.",
    "Ярмарка длилась девять дней и собирала гостей из дальних земель.",
    "Хроники упоминают засуху, но без точных дат.",
    "Городской совет дважды переносил заседание из-за метели.",
]


@dataclass
class Task:
    id: str
    type: str
    question: str
    answer: str
    context: str
    char_len: int
    answer_kind: str  # "exact" | "judge" | "none" (реальные документы без эталона)
    source: str = ""   # папка базы знаний (для режима реальных документов)
    n_files: int = 0   # сколько файлов прочитано из папки


def _pad_to_length(rng: random.Random, pieces: list[str], target_chars: int) -> list[str]:
    """Доливаем distractor-предложения, пока не достигнем целевой длины."""
    out = list(pieces)
    while sum(len(p) for p in out) + len(out) < target_chars:
        out.append(rng.choice(_FILLERS))
    rng.shuffle(out)
    return out


def _assemble(rng: random.Random, needles: list[str], target_chars: int) -> str:
    """Перемешать иголки с наполнителем и собрать единый текст."""
    pieces = _pad_to_length(rng, needles, target_chars)
    # Гарантируем, что иголки разбросаны (после shuffle уже так), нумеруем строки —
    # это помогает aggregation-стратегиям и делает текст реалистичнее.
    return "\n".join(f"[{i:04d}] {s}" for i, s in enumerate(pieces))


def _make_single_hop(rng: random.Random, target_chars: int, idx: int) -> Task:
    entity = rng.choice(_ENTITIES) + "-" + str(rng.randint(100, 999))
    code = "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(6))
    needle = f"Секретный код объекта «{entity}» — {code}."
    ctx = _assemble(rng, [needle], target_chars)
    return Task(
        id=f"single_{idx}",
        type="single_hop",
        question=f"Назови секретный код объекта «{entity}».",
        answer=code,
        context=ctx,
        char_len=len(ctx),
        answer_kind="exact",
    )


def _make_multi_hop(rng: random.Random, target_chars: int, idx: int) -> Task:
    project = rng.choice(_PROJECTS) + "-" + str(rng.randint(10, 99))
    person = rng.choice(_PEOPLE)
    city = rng.choice(_CITIES)
    needle1 = f"Проектом «{project}» руководит {person}."
    needle2 = f"{person} постоянно работает в городе {city}."
    ctx = _assemble(rng, [needle1, needle2], target_chars)
    return Task(
        id=f"multi_{idx}",
        type="multi_hop",
        question=f"В каком городе ведётся руководство проектом «{project}»?",
        answer=city,
        context=ctx,
        char_len=len(ctx),
        answer_kind="exact",
    )


def _make_aggregation(rng: random.Random, target_chars: int, idx: int) -> Task:
    target_dept = rng.choice(_DEPTS)
    # Сколько сотрудников будет в целевом отделе.
    n_target = rng.randint(3, 7)
    needles = []
    used_names = set()

    def fresh_name() -> str:
        while True:
            name = rng.choice(_PEOPLE) + "-" + str(rng.randint(1000, 9999))
            if name not in used_names:
                used_names.add(name)
                return name

    for _ in range(n_target):
        needles.append(f"Сотрудник {fresh_name()} числится в отделе {target_dept}.")
    # Добавим записи по другим отделам как «шум по теме».
    for _ in range(rng.randint(8, 16)):
        other = rng.choice([d for d in _DEPTS if d != target_dept])
        needles.append(f"Сотрудник {fresh_name()} числится в отделе {other}.")

    ctx = _assemble(rng, needles, target_chars)
    return Task(
        id=f"agg_{idx}",
        type="aggregation",
        question=f"Сколько сотрудников числится в отделе {target_dept}? Ответь одним числом.",
        answer=str(n_target),
        context=ctx,
        char_len=len(ctx),
        answer_kind="exact",
    )


_MAKERS = {
    "single_hop": _make_single_hop,
    "multi_hop": _make_multi_hop,
    "aggregation": _make_aggregation,
}


def generate_tasks(
    context_lengths: list[int],
    tasks_per_type: int,
    seed: int = 42,
) -> list[Task]:
    """Сгенерировать набор задач по сетке длин и типов."""
    rng = random.Random(seed)
    tasks: list[Task] = []
    counter = 0
    for length in context_lengths:
        for ttype, maker in _MAKERS.items():
            for _ in range(tasks_per_type):
                counter += 1
                task = maker(rng, length, counter)
                tasks.append(task)
    return tasks


def save_tasks(tasks: list[Task], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(t) for t in tasks], f, ensure_ascii=False, indent=2)


def load_tasks(path: str) -> list[Task]:
    with open(path, "r", encoding="utf-8") as f:
        return [Task(**d) for d in json.load(f)]
