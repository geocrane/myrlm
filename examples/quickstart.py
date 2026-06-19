"""Минимальный демо-запуск одного RLM-вопроса.

Требует запущенного LM Studio с загруженной моделью (см. config.yaml).

    python -m examples.quickstart
"""

from __future__ import annotations

from rlm.engine import RLMEngine
from eval.datasets import generate_tasks


def main() -> None:
    # Берём одну aggregation-задачу ~8k символов — на ней нагляднее видно работу REPL.
    task = next(t for t in generate_tasks([8000], tasks_per_type=1, seed=7) if t.type == "aggregation")

    print(f"ВОПРОС: {task.question}")
    print(f"Эталон: {task.answer} | длина контекста: {task.char_len} символов\n")

    engine = RLMEngine()
    result = engine.run(task.question, task.context)

    print("\n--- ХОД РАССУЖДЕНИЙ (transcript) ---")
    for m in result.transcript:
        if m["role"] == "system":
            continue
        prefix = {"assistant": "МОДЕЛЬ", "user": "СРЕДА"}.get(m["role"], m["role"])
        print(f"\n[{prefix}]\n{m['content'][:1200]}")

    print("\n" + "=" * 60)
    print(f"ОТВЕТ RLM: {result.answer}")
    print(f"Причина остановки: {result.stopped_reason} | итераций: {result.iterations} | время: {result.elapsed}с")
    print(f"Телеметрия: {result.usage}")
    print(f"Верно: {result.answer.strip() == task.answer}")


if __name__ == "__main__":
    main()
