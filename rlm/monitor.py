"""Лёгкий интерфейс событий для наблюдения за прогоном (без тяжёлых зависимостей).

Движок и стенд вызывают методы монитора в ключевых точках; конкретные реализации
(консольный вывод, Jupyter-виджеты) наследуются от RunMonitor. По умолчанию все
методы — no-op, поэтому передавать монитор необязательно.
"""

from __future__ import annotations

from typing import Any


class RunMonitor:
    """Базовый монитор. Переопредели нужные методы в наследнике."""

    # --- жизненный цикл всего прогона ---
    def suite_start(self, total: int, meta: dict[str, Any]) -> None: ...
    def suite_end(self, records: list[dict[str, Any]], paths: dict[str, str]) -> None: ...

    # --- отдельный запуск (одна задача × один метод) ---
    def run_start(self, idx: int, total: int, task: Any, method: str) -> None: ...
    def run_end(self, record: dict[str, Any]) -> None: ...

    # --- внутренняя жизнь RLM-цикла ---
    def token(self, text: str) -> None:
        """Дельта стриминга (сырой текст, включая <think>)."""

    def step(self, kind: str, content: Any) -> None:
        """Событие шага. kind: iteration | code | repl | final | info."""

    # --- кооперативная остановка ---
    def should_stop(self) -> bool:
        return False
