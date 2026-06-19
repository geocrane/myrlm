"""Персистентная Python-песочница для RLM.

Здесь живёт длинный контекст (переменная `context`) и состояние между ходами
корневой модели. Модель пишет код, мы исполняем его в общем namespace, ловим
stdout/ошибки и усекаем вывод — это и есть защита от «context rot».

ВНИМАНИЕ по безопасности: исполняется произвольный код через exec(). Это
осознанный выбор для локального исследовательского прототипа. Не подключайте
сюда недоверенный контекст/модель на чужой машине без изоляции (subprocess,
контейнер, seccomp и т.п.).
"""

from __future__ import annotations

import contextlib
import io
import json
import re
import traceback
from typing import Any, Callable


class ReplSession:
    """Долгоживущее пространство имён + исполнение код-блоков."""

    def __init__(
        self,
        context: str,
        *,
        output_limit: int = 3000,
        recursive_llm: Callable[..., str] | None = None,
    ):
        self.output_limit = output_limit
        # Базовые модули и данные, доступные модели без import.
        self.namespace: dict[str, Any] = {
            "context": context,
            "re": re,
            "json": json,
            "__builtins__": __builtins__,
        }
        # Рекурсивный вызов LLM над фрагментом (инъекция из движка).
        if recursive_llm is not None:
            self.namespace["llm"] = recursive_llm

    def execute(self, code: str) -> str:
        """Выполнить код, вернуть усечённый stdout (или текст ошибки)."""
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                # Единый namespace и для globals, и для locals — состояние
                # (переменные, функции) сохраняется между ходами.
                exec(code, self.namespace, self.namespace)
        except Exception:
            tb = traceback.format_exc()
            return self._truncate(f"{buf.getvalue()}\n[ОШИБКА]\n{tb}")

        out = buf.getvalue()
        if not out.strip():
            return "[код выполнен без вывода — используйте print(), чтобы что-то увидеть]"
        return self._truncate(out)

    def get_var(self, name: str) -> Any:
        """Достать значение переменной из namespace (для FINAL_VAR)."""
        return self.namespace.get(name)

    def _truncate(self, text: str) -> str:
        if len(text) <= self.output_limit:
            return text
        head = self.output_limit // 2
        tail = self.output_limit - head
        omitted = len(text) - self.output_limit
        return (
            text[:head]
            + f"\n\n... [усечено {omitted} символов] ...\n\n"
            + text[-tail:]
        )
