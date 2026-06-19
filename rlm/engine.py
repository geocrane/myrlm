"""Корневой цикл RLM.

Связывает воедино клиент, REPL и парсер: модель пишет код -> исполняем ->
возвращаем усечённый вывод -> повторяем, пока не получим FINAL/FINAL_VAR либо не
упрёмся в лимит итераций/времени. Внутри REPL модели доступна функция llm() для
рекурсивного вызова над фрагментом (depth=1 по умолчанию).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .config import load_config
from .llm_client import LLMClient
from .monitor import RunMonitor
from .parser import parse_turn
from .prompts import build_initial_messages, build_recursive_messages
from .repl import ReplSession


@dataclass
class RLMResult:
    """Ответ движка + телеметрия для сравнения подходов."""

    answer: str
    iterations: int
    stopped_reason: str            # "final" | "max_iterations" | "timeout"
    elapsed: float
    usage: dict[str, Any] = field(default_factory=dict)
    transcript: list[dict[str, str]] = field(default_factory=list)


class RLMEngine:
    def __init__(self, client: LLMClient | None = None, config: dict[str, Any] | None = None):
        self.config = config or load_config()
        self.client = client or LLMClient(self.config)
        self.rcfg = self.config["rlm"]

    def run(self, question: str, context: str, *, depth: int = 0,
            monitor: RunMonitor | None = None) -> RLMResult:
        """Запустить корневой RLM-цикл над контекстом.

        monitor (опционально) получает события: стриминг токенов, код, вывод REPL,
        итерации и финал — для живого отображения в Jupyter/консоли.
        """
        started = time.time()
        max_iters = self.rcfg["max_iterations"]
        timeout = self.rcfg["wall_clock_timeout"]
        on_token = monitor.token if monitor is not None else None

        repl = ReplSession(
            context,
            output_limit=self.rcfg["repl_output_limit"],
            recursive_llm=self._make_recursive_llm(depth, on_token),
        )
        messages = build_initial_messages(question)

        stopped = "max_iterations"
        answer = ""
        used_iters = 0

        for i in range(1, max_iters + 1):
            used_iters = i
            if time.time() - started > timeout:
                stopped = "timeout"
                break
            if monitor is not None:
                monitor.step("iteration", i)

            reply = self.client.chat(
                messages, role="root" if depth == 0 else "recursive", on_token=on_token,
            )
            messages.append({"role": "assistant", "content": reply})
            parsed = parse_turn(reply)

            if parsed.is_final:
                if parsed.final_var is not None:
                    val = repl.get_var(parsed.final_var)
                    answer = "" if val is None else str(val)
                else:
                    answer = parsed.final_answer or ""
                stopped = "final"
                if monitor is not None:
                    monitor.step("final", answer)
                break

            if parsed.code:
                if monitor is not None:
                    monitor.step("code", parsed.code)
                output = repl.execute(parsed.code)
            else:
                # Модель не дала ни кода, ни финала — подсказываем формат.
                output = (
                    "[нет код-блока и нет FINAL]. Выполни код в ```python ...``` "
                    "или заверши через FINAL(...) / FINAL_VAR(...)."
                )
            if monitor is not None:
                monitor.step("repl", output)
            messages.append({"role": "user", "content": f"[REPL]\n{output}"})

        # Если вышли по лимиту/таймауту — попросим модель дать ответ из накопленного.
        if stopped != "final":
            answer = self._force_final(messages)

        return RLMResult(
            answer=answer,
            iterations=used_iters,
            stopped_reason=stopped,
            elapsed=round(time.time() - started, 2),
            usage=self.client.usage.snapshot(),
            transcript=messages,
        )

    def _force_final(self, messages: list[dict[str, str]]) -> str:
        """Принудительно вытянуть лучший возможный ответ при исчерпании лимита."""
        nudge = list(messages) + [
            {
                "role": "user",
                "content": (
                    "Лимит шагов исчерпан. На основе уже собранной информации дай "
                    "лучший ответ ОДНОЙ строкой после FINAL(...). Код больше не пиши."
                ),
            }
        ]
        reply = self.client.chat(nudge, role="root")
        parsed = parse_turn(reply)
        if parsed.final_answer is not None:
            return parsed.final_answer
        # Фолбэк: вернуть текст ответа как есть.
        return reply.strip()

    def _make_recursive_llm(self, parent_depth: int, on_token=None):
        """Фабрика функции llm(question, text), доступной внутри REPL.

        На глубине < max_depth разрешён вызов вложенной модели; глубже — запрет,
        чтобы избежать неконтролируемой рекурсии (как в статье, по умолчанию depth=1).
        Рекурсивные вызовы тоже стримятся в монитор (on_token).
        """
        max_depth = self.rcfg["max_depth"]
        chunk_limit = self.rcfg["recursive_chunk_limit"]
        client = self.client

        def llm(question: str, text: str = "") -> str:
            if parent_depth + 1 > max_depth:
                return "[recursion-limit] глубже рекурсия отключена (max_depth)."
            text = str(text)
            if len(text) > chunk_limit:
                # Защита от передачи всего контекста целиком в один вызов.
                text = text[:chunk_limit]
            messages = build_recursive_messages(question, text)
            return client.chat(messages, role="recursive", on_token=on_token)

        return llm
