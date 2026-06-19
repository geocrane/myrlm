"""Клиент к LM Studio (OpenAI-совместимый API).

Отвечает за единый вход к модели: формирование запроса, управление режимом
рассуждений Qwen3, вырезание <think>...</think> и учёт токенов/времени для
последующего сравнения подходов.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .config import load_config

# Для приблизительного учёта токенов (если tiktoken не установлен — символьный фолбэк).
try:
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover - зависит от окружения
    _ENC = None


def count_tokens(text: str) -> int:
    """Приблизительное число токенов. Для метрик, не для биллинга."""
    if not text:
        return 0
    if _ENC is not None:
        return len(_ENC.encode(text))
    # Грубый фолбэк: ~4 символа на токен.
    return max(1, len(text) // 4)


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think(text: str) -> str:
    """Убрать блоки рассуждений Qwen3. Обрабатывает и незакрытый <think>."""
    text = _THINK_RE.sub("", text)
    # Незакрытый блок (модель не успела закрыть тег) — отрезаем всё до конца.
    if "<think>" in text:
        text = text.split("<think>", 1)[0]
    return text.strip()


@dataclass
class Usage:
    """Накопительная телеметрия по всем вызовам клиента."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_seconds: float = 0.0
    # Разбивка по «ролям» вызова: root / recursive / rag / naive / judge.
    by_role: dict[str, int] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, role: str, p_tok: int, c_tok: int, secs: float) -> None:
        self.calls += 1
        self.prompt_tokens += p_tok
        self.completion_tokens += c_tok
        self.total_seconds += secs
        self.by_role[role] = self.by_role.get(role, 0) + 1

    def snapshot(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "total_seconds": round(self.total_seconds, 2),
            "by_role": dict(self.by_role),
        }


class LLMClient:
    """Тонкая обёртка над OpenAI SDK, нацеленная на LM Studio."""

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = (config or load_config())["llm"]
        self.cfg = cfg
        self.model = cfg["model"]
        # Ленивый импорт: пакет rlm можно импортировать без openai, пока клиент
        # реально не создаётся (нужно для офлайн-тестов REPL/парсера).
        from openai import OpenAI

        self._client = OpenAI(
            base_url=cfg["base_url"],
            api_key=cfg.get("api_key", "lm-studio"),
            timeout=cfg.get("request_timeout", 600),
        )
        self.usage = Usage()

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        role: str = "root",
        temperature: float | None = None,
        max_tokens: int | None = None,
        enable_thinking: bool | None = None,
        on_token: "Callable[[str], None] | None" = None,
    ) -> str:
        """Один запрос к модели. Возвращает текст без блоков <think>.

        role — метка для телеметрии (root/recursive/rag/naive/judge).
        on_token — если задан, ответ запрашивается стримингом и каждая дельта
        (сырой текст, включая <think>) отдаётся в колбэк для живого отображения.
        """
        think = self.cfg.get("enable_thinking", False) if enable_thinking is None else enable_thinking
        messages = self._apply_thinking_switch(messages, think)

        temp = self.cfg.get("temperature", 0.3) if temperature is None else temperature
        mtok = self.cfg.get("max_tokens", 2048) if max_tokens is None else max_tokens

        started = time.time()
        if on_token is None:
            raw, usage = self._complete(messages, temp, mtok)
        else:
            raw, usage = self._stream(messages, temp, mtok, on_token)
        elapsed = time.time() - started

        text = strip_think(raw)

        # Учёт токенов: предпочитаем данные API, иначе оцениваем сами.
        if usage is not None and getattr(usage, "prompt_tokens", None):
            p_tok, c_tok = usage.prompt_tokens, usage.completion_tokens
        else:
            p_tok = sum(count_tokens(m.get("content", "")) for m in messages)
            c_tok = count_tokens(raw)
        self.usage.add(role, p_tok, c_tok, elapsed)

        return text

    def _complete(self, messages, temp, mtok):
        """Обычный (нестриминговый) запрос."""
        resp = self._client.chat.completions.create(
            model=self.model, messages=messages, temperature=temp, max_tokens=mtok,
        )
        return (resp.choices[0].message.content or ""), getattr(resp, "usage", None)

    def _stream(self, messages, temp, mtok, on_token):
        """Стриминг: собираем ответ по дельтам, каждую отдаём в on_token."""
        stream = self._client.chat.completions.create(
            model=self.model, messages=messages, temperature=temp, max_tokens=mtok,
            stream=True, stream_options={"include_usage": True},
        )
        parts: list[str] = []
        usage = None
        for chunk in stream:
            if getattr(chunk, "usage", None):
                usage = chunk.usage
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            piece = getattr(delta, "content", None)
            if piece:
                parts.append(piece)
                try:
                    on_token(piece)
                except Exception:
                    pass  # сбой отрисовки не должен ронять прогон
        return "".join(parts), usage

    def _apply_thinking_switch(self, messages: list[dict[str, str]], think: bool) -> list[dict[str, str]]:
        """Переключение режима рассуждений.

        thinking_style=qwen — добавляем токен /think|/no_think к последнему
        пользовательскому сообщению (как у Qwen3). thinking_style=none — ничего не
        вставляем (для не-Qwen моделей на vLLM).
        """
        if self.cfg.get("thinking_style", "qwen") != "qwen":
            return messages
        switch = "/think" if think else "/no_think"
        out = [dict(m) for m in messages]
        for m in reversed(out):
            if m.get("role") == "user":
                m["content"] = f"{m['content']}\n\n{switch}"
                break
        return out
