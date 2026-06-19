"""Разбор ответа корневой модели: код-блоки и команды завершения.

Протокол взаимодействия:
- модель пишет ОДИН блок ```python ...``` за ход — он исполняется в REPL;
- завершение: FINAL(<выражение>) либо FINAL_VAR(<имя_переменной_в_REPL>).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Код-блок: ```python ... ``` или просто ``` ... ```
_CODE_RE = re.compile(r"```(?:python|py)?\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE)

# FINAL(...) / FINAL_VAR(...) — берём содержимое скобок (нежадно, по последней закрывающей в строке).
_FINAL_RE = re.compile(r"\bFINAL\s*\((.*)\)", re.DOTALL)
_FINAL_VAR_RE = re.compile(r"\bFINAL_VAR\s*\(\s*([A-Za-z_]\w*)\s*\)")


@dataclass
class ParsedTurn:
    """Результат разбора одного ответа модели."""

    code: str | None = None          # python-код для исполнения
    final_answer: str | None = None  # готовый ответ из FINAL(...)
    final_var: str | None = None     # имя переменной из FINAL_VAR(...)

    @property
    def is_final(self) -> bool:
        return self.final_answer is not None or self.final_var is not None


def parse_turn(text: str) -> ParsedTurn:
    """Разобрать ответ модели.

    Приоритет завершения над кодом: если в ответе есть FINAL/FINAL_VAR — считаем
    ход финальным (модель иногда печатает и код, и финал; финал важнее).
    Исключение: если FINAL встречается ВНУТРИ код-блока, это часть кода (например
    print), а не команда завершения — такой случай игнорируем.
    """
    code_match = _CODE_RE.search(text)
    code = code_match.group(1).strip() if code_match else None

    # Ищем FINAL_VAR / FINAL вне код-блоков.
    outside = _strip_code_blocks(text)

    var_match = _FINAL_VAR_RE.search(outside)
    if var_match:
        return ParsedTurn(final_var=var_match.group(1))

    final_match = _FINAL_RE.search(outside)
    if final_match:
        answer = final_match.group(1).strip()
        # Снимаем обрамляющие кавычки, если модель обернула строку.
        answer = _unquote(answer)
        return ParsedTurn(final_answer=answer)

    return ParsedTurn(code=code)


def _strip_code_blocks(text: str) -> str:
    return _CODE_RE.sub("", text)


def _unquote(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in {'"', "'"}:
        return s[1:-1]
    return s
