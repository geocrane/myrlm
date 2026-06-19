"""Офлайн-тесты (без LLM): REPL, парсер, нарезка чанков, генератор задач, судья.

    python -m pytest tests/ -q       # если установлен pytest
    python tests/test_offline.py     # запуск без pytest
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rlm.repl import ReplSession
from rlm.parser import parse_turn
from rlm.llm_client import strip_think
from baselines.rag import chunk_text
from eval.datasets import generate_tasks
from eval.datasets_complex import generate_complex_tasks, _CATEGORY_LIMITS
from eval.judge import exact_match, set_match, all_match


def test_repl_persistence_and_context():
    repl = ReplSession("ABCDEFG", output_limit=1000)
    out = repl.execute("print(len(context)); x = 41")
    assert "7" in out
    out2 = repl.execute("print(x + 1)")          # состояние сохранилось между ходами
    assert "42" in out2


def test_repl_truncation():
    repl = ReplSession("", output_limit=100)
    out = repl.execute("print('z' * 5000)")
    assert len(out) <= 200
    assert "усечено" in out


def test_repl_error_capture():
    repl = ReplSession("", output_limit=1000)
    out = repl.execute("1/0")
    assert "ОШИБКА" in out and "ZeroDivisionError" in out


def test_repl_no_output_hint():
    repl = ReplSession("", output_limit=1000)
    out = repl.execute("y = 5")
    assert "без вывода" in out


def test_parser_code_block():
    p = parse_turn("Давай посмотрим\n```python\nprint(1)\n```")
    assert p.code == "print(1)"
    assert not p.is_final


def test_parser_final():
    p = parse_turn("Ответ найден.\nFINAL(1487)")
    assert p.final_answer == "1487"
    assert p.is_final


def test_parser_final_var():
    p = parse_turn("Готово\nFINAL_VAR(result)")
    assert p.final_var == "result"


def test_parser_final_in_code_ignored():
    # FINAL внутри кода (print) не должен считаться завершением.
    p = parse_turn("```python\nprint('FINAL(123)')\n```")
    assert not p.is_final
    assert p.code is not None


def test_strip_think():
    assert strip_think("<think>рассуждаю</think>ответ") == "ответ"
    assert strip_think("<think>обрыв без конца") == ""


def test_chunking():
    text = "a" * 1000
    chunks = chunk_text(text, chunk_size=300, overlap=50)
    assert all(len(c) <= 300 for c in chunks)
    assert len(chunks) > 1


def test_dataset_generation_deterministic():
    t1 = generate_tasks([5000], 1, seed=1)
    t2 = generate_tasks([5000], 1, seed=1)
    assert [t.answer for t in t1] == [t.answer for t in t2]
    # Каждая иголка реально присутствует в контексте.
    for t in t1:
        if t.type == "single_hop":
            assert t.answer in t.context
        if t.type == "aggregation":
            assert int(t.answer) >= 1


def test_exact_match():
    assert exact_match("Код объекта — ABC123.", "ABC123")
    assert exact_match("Ответ: 5", "5")
    assert not exact_match("Ответ: 50", "5")     # не путаем число-подстроку


def test_set_match():
    assert set_match("Нарушают: D-1042, D-9001", "D-9001, D-1042")  # порядок не важен
    assert set_match("Это договоры D1042 и D-9001.", "D-1042, D-9001")  # дефис не важен
    assert set_match("['1042', '9001']", "D-1042, D-9001")          # голые числа (fallback)
    assert not set_match("D-1042", "D-1042, D-9001")                # пропущен один
    assert not set_match("Нарушений нет", "D-1042")                 # пустой ответ


def test_all_match():
    assert all_match("Контрагент Вектор, ответственный Орен", "Вектор | Орен")
    assert not all_match("Контрагент Вектор", "Вектор | Орен")      # нет второй части


def test_complex_deterministic_and_gold():
    t1 = generate_complex_tasks([6000], 1, seed=5)
    t2 = generate_complex_tasks([6000], 1, seed=5)
    assert [t.answer for t in t1] == [t.answer for t in t2]
    types = {t.type for t in t1}
    assert types == {"range_count", "join_lookup", "violation_flag", "violation_rule"}
    # violation_rule: каждый номер из gold реально нарушает лимит в тексте карточек.
    vr = next(t for t in t1 if t.type == "violation_rule")
    import re
    golds = set(re.findall(r"D-(\d+)", vr.answer))
    assert golds  # эталон непустой
    # Разберём карточки и проверим, что золотые номера действительно превышают лимит.
    for block in vr.context.split("=== ДОГОВОР"):
        m_num = re.search(r"№D-(\d+)", block)
        m_cat = re.search(r"Категория: (\w+)", block)
        m_sum = re.search(r"Сумма: (\d+)", block)
        if not (m_num and m_cat and m_sum):
            continue
        num, cat, amount = m_num.group(1), m_cat.group(1), int(m_sum.group(1))
        if num in golds:
            assert amount > _CATEGORY_LIMITS[cat]


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  ok  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERR   {fn.__name__}: {e!r}")
    print(f"\n{len(fns) - failed}/{len(fns)} тестов прошло")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
