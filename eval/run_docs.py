"""Прогон методов (RLM / RAG / naive) на своих документах из командной строки.

Без Jupyter/ipywidgets. Указываешь папку базы знаний и вопрос — скрипт грузит
все файлы папки в один контекст, прогоняет выбранные методы и пишет результат в
results_real.{json,xlsx} (ответы методов рядом, эталон не нужен).

Примеры:
    # один вопрос к одной папке
    python -m eval.run_docs --ask ~/Документы/проект "Какой бюджет проекта?"

    # несколько пар «папка + вопрос», только два метода
    python -m eval.run_docs \
        --ask ./kb_договоры "Какие договоры превышают лимит?" \
        --ask ./kb_отчёты   "Сколько всего сотрудников?" \
        --methods rlm rag

    # свой конфиг и путь результата
    python -m eval.run_docs --ask ./docs "Вопрос" --config config.vllm.yaml --out my_results.json
"""

from __future__ import annotations

import argparse
from typing import Any

from rlm.config import load_config
from rlm.monitor import RunMonitor

from eval.doc_loader import build_real_tasks
from eval.run_experiment import run_suite, ALL_METHODS


class DocsConsoleMonitor(RunMonitor):
    """Консольный вывод для режима реальных документов (без accuracy/эталона)."""

    def suite_start(self, total: int, meta: dict[str, Any]) -> None:
        print(f"Задач: {meta['tasks']} | методы: {meta['methods']} | прогонов: {total}\n")

    def run_start(self, idx: int, total: int, task: Any, method: str) -> None:
        print(f"[{idx}/{total}] {method.upper():5} · {task.id} "
              f"({task.n_files} файлов, {task.char_len} симв.) · {task.question[:70]}", flush=True)

    def run_end(self, record: dict[str, Any]) -> None:
        ans = str(record["answer"]).replace("\n", " ")
        tok = record.get("usage", {}).get("total_tokens", 0)
        print(f"      → {ans[:200]}")
        print(f"        ({record.get('elapsed')}с · {tok} ток.)\n")

    def suite_end(self, records: list[dict[str, Any]], paths: dict[str, str]) -> None:
        print(f"Готово. JSON: {paths['json']}")
        if paths.get("xlsx"):
            print(f"Excel:       {paths['xlsx']}  (листы «Сравнение» и «Ресурсы»)")

    def error(self, exc: Exception) -> None:
        print(f"ОШИБКА ПРОГОНА: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Прогон RLM/RAG/naive на своих документах (без Jupyter)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--ask", nargs=2, metavar=("ПАПКА", "ВОПРОС"), action="append", required=True,
        help="папка базы знаний и вопрос к ней; можно повторять для нескольких пар",
    )
    parser.add_argument("--methods", nargs="+", choices=ALL_METHODS, default=ALL_METHODS)
    parser.add_argument("--config", default=None, help="путь к config.yaml (по умолч. config.yaml/$RLM_CONFIG)")
    parser.add_argument("--out", default="results_real.json", help="куда писать JSON (xlsx — рядом)")
    parser.add_argument("--no-recursive", action="store_true", help="не заходить в подпапки")
    args = parser.parse_args()

    config = load_config(args.config)
    rows = [(folder, question) for folder, question in args.ask]
    tasks = build_real_tasks(rows, recursive=not args.no_recursive)
    if not tasks:
        print("Нет задач: укажи хотя бы одну пару --ask ПАПКА ВОПРОС.")
        return

    run_suite(config, args.methods, tasks=tasks,
              results_path=args.out, monitor=DocsConsoleMonitor())


if __name__ == "__main__":
    main()
