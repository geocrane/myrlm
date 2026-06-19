"""Сравнительный эксперимент: RLM vs RAG vs naive на Qwen3-8B (LM Studio).

Гоняет три подхода по сетке синтетических задач, собирает accuracy / latency /
токены / число REPL-итераций и печатает сводную таблицу. Это и есть ответ на
вопрос «насколько 8B-модель справляется с RLM против привычного RAG».

Запуск:
    python -m eval.run_experiment              # полный прогон по config.yaml
    python -m eval.run_experiment --quick      # быстрый: 1 задача/тип, длины 8k
    python -m eval.run_experiment --methods rlm rag
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from typing import Any

from rlm.config import load_config
from rlm.engine import RLMEngine
from rlm.llm_client import LLMClient
from rlm.monitor import RunMonitor

from baselines.naive import run_naive
from baselines.rag import run_rag, _Embedder
from eval.datasets import Task, generate_tasks
from eval.judge import evaluate

ALL_METHODS = ["rlm", "rag", "naive"]


def _run_one(method: str, task: Task, config: dict[str, Any], embedder, judge_client,
             monitor: RunMonitor | None = None) -> dict[str, Any]:
    """Прогнать один метод на одной задаче. Свежий клиент => изолированная телеметрия."""
    client = LLMClient(config)
    iterations = None
    stopped = None

    if method == "rlm":
        engine = RLMEngine(client=client, config=config)
        res = engine.run(task.question, task.context, monitor=monitor)
        answer, elapsed = res.answer, res.elapsed
        iterations, stopped = res.iterations, res.stopped_reason
    elif method == "rag":
        res = run_rag(task.question, task.context, client=client, config=config, embedder=embedder)
        answer, elapsed = res.answer, res.elapsed
    elif method == "naive":
        res = run_naive(task.question, task.context, client=client, config=config)
        answer, elapsed = res.answer, res.elapsed
    else:
        raise ValueError(f"неизвестный метод: {method}")

    verdict = evaluate(
        task.question, task.answer, answer,
        answer_kind=task.answer_kind, judge_client=judge_client,
    )
    return {
        "task_id": task.id,
        "type": task.type,
        "char_len": task.char_len,
        "question": task.question,
        "method": method,
        "answer": answer,
        "gold": task.answer,
        "correct": verdict["correct"],
        "judge_method": verdict["method"],
        "elapsed": elapsed,
        "iterations": iterations,
        "stopped_reason": stopped,
        "usage": client.usage.snapshot(),
    }


def _summarize(records: list[dict[str, Any]]) -> None:
    """Печать сводки: accuracy/latency/токены по (метод, тип, длина)."""
    # Группировка по методу и типу.
    agg: dict[tuple, dict[str, float]] = defaultdict(lambda: {"n": 0, "correct": 0, "secs": 0.0, "tok": 0})
    for r in records:
        key = (r["method"], r["type"])
        a = agg[key]
        a["n"] += 1
        a["correct"] += 1 if r["correct"] else 0
        a["secs"] += r["elapsed"]
        a["tok"] += r["usage"]["total_tokens"]

    print("\n" + "=" * 78)
    print(f"{'МЕТОД':<8}{'ТИП':<14}{'ТОЧНОСТЬ':<12}{'СР.ВРЕМЯ,с':<12}{'СР.ТОКЕНЫ':<12}")
    print("-" * 78)
    for (method, ttype), a in sorted(agg.items()):
        acc = a["correct"] / a["n"]
        print(
            f"{method:<8}{ttype:<14}{acc:>6.0%} ({int(a['correct'])}/{int(a['n'])})  "
            f"{a['secs']/a['n']:>9.1f}   {int(a['tok']/a['n']):>10}"
        )

    # Итог по методам.
    by_method: dict[str, dict[str, float]] = defaultdict(lambda: {"n": 0, "correct": 0})
    for r in records:
        by_method[r["method"]]["n"] += 1
        by_method[r["method"]]["correct"] += 1 if r["correct"] else 0
    print("-" * 78)
    for method, a in sorted(by_method.items()):
        print(f"ИТОГО {method:<6} точность: {a['correct']/a['n']:.0%} ({int(a['correct'])}/{int(a['n'])})")

    # Диагностика RLM: как часто доходил до FINAL, а не упирался в лимит.
    rlm_recs = [r for r in records if r["method"] == "rlm"]
    if rlm_recs:
        reasons = defaultdict(int)
        for r in rlm_recs:
            reasons[r["stopped_reason"]] += 1
        avg_iters = sum(r["iterations"] for r in rlm_recs) / len(rlm_recs)
        print("-" * 78)
        print(f"RLM: ср. итераций {avg_iters:.1f}; причины остановки: {dict(reasons)}")
    print("=" * 78)


def _empty_record(task: Task, method: str, msg: str) -> dict[str, Any]:
    return {
        "task_id": task.id, "type": task.type, "char_len": task.char_len,
        "question": task.question, "method": method, "answer": msg, "gold": task.answer,
        "correct": False, "judge_method": "error", "elapsed": 0.0,
        "iterations": None, "stopped_reason": "exception", "usage": {"total_tokens": 0},
    }


def run_suite(
    config: dict[str, Any],
    methods: list[str] | None = None,
    dataset: str = "simple",
    *,
    quick: bool = False,
    monitor: RunMonitor | None = None,
) -> dict[str, Any]:
    """Программный прогон стенда. Используется и CLI (main), и ноутбуком.

    monitor получает события (suite_start/run_start/token/step/run_end/suite_end) и
    может попросить остановку (should_stop). Возвращает словарь с записями и таймингом.
    """
    methods = methods or ALL_METHODS
    monitor = monitor or RunMonitor()

    if dataset == "complex":
        from eval.datasets_complex import generate_complex_tasks as gen_tasks
        ecfg = config["eval_complex"]
    else:
        gen_tasks = generate_tasks
        ecfg = config["eval"]

    if quick:
        lengths = [ecfg["context_lengths"][0]]
        per_type = 1
    else:
        lengths = ecfg["context_lengths"]
        per_type = ecfg["tasks_per_type"]

    tasks = gen_tasks(lengths, per_type, seed=ecfg["seed"])
    total = len(tasks) * len(methods)
    monitor.suite_start(total, {
        "dataset": dataset, "lengths": lengths, "methods": methods, "tasks": len(tasks),
    })

    # Общие ресурсы между задачами: эмбеддер (тяжёлый) и судья.
    embedder = _Embedder(config["rag"], config["llm"]) if "rag" in methods else None
    judge_client = LLMClient(config)

    records: list[dict[str, Any]] = []
    started = time.time()
    idx = 0
    stopped_early = False
    for task in tasks:
        for method in methods:
            if monitor.should_stop():
                stopped_early = True
                break
            idx += 1
            monitor.run_start(idx, total, task, method)
            try:
                rec = _run_one(method, task, config, embedder, judge_client, monitor)
            except Exception as e:  # один сбой не должен ронять весь прогон
                rec = _empty_record(task, method, f"[EXCEPTION] {e}")
            records.append(rec)
            monitor.run_end(rec)
        if stopped_early:
            break

    out = {
        "config": {"dataset": dataset, "lengths": lengths, "per_type": per_type, "methods": methods},
        "elapsed_total": round(time.time() - started, 1),
        "stopped_early": stopped_early,
        "records": records,
    }
    json_path = ecfg["results_path"]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # Excel-отчёт (не роняем прогон, если openpyxl не установлен).
    xlsx_path = json_path.rsplit(".", 1)[0] + ".xlsx"
    try:
        from eval.report import write_report
        write_report(json_path, xlsx_path)
    except Exception as e:
        xlsx_path = ""
        print(f"[предупреждение] Excel-отчёт не создан: {e}")

    monitor.suite_end(records, {"json": json_path, "xlsx": xlsx_path})
    return out


class ConsoleMonitor(RunMonitor):
    """Консольный монитор: печатает прогресс как раньше + сводку в конце."""

    def suite_start(self, total, meta):
        print(f"Датасет: {meta['dataset']} | задач: {meta['tasks']} | "
              f"методы: {meta['methods']} | длины: {meta['lengths']} | прогонов: {total}")

    def run_start(self, idx, total, task, method):
        print(f"[{idx}/{total}] {task.id} ({task.char_len} симв.) -> {method} ...", flush=True)

    def run_end(self, record):
        mark = "✓" if record["correct"] else "✗"
        print(f"    {mark} ответ: {str(record['answer'])[:80]!r}")

    def suite_end(self, records, paths):
        print(f"\nСырые результаты: {paths['json']}")
        if paths.get("xlsx"):
            print(f"Excel-отчёт: {paths['xlsx']}")
        _summarize(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="RLM vs RAG vs naive (LM Studio / vLLM)")
    parser.add_argument("--methods", nargs="+", choices=ALL_METHODS, default=ALL_METHODS)
    parser.add_argument("--quick", action="store_true", help="быстрый прогон: 1 задача/тип, длина 8k")
    parser.add_argument("--dataset", choices=["simple", "complex"], default="simple",
                        help="simple — синтетический текст; complex — договоры (фильтр/join/нарушения)")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    run_suite(config, args.methods, args.dataset, quick=args.quick, monitor=ConsoleMonitor())


if __name__ == "__main__":
    main()
