"""Jupyter-интерфейс для прогона стенда на ipywidgets.

Показывает: прогресс-бар, статус (текущая задача/метод/время/токены/точность) и
прокручиваемое окно генерации RLM в реальном времени. Поток генерации (включая
<think>, отрисованный тускло) копится в логе и не исчезает — можно скроллить и
читать прошлые ответы прямо во время теста.

Использование в ноутбуке:
    from eval.notebook_ui import launch
    ui = launch(dataset="simple", quick=True)   # дальше нажать «Запустить»
"""

from __future__ import annotations

import html as _html
import threading
from typing import Any

import ipywidgets as W
from IPython.display import HTML, display

from rlm.config import load_config
from rlm.monitor import RunMonitor
from eval.run_experiment import run_suite, ALL_METHODS

_LOG_HEIGHT = "440px"


def _dim_think(raw: str) -> str:
    """Экранировать текст и отрисовать блоки <think> тускло-серым курсивом."""
    open_n = raw.count("<think>")
    close_n = raw.count("</think>")
    esc = _html.escape(raw)
    esc = esc.replace("&lt;think&gt;", "<span style='color:#9aa0a6;font-style:italic'>")
    esc = esc.replace("&lt;/think&gt;", "</span>")
    if open_n > close_n:  # незакрытый <think> в текущем стриме — закроем сами
        esc += "</span>" * (open_n - close_n)
    return esc


def _block(html_inner: str, *, bg: str = "transparent", border: str = "#e0e0e0") -> str:
    return (
        f"<div style='border-left:3px solid {border};background:{bg};"
        f"padding:4px 8px;margin:3px 0;white-space:pre-wrap;"
        f"font-family:monospace;font-size:12px;'>{html_inner}</div>"
    )


class NotebookMonitor(RunMonitor):
    """Монитор, который рисует прогресс и живой лог генерации в ipywidgets."""

    def __init__(self):
        self.progress = W.IntProgress(value=0, min=0, max=1, description="Прогон:",
                                      layout=W.Layout(width="60%"))
        self.lbl_status = W.HTML("Готов к запуску.")
        self.lbl_stats = W.HTML("")
        self.live = W.HTML("", layout=W.Layout(width="100%"))
        self.log_out = W.Output(layout=W.Layout(
            height=_LOG_HEIGHT, overflow_y="auto", border="1px solid #ccc", width="100%",
        ))
        self.panel = W.VBox([
            self.progress, self.lbl_status, self.lbl_stats,
            W.HTML("<b>Живая генерация RLM ↓</b> (история копится, можно прокручивать)"),
            self.live, self.log_out,
        ])
        self._stop = False
        self.reset()

    # ---- управление ----
    def reset(self) -> None:
        self._live_buf = ""
        self._n_done = 0
        self._n_correct = 0
        self._sum_secs = 0.0
        self._sum_tokens = 0
        self.progress.value = 0
        self.live.value = ""
        self.log_out.clear_output()

    def request_stop(self) -> None:
        self._stop = True
        self.lbl_status.value = "⏹ Остановка после текущего прогона…"

    def should_stop(self) -> bool:
        return self._stop

    # ---- события прогона ----
    def suite_start(self, total: int, meta: dict[str, Any]) -> None:
        self.progress.max = max(1, total)
        self.progress.value = 0
        self._append(_block(
            f"▶ старт: датасет <b>{meta['dataset']}</b>, задач {meta['tasks']}, "
            f"методы {meta['methods']}, длины {meta['lengths']}, прогонов {total}",
            bg="#eef5ff", border="#4a90d9",
        ))

    def run_start(self, idx: int, total: int, task: Any, method: str) -> None:
        self._flush_live()
        self.lbl_status.value = (
            f"[{idx}/{total}] <b>{method.upper()}</b> · {task.type} · "
            f"{task.id} · {task.char_len} симв."
        )
        self._append(_block(
            f"▶ [{idx}/{total}] {task.id} · {task.type} · <b>{method.upper()}</b><br>"
            f"<span style='color:#666'>{_html.escape(task.question[:160])}</span>",
            bg="#f3f3f3", border="#999",
        ))

    def token(self, text: str) -> None:
        self._live_buf += text
        # Рисуем текущую генерацию (с тусклым <think>); префикс — «печатает…».
        self.live.value = (
            "<div style='border:1px dashed #4a90d9;background:#fbfdff;padding:6px 8px;"
            "white-space:pre-wrap;font-family:monospace;font-size:12px;max-height:220px;"
            f"overflow:auto'>✍️ {_dim_think(self._live_buf)}</div>"
        )

    def step(self, kind: str, content: Any) -> None:
        if kind == "iteration":
            self._flush_live()
            self._append(_block(f"— итерация {content} —", border="#bbb"))
        elif kind == "code":
            self._flush_live()  # стримленный текст хода уже зафиксирован в истории
        elif kind == "repl":
            self._append(_block(
                "REPL ▸\n" + _html.escape(str(content)[:2000]),
                bg="#f7f7ef", border="#caa",
            ))
        elif kind == "final":
            self._flush_live()
            self._append(_block(
                "✅ FINAL: " + _html.escape(str(content)[:400]),
                bg="#eafaef", border="#2e9e4f",
            ))

    def run_end(self, record: dict[str, Any]) -> None:
        self._flush_live()
        self._n_done += 1
        self._n_correct += 1 if record["correct"] else 0
        self._sum_secs += record.get("elapsed", 0) or 0
        self._sum_tokens += record.get("usage", {}).get("total_tokens", 0) or 0
        self.progress.value = self._n_done
        mark = "✓" if record["correct"] else "✗"
        extra = ""
        if record.get("iterations") is not None:
            extra = f" · итер {record['iterations']} · {record.get('stopped_reason')}"
        self._append(_block(
            f"{mark} <b>{record['method'].upper()}</b> → {_html.escape(str(record['answer'])[:160])}"
            f"<br><span style='color:#666'>эталон: {_html.escape(str(record['gold'])[:120])} · "
            f"{record.get('elapsed')}с · {record.get('usage', {}).get('total_tokens', 0)} ток{extra}</span>",
            bg="#eafaef" if record["correct"] else "#fdeeee",
            border="#2e9e4f" if record["correct"] else "#c0392b",
        ))
        acc = self._n_correct / self._n_done if self._n_done else 0
        self.lbl_stats.value = (
            f"Готово: <b>{self._n_done}/{self.progress.max}</b> · "
            f"точность пока: <b>{acc:.0%}</b> ({self._n_correct}/{self._n_done}) · "
            f"суммарно: {self._sum_secs:.0f} с · {self._sum_tokens} токенов"
        )

    def suite_end(self, records: list[dict[str, Any]], paths: dict[str, str]) -> None:
        self.live.value = ""
        msg = f"🏁 Готово. JSON: {paths.get('json')}"
        if paths.get("xlsx"):
            msg += f" · Excel: {paths['xlsx']}"
        self.lbl_status.value = msg
        self._append(_block(msg, bg="#eef5ff", border="#4a90d9"))

    def error(self, exc: Exception) -> None:
        self._append(_block(f"❌ ОШИБКА ПРОГОНА: {_html.escape(str(exc))}",
                            bg="#fdeeee", border="#c0392b"))

    # ---- внутреннее ----
    def _flush_live(self) -> None:
        if self._live_buf.strip():
            self._append(_block(_dim_think(self._live_buf)))
        self._live_buf = ""
        self.live.value = ""

    def _append(self, html_str: str) -> None:
        # append_display_data безопаснее context-manager'а при вызове из потока.
        self.log_out.append_display_data(HTML(html_str))


class ExperimentUI:
    """Полный интерфейс: селекторы конфига/датасета/методов + кнопки + монитор."""

    def __init__(self, *, config_path: str = "", dataset: str = "simple",
                 methods: list[str] | None = None, quick: bool = False):
        self.config_text = W.Text(value=config_path, description="config:",
                                  placeholder="пусто = config.yaml или $RLM_CONFIG",
                                  layout=W.Layout(width="60%"))
        self.dataset_dd = W.Dropdown(options=["simple", "complex"], value=dataset, description="датасет:")
        self.methods_sel = W.SelectMultiple(options=ALL_METHODS, value=tuple(methods or ALL_METHODS),
                                            description="методы:", rows=3)
        self.quick_cb = W.Checkbox(value=quick, description="quick (1 задача/тип, 1 длина)")
        self.btn_run = W.Button(description="Запустить", button_style="success", icon="play")
        self.btn_stop = W.Button(description="Остановить", button_style="warning", icon="stop")
        self.monitor = NotebookMonitor()
        self._thread: threading.Thread | None = None

        self.btn_run.on_click(self._on_run)
        self.btn_stop.on_click(lambda _: self.monitor.request_stop())

        controls = W.VBox([
            W.HBox([self.config_text]),
            W.HBox([self.dataset_dd, self.quick_cb]),
            W.HBox([self.methods_sel]),
            W.HBox([self.btn_run, self.btn_stop]),
        ])
        self.container = W.VBox([controls, self.monitor.panel])

    def show(self) -> None:
        display(self.container)

    def _on_run(self, _) -> None:
        if self._thread and self._thread.is_alive():
            return  # уже идёт
        config = load_config(self.config_text.value or None)
        methods = list(self.methods_sel.value) or None
        dataset = self.dataset_dd.value
        quick = self.quick_cb.value
        self.monitor.reset()
        self.monitor._stop = False

        def work():
            try:
                run_suite(config, methods, dataset, quick=quick, monitor=self.monitor)
            except Exception as e:  # noqa: BLE001
                self.monitor.error(e)

        self._thread = threading.Thread(target=work, daemon=True)
        self._thread.start()


def launch(*, config_path: str = "", dataset: str = "simple",
           methods: list[str] | None = None, quick: bool = False) -> ExperimentUI:
    """Создать и показать интерфейс. Возвращает ExperimentUI."""
    ui = ExperimentUI(config_path=config_path, dataset=dataset, methods=methods, quick=quick)
    ui.show()
    return ui
