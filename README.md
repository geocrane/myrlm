# RLM на локальной LLM — собственный движок + сравнение с RAG

Минимальная реализация **Recursive Language Models** (RLM) с нуля на Python,
работающая поверх **Qwen3-8B в LM Studio**, плюс стенд для сравнения RLM с RAG и
наивным «всё в контекст».

Подход RLM (Zhang, Kraska, Khattab, MIT, arXiv:2512.24601, дек. 2025): длинный
контекст **не кладётся в промпт**, а хранится как переменная `context` в
Python-REPL. Корневая модель видит только вопрос и программно исследует контекст
(заглянуть / grep / нарезать+обойти / суммаризировать), при необходимости
**рекурсивно вызывая LLM** над фрагментами (`depth=1`). Это снижает «context rot»
и позволяет работать с контекстом во много раз больше окна модели.

## Зачем
Qwen3-8B неплохо тянет RAG. Открытый вопрос — справится ли 8B-модель с агентным
RLM-циклом (надёжно писать код, управлять REPL, решать, когда рекурсировать).
Этот стенд позволяет измерить разницу на одинаковых задачах.

## Структура
```
rlm/         движок: llm_client, repl, parser, prompts, engine
baselines/   naive (всё в контекст) и rag (retrieval)
eval/        генератор задач, LLM-судья, прогон эксперимента
examples/    quickstart — один RLM-вопрос с печатью хода рассуждений
tests/       офлайн-тесты (без LLM)
```

## Установка
```bash
pip install -r requirements.txt
```

## Подготовка LM Studio
1. Загрузить модель **Qwen3-8B**.
2. Вкладка **Developer / Server → Start** (эндпоинт `http://localhost:1234/v1`).
3. При несовпадении имени модели — поправить `llm.model` в `config.yaml`.
4. Для RAG-эмбеддингов по умолчанию используется офлайн `sentence-transformers`
   (`config.yaml → rag.embedding_backend`); можно переключить на `lmstudio`,
   загрузив в LM Studio эмбеддинг-модель.

## Запуск
```bash
# Офлайн-тесты (LM Studio не нужен)
python tests/test_offline.py

# Демо одного RLM-вопроса с трассировкой
python -m examples.quickstart

# Сравнительный эксперимент
python -m eval.run_experiment --quick                 # быстрый прогон
python -m eval.run_experiment                          # полный, по config.yaml
python -m eval.run_experiment --methods rlm rag        # только выбранные методы
```
Результаты пишутся в `results.json`, сводка печатается в консоль (accuracy /
время / токены по типам задач + диагностика RLM: число итераций и причины
остановки).

Сложный датасет «договоры» (фильтр по сумме, join, нарушения):
```bash
python -m eval.run_experiment --dataset complex        # → results_complex.{json,xlsx}
```

## Запуск из Jupyter (живой стриминг)
Интерактивный прогон с прогресс-баром, статусом (тест/осталось/время/токены) и
**живым окном генерации RLM** (видно, какой код пишет модель и какие ошибки в REPL;
история копится и прокручивается, `<think>` показан тускло):
```bash
pip install -r requirements.txt        # нужен ipywidgets
jupyter lab                            # открыть notebooks/run_experiment.ipynb
```
В ноутбуке: выбрать конфиг/датасет/методы → «Запустить». Прогон идёт в фоне, кнопка
«Остановить» завершает после текущего прогона. Под капотом — `eval.notebook_ui.launch`
и программный API `eval.run_experiment.run_suite(...)`.

## Запуск на другом ПК (vLLM + локальный эмбеддер)
1. **Поднять модель в vLLM** (OpenAI-совместимый API):
   ```bash
   vllm serve Qwen/Qwen3-32B --host 0.0.0.0 --port 8000 --max-model-len 40000
   ```
2. **Скачать модель эмбеддинга** в папку инструмента:
   ```bash
   python -m scripts.download_embedder            # → ./models/multilingual-e5-large-instruct
   ```
3. **Указать конфиг** — готовый пример `config.vllm.yaml` (правь `base_url`, `model`,
   `naive.context_token_budget` под `--max-model-len`). Выбрать его можно тремя способами:
   ```bash
   export RLM_CONFIG=$(pwd)/config.vllm.yaml      # переменная окружения
   python -m eval.run_experiment --config config.vllm.yaml --dataset complex   # флаг
   ```
   …либо вписать путь в поле `config` в ноутбуке.
4. Для не-Qwen моделей выставь `llm.thinking_style: none` (не вставлять Qwen-токены).

> На ПК с локальным эмбеддером нужен `torch` (ставится вместе с `sentence-transformers`);
> на GPU-машине это как раз даёт быстрые эмбеддинги.

## Типы задач (eval/datasets.py)
- **single_hop** — найти один факт в длинном тексте (needle-in-haystack);
- **multi_hop** — объединить два разнесённых факта;
- **aggregation** — посчитать записи по критерию (аналог OOLONG; здесь RLM обычно
  выигрывает у «всё в окно»).

Контекст синтетический, из вымышленного мира (нельзя «угадать» из претрейна),
детерминирован по seed.

## Параметры (config.yaml)
- `rlm.max_iterations`, `wall_clock_timeout` — лимиты корневого цикла;
- `rlm.repl_output_limit` — усечение вывода REPL (защита от context rot);
- `rlm.max_depth` — глубина рекурсии (по умолчанию 1, как в статье);
- `rag.chunk_size / overlap / top_k` — параметры retrieval;
- `eval.context_lengths / tasks_per_type` — сетка эксперимента.

## ⚠️ Безопасность
Движок исполняет код, сгенерированный моделью, через `exec()` в локальном
процессе (`rlm/repl.py`). Это осознанный выбор для исследовательского прототипа на
доверенной машине. **Не запускайте с недоверенным контекстом/моделью без
изоляции** (subprocess с таймаутом, контейнер, seccomp и т.п.).

## Известные ограничения
- Рекурсивные вызовы синхронны (без параллелизма и кэширования префиксов).
- Нет жёстких гарантий по стоимости/времени одного вопроса.
- На 8B-модели возможны срывы формата и зацикливания — это часть предмета
  измерения, частично смягчается few-shot в `rlm/prompts.py` и принудительным
  финалом при исчерпании лимита.
