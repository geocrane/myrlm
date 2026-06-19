"""Загрузка config.yaml. Используется движком, бейзлайнами и стендом."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

import yaml

# Корень проекта = родитель пакета rlm/
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "config.yaml")


@lru_cache(maxsize=8)
def load_config(path: str | None = None) -> dict[str, Any]:
    """Прочитать YAML-конфиг. Кэшируется по пути.

    Приоритет пути: явный аргумент > переменная окружения RLM_CONFIG > config.yaml.
    Это позволяет на другом ПК переключаться (напр. на config.vllm.yaml) без правки кода.
    """
    path = path or os.environ.get("RLM_CONFIG") or DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def project_root() -> str:
    return _PROJECT_ROOT
