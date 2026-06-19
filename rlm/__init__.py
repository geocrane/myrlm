"""Минимальная реализация Recursive Language Models (RLM) с нуля.

Подход (Zhang, Kraska, Khattab, MIT, arXiv:2512.24601): длинный контекст хранится
как переменная в Python-REPL; корневая модель программно исследует его и при
необходимости рекурсивно вызывает LLM над фрагментами.
"""

from .llm_client import LLMClient
from .engine import RLMEngine

__all__ = ["LLMClient", "RLMEngine"]
