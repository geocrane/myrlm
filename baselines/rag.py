"""Бейзлайн RAG: чанкинг -> эмбеддинги -> top-k retrieval -> ответ.

Эмбеддинги по умолчанию офлайн через sentence-transformers; опционально — через
эмбеддинг-модель в LM Studio (OpenAI-совместимый /embeddings).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rlm.config import load_config
from rlm.llm_client import LLMClient

from .naive import BaselineResult

RAG_SYSTEM_PROMPT = (
    "Ответь на вопрос, опираясь ТОЛЬКО на приведённые фрагменты документа. "
    "Если ответа в них нет — так и скажи. Отвечай кратко."
)


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Нарезать текст на перекрывающиеся куски по символам."""
    if chunk_size <= 0:
        return [text]
    step = max(1, chunk_size - overlap)
    chunks = []
    for start in range(0, len(text), step):
        chunk = text[start : start + chunk_size]
        if chunk.strip():
            chunks.append(chunk)
        if start + chunk_size >= len(text):
            break
    return chunks


class _Embedder:
    """Ленивая обёртка над выбранным бэкендом эмбеддингов."""

    def __init__(self, rag_cfg: dict[str, Any], llm_cfg: dict[str, Any]):
        self.backend = rag_cfg.get("embedding_backend", "sentence_transformers")
        self.rag_cfg = rag_cfg
        self.llm_cfg = llm_cfg
        self._st_model = None
        self._oa_client = None
        # Имя активной модели — для определения нужных префиксов (семейство e5).
        self._model_name = (
            rag_cfg["lmstudio_embedding_model"] if self.backend == "lmstudio"
            else rag_cfg["embedding_model"]
        )

    def encode(self, texts: list[str], kind: str = "passage") -> np.ndarray:
        """kind: 'query' или 'passage' — влияет на префиксы для e5-моделей."""
        texts = [self._with_prefix(t, kind) for t in texts]
        if self.backend == "lmstudio":
            return self._encode_lmstudio(texts)
        return self._encode_st(texts)

    def _with_prefix(self, text: str, kind: str) -> str:
        """Навесить префикс по соглашению модели. Для не-e5 возвращает текст как есть."""
        name = self._model_name.lower()
        if "e5" not in name:
            return text
        if "instruct" in name:
            # e5-instruct: инструктивный префикс только к запросу, пассажи — без префикса.
            if kind == "query":
                task = "Найди фрагмент текста, отвечающий на вопрос"
                return f"Instruct: {task}\nQuery: {text}"
            return text
        # Классический e5: query:/passage:.
        return f"{kind}: {text}"

    def _encode_st(self, texts: list[str]) -> np.ndarray:
        if self._st_model is None:
            from sentence_transformers import SentenceTransformer

            self._st_model = SentenceTransformer(self.rag_cfg["embedding_model"])
        emb = self._st_model.encode(texts, normalize_embeddings=True)
        return np.asarray(emb, dtype=np.float32)

    def _encode_lmstudio(self, texts: list[str]) -> np.ndarray:
        if self._oa_client is None:
            from openai import OpenAI

            self._oa_client = OpenAI(
                base_url=self.llm_cfg["base_url"],
                api_key=self.llm_cfg.get("api_key", "lm-studio"),
            )
        resp = self._oa_client.embeddings.create(
            model=self.rag_cfg["lmstudio_embedding_model"], input=texts
        )
        vecs = np.asarray([d.embedding for d in resp.data], dtype=np.float32)
        # Нормируем для косинусной близости через скалярное произведение.
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.clip(norms, 1e-8, None)


def run_rag(
    question: str,
    context: str,
    *,
    client: LLMClient | None = None,
    config: dict[str, Any] | None = None,
    embedder: _Embedder | None = None,
) -> BaselineResult:
    """Классический retrieval-augmented ответ."""
    config = config or load_config()
    client = client or LLMClient(config)
    rag_cfg = config["rag"]

    started = time.time()
    chunks = chunk_text(context, rag_cfg["chunk_size"], rag_cfg["chunk_overlap"])

    embedder = embedder or _Embedder(rag_cfg, config["llm"])
    chunk_vecs = embedder.encode(chunks, kind="passage")
    q_vec = embedder.encode([question], kind="query")[0]

    # Косинусная близость (векторы уже нормированы) -> top-k.
    scores = chunk_vecs @ q_vec
    top_k = min(rag_cfg["top_k"], len(chunks))
    top_idx = np.argsort(-scores)[:top_k]
    # Сохраняем исходный порядок документа среди отобранных — так связнее.
    top_idx = sorted(top_idx.tolist())
    retrieved = "\n\n---\n\n".join(chunks[i] for i in top_idx)

    messages = [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {"role": "user", "content": f"ФРАГМЕНТЫ:\n{retrieved}\n\nВОПРОС: {question}"},
    ]
    answer = client.chat(messages, role="rag")

    return BaselineResult(
        answer=answer,
        elapsed=round(time.time() - started, 2),
        method="rag",
        usage=client.usage.snapshot(),
        note=f"чанков: {len(chunks)}, отобрано: {top_k}",
    )
