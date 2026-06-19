"""Скачать модель эмбеддинга в локальную папку инструмента (./models/).

После скачивания укажи путь в конфиге:
    rag:
      embedding_backend: sentence_transformers
      embedding_model: ./models/multilingual-e5-large-instruct

Использование:
    python -m scripts.download_embedder
    python -m scripts.download_embedder --model intfloat/multilingual-e5-large-instruct
    python -m scripts.download_embedder --model BAAI/bge-m3 --out ./models/bge-m3
"""

from __future__ import annotations

import argparse
import os

DEFAULT_MODEL = "intfloat/multilingual-e5-large-instruct"


def main() -> None:
    parser = argparse.ArgumentParser(description="Скачать sentence-transformers модель в ./models/")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="имя модели на HuggingFace")
    parser.add_argument("--out", default=None, help="каталог назначения (по умолчанию ./models/<имя>)")
    args = parser.parse_args()

    out = args.out or os.path.join("models", args.model.split("/")[-1])
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    print(f"Скачиваю {args.model} -> {out} (нужен интернет; на GPU-ПК будет использован torch)…")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.model)
    model.save(out)
    print(f"Готово. Укажи в конфиге: rag.embedding_model: {out}")


if __name__ == "__main__":
    main()
