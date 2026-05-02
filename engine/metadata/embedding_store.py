"""
MetadataEmbeddingStore — embeds table metadata documents for semantic table routing.

Each table in tables.json is converted into a text document describing its channel,
description, dimensions, and metrics (with aliases). Documents are embedded via a
LangChain Embeddings instance and cached to disk so restarts do not re-embed.

Usage:
    from adapters.embeddings import create_embeddings
    from engine.metadata.embedding_store import MetadataEmbeddingStore

    store = MetadataEmbeddingStore(config, create_embeddings(), cache_path=".cache/embeddings.json")
    store.build()
    tables = store.find_tables("revenue by region last month", top_k=3)
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

import numpy as np
from langchain_core.embeddings import Embeddings

from engine.metadata.loader import MetadataConfig


class MetadataEmbeddingStore:
    """
    Builds and queries semantic embeddings of table metadata.

    Each table becomes one document:
        Table: orders | Channel: ecommerce | Description: ... |
        Dimensions: region(area, zone), product_name(item, sku) |
        Metrics: revenue(sales, turnover), orders_count(order volume)

    Cosine similarity is used to rank tables against a free-text query.
    """

    def __init__(
        self,
        config: MetadataConfig,
        embeddings: Embeddings,
        cache_path: str | None = None,
    ):
        self._config = config
        self._embeddings = embeddings
        self._cache_path = Path(cache_path) if cache_path else None
        self._docs: dict[str, str] = {}
        self._vectors: dict[str, list[float]] = {}
        self._built = False

    # ── Public API ────────────────────────────────────────────────────────────

    def build(self) -> None:
        """Build embeddings for all tables. Loads from cache if config is unchanged."""
        self._docs = self._build_documents()

        if self._cache_path and self._cache_path.exists():
            try:
                cached = json.loads(self._cache_path.read_text())
                if cached.get("hash") == self._config_hash():
                    self._vectors = cached["vectors"]
                    self._built = True
                    return
            except (json.JSONDecodeError, KeyError):
                pass  # corrupt cache — rebuild

        self._embed_and_cache()

    def find_tables(self, query: str, top_k: int = 3) -> list[str]:
        """
        Return up to top_k table names ranked by cosine similarity to query.
        Calls build() automatically if not yet built.
        """
        if not self._built:
            self.build()

        if not self._vectors:
            return []

        query_vec = np.array(self._embeddings.embed_query(query), dtype=float)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return list(self._vectors.keys())[:top_k]

        scores: dict[str, float] = {}
        for table_name, vec in self._vectors.items():
            tv = np.array(vec, dtype=float)
            tv_norm = np.linalg.norm(tv)
            if tv_norm == 0:
                scores[table_name] = 0.0
            else:
                scores[table_name] = float(np.dot(query_vec, tv) / (query_norm * tv_norm))

        return sorted(scores, key=lambda t: scores[t], reverse=True)[:top_k]

    def invalidate_cache(self) -> None:
        """Delete the on-disk cache, forcing a full rebuild on next build()."""
        if self._cache_path and self._cache_path.exists():
            self._cache_path.unlink()
        self._built = False
        self._vectors = {}

    # ── Private ───────────────────────────────────────────────────────────────

    def _embed_and_cache(self) -> None:
        table_names = list(self._docs.keys())
        doc_texts = [self._docs[t] for t in table_names]
        vectors = self._embeddings.embed_documents(doc_texts)
        self._vectors = {t: v for t, v in zip(table_names, vectors)}

        if self._cache_path:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(json.dumps({
                "hash": self._config_hash(),
                "vectors": self._vectors,
            }))

        self._built = True

    def _build_documents(self) -> dict[str, str]:
        dim_index = {d.name: d for d in self._config.dimensions}
        metric_index = {m.name: m for m in self._config.metrics}

        docs = {}
        for table in self._config.tables:
            dim_labels = []
            for d_name in table.dimensions:
                d = dim_index.get(d_name)
                aliases = ", ".join(d.aliases[:3]) if d and d.aliases else ""
                dim_labels.append(f"{d_name}({aliases})" if aliases else d_name)

            metric_labels = []
            for m_name in table.metrics:
                m = metric_index.get(m_name)
                aliases = ", ".join(m.aliases[:3]) if m and m.aliases else ""
                metric_labels.append(f"{m_name}({aliases})" if aliases else m_name)

            parts = [f"Table: {table.name}", f"Channel: {table.channel}"]
            if table.description:
                parts.append(f"Description: {table.description}")
            parts.append(f"Dimensions: {', '.join(dim_labels) or 'none'}")
            parts.append(f"Metrics: {', '.join(metric_labels) or 'none'}")

            docs[table.name] = " | ".join(parts)

        return docs

    def _config_hash(self) -> str:
        return hashlib.md5(
            json.dumps(self._docs, sort_keys=True).encode()
        ).hexdigest()
