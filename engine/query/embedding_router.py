"""
EmbeddingTableRouter — semantic table routing via MetadataEmbeddingStore.

Drop-in alternative to TableRouter. Uses cosine similarity over embedded table
documents instead of table_affinity set intersection, making it more robust for
questions with unfamiliar phrasing or when table_affinity lists are incomplete.

The affinity-based TableRouter remains the default; this router is opted-in via
router_mode="embedding" in build_engine().
"""
from __future__ import annotations

from engine.metadata.embedding_store import MetadataEmbeddingStore
from engine.metadata.registry import MetadataRegistry
from engine.query.router import TableGroup
from models.intent import ParsedIntent


class EmbeddingTableRouter:
    """
    Resolves tables by semantic similarity between a query derived from the
    ParsedIntent and the embedded table metadata documents.

    The query is constructed from intent dimensions + metrics + formula_metrics.
    Channel filtering (from business rules) is applied after similarity ranking.
    """

    def __init__(
        self,
        store: MetadataEmbeddingStore,
        registry: MetadataRegistry,
        top_k: int = 3,
    ):
        self._store = store
        self._registry = registry
        self._top_k = top_k

    def resolve(self, intent: ParsedIntent) -> list[TableGroup]:
        query = self._build_query(intent)
        candidates = self._store.find_tables(query, top_k=self._top_k)

        if not candidates:
            return []

        # Filter by channels specified by business rules; fall back to all if
        # the filter would remove every candidate.
        if intent.channels:
            filtered = [
                t for t in candidates
                if (tbl := self._registry.get_table(t)) and tbl.channel in intent.channels
            ]
            candidates = filtered or candidates

        return [
            TableGroup(
                table=t,
                channel=self._registry.get_table(t).channel
                if self._registry.get_table(t) else "unknown",
                label=t,
            )
            for t in candidates
        ]

    def can_resolve(self, intent: ParsedIntent) -> bool:
        return len(self.resolve(intent)) > 0

    # ── Private ───────────────────────────────────────────────────────────────

    def _build_query(self, intent: ParsedIntent) -> str:
        terms = intent.dimensions + intent.metrics + intent.formula_metrics
        return " ".join(terms) if terms else "general data query"
