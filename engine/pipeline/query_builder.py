"""
Stage 2 — Query Builder
Converts an enriched ParsedIntent into SQL queries.
Pure Python, metadata-driven — no LLM involved.
"""
from __future__ import annotations
import logging
from models.intent import ParsedIntent
from models.query import BuiltQuery
from engine.query.builder import QueryBuilder
from engine.rules.engine import BusinessRulesEngine

logger = logging.getLogger(__name__)


class QueryBuilderStage:
    """
    Stage 2: ParsedIntent (enriched) → list[BuiltQuery]

    Flow:
        enriched intent
          → BusinessRulesEngine.enrich() applies column/formula/sort rules
          → QueryBuilder.build() routes to tables and assembles SQL
          → Returns list of BuiltQuery (one per resolved table)
    """

    def __init__(self, rules_engine: BusinessRulesEngine, query_builder: QueryBuilder):
        self.rules_engine = rules_engine
        self.query_builder = query_builder

    def build(self, intent: ParsedIntent, question: str) -> list[BuiltQuery]:
        enriched = self.rules_engine.enrich(intent, question)
        logger.debug(f"Rules applied: {enriched.applied_rules}")
        queries = self.query_builder.build(enriched)
        if not queries:
            logger.warning("Stage 2: no queries built from intent.")
        return queries
