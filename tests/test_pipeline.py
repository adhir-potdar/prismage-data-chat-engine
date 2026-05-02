"""
Integration tests for the pipeline stages:
  - QueryBuilderStage (rules_engine + query_builder wired together)
  - End-to-end intent → SQL path (no LLM required)
"""
import pytest
from pathlib import Path
from engine.metadata.loader import MetadataLoader
from engine.metadata.registry import MetadataRegistry
from engine.rules.engine import BusinessRulesEngine
from engine.query.router import TableRouter
from engine.query.formula_engine import FormulaEngine, QueryContext
from engine.query.having_engine import HavingEngine
from engine.query.builder import QueryBuilder
from engine.pipeline.query_builder import QueryBuilderStage
from models.intent import ParsedIntent, SortConfig

FIXTURES_DIR = str(Path(__file__).parent / "fixtures")


@pytest.fixture
def stage():
    config = MetadataLoader(FIXTURES_DIR).load()
    registry = MetadataRegistry(config)
    rules_engine = BusinessRulesEngine(config.rules, registry)
    router = TableRouter(registry)
    formula_engine = FormulaEngine(registry)
    having_engine = HavingEngine(registry, config.having_patterns)
    query_builder = QueryBuilder(registry, router, formula_engine, having_engine)
    return QueryBuilderStage(rules_engine, query_builder)


def _intent(**kwargs) -> ParsedIntent:
    defaults = dict(dimensions=[], metrics=[], formula_metrics=[], filters={})
    defaults.update(kwargs)
    return ParsedIntent(**defaults)


# ── QueryBuilderStage wires rules + builder ───────────────────────────────────

def test_stage_enrich_and_build(stage):
    intent = _intent(dimensions=["region"], metrics=["revenue"])
    queries = stage.build(intent, "show revenue by region")
    assert len(queries) > 0
    assert "revenue" in queries[0].sql


def test_stage_applies_rules_before_building(stage):
    """Keyword 'top' triggers sort rule; SQL should have ORDER BY DESC."""
    intent = _intent(dimensions=["region"], metrics=["revenue"])
    queries = stage.build(intent, "show top regions by revenue")
    sql = queries[0].sql
    assert "ORDER BY" in sql
    assert "DESC" in sql


def test_stage_adds_formula_via_highlight_rule(stage):
    """'highlight' keyword adds profit_margin_pct formula to intent before building."""
    intent = _intent(dimensions=["region"], metrics=["revenue"])
    queries = stage.build(intent, "highlight underperforming regions")
    sql = queries[0].sql
    assert "profit_margin_pct" in sql


def test_stage_cumulative_metric_adds_growth_formula(stage):
    """MTD metric triggers cumulative rule which adds mtd_growth_pct formula."""
    intent = _intent(dimensions=["region"], metrics=["mtd_revenue"])
    queries = stage.build(intent, "show MTD revenue by region")
    sql = queries[0].sql
    assert "mtd_growth_pct" in sql


# ── BuiltQuery metadata ───────────────────────────────────────────────────────

def test_built_query_has_table_name(stage):
    intent = _intent(dimensions=["region"], metrics=["revenue"])
    queries = stage.build(intent, "show revenue by region")
    assert queries[0].table == "orders"


def test_built_query_has_channel(stage):
    intent = _intent(dimensions=["region"], metrics=["revenue"])
    queries = stage.build(intent, "show revenue by region")
    assert queries[0].channel is not None


# ── Empty and edge cases ──────────────────────────────────────────────────────

def test_empty_intent_produces_no_queries(stage):
    intent = _intent()
    queries = stage.build(intent, "")
    assert queries == []


def test_unknown_metric_skipped_gracefully(stage):
    """Metrics not in any table should not cause an error."""
    intent = _intent(dimensions=["region"], metrics=["revenue", "nonexistent_metric"])
    queries = stage.build(intent, "show revenue by region")
    # Should still produce a query for revenue
    assert len(queries) > 0


# ── Intent preserves applied_rules after stage ────────────────────────────────

def test_applied_rules_set_on_intent_after_stage(stage):
    intent = _intent(dimensions=["region"], metrics=["revenue"])
    stage.build(intent, "show top regions by revenue")
    # intent is enriched in-place, applied_rules should be set
    assert isinstance(intent.applied_rules, list)
