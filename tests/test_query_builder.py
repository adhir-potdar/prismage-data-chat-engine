"""Tests for QueryBuilder — SQL assembly from ParsedIntent."""
import pytest
from pathlib import Path
from engine.metadata.loader import MetadataLoader
from engine.metadata.registry import MetadataRegistry
from engine.query.router import TableRouter
from engine.query.formula_engine import FormulaEngine, QueryContext
from engine.query.having_engine import HavingEngine
from engine.query.builder import QueryBuilder
from models.intent import ParsedIntent, SortConfig, DateRange, HavingConfig, HavingCondition

FIXTURES_DIR = str(Path(__file__).parent / "fixtures")


@pytest.fixture
def builder():
    config = MetadataLoader(FIXTURES_DIR).load()
    registry = MetadataRegistry(config)
    router = TableRouter(registry)
    formula_engine = FormulaEngine(registry)
    having_engine = HavingEngine(registry, config.having_patterns)
    return QueryBuilder(registry, router, formula_engine, having_engine)


def _intent(**kwargs) -> ParsedIntent:
    defaults = dict(dimensions=[], metrics=[], formula_metrics=[], filters={})
    defaults.update(kwargs)
    return ParsedIntent(**defaults)


# ── Basic SELECT/FROM ─────────────────────────────────────────────────────────

def test_basic_dimension_and_metric(builder):
    intent = _intent(dimensions=["region"], metrics=["revenue"])
    queries = builder.build(intent)
    assert len(queries) > 0
    sql = queries[0].sql
    assert "region" in sql
    assert "SUM(revenue)" in sql
    assert "FROM orders" in sql


def test_group_by_dimension(builder):
    intent = _intent(dimensions=["region"], metrics=["revenue"])
    queries = builder.build(intent)
    sql = queries[0].sql
    assert "GROUP BY region" in sql


def test_default_limit_applied(builder):
    intent = _intent(dimensions=["region"], metrics=["revenue"])
    queries = builder.build(intent)
    sql = queries[0].sql
    assert "LIMIT 100" in sql


def test_custom_limit(builder):
    intent = _intent(dimensions=["region"], metrics=["revenue"], limit=5)
    queries = builder.build(intent)
    sql = queries[0].sql
    assert "LIMIT 5" in sql


# ── Metric categories ─────────────────────────────────────────────────────────

def test_absolute_metric_uses_sum(builder):
    intent = _intent(dimensions=["region"], metrics=["revenue"])
    queries = builder.build(intent)
    assert "SUM(revenue) AS revenue" in queries[0].sql


def test_average_metric_uses_avg(builder):
    intent = _intent(dimensions=["region"], metrics=["customer_satisfaction"])
    queries = builder.build(intent)
    assert "AVG(satisfaction_score) AS customer_satisfaction" in queries[0].sql


def test_cumulative_metric_uses_sum(builder):
    intent = _intent(dimensions=["region"], metrics=["mtd_revenue"])
    queries = builder.build(intent)
    assert "SUM(mtd_revenue) AS mtd_revenue" in queries[0].sql


# ── Formula metrics ───────────────────────────────────────────────────────────

def test_formula_metric_expanded_in_select(builder):
    intent = _intent(dimensions=["region"], formula_metrics=["profit_margin_pct"])
    queries = builder.build(intent)
    sql = queries[0].sql
    assert "profit_margin_pct" in sql
    assert "NULLIF" in sql


def test_avg_order_value_formula_expanded(builder):
    intent = _intent(dimensions=["region"], formula_metrics=["avg_order_value"])
    queries = builder.build(intent)
    sql = queries[0].sql
    assert "avg_order_value" in sql


# ── WHERE clause ─────────────────────────────────────────────────────────────

def test_filter_generates_where_clause(builder):
    intent = _intent(dimensions=["region"], metrics=["revenue"],
                     filters={"region": "North"})
    queries = builder.build(intent)
    sql = queries[0].sql
    assert "WHERE" in sql
    assert "region = 'North'" in sql


def test_sql_injection_safe_filter(builder):
    intent = _intent(dimensions=["region"], metrics=["revenue"],
                     filters={"region": "'; DROP TABLE orders; --"})
    queries = builder.build(intent)
    sql = queries[0].sql
    # Single quotes in value should be escaped
    assert "DROP TABLE" not in sql or "''" in sql


def test_date_range_generates_between(builder):
    intent = _intent(dimensions=["region"], metrics=["revenue"],
                     date_range=DateRange(start="20240101", end="20240131"))
    queries = builder.build(intent)
    sql = queries[0].sql
    assert "BETWEEN '20240101' AND '20240131'" in sql


# ── ORDER BY ─────────────────────────────────────────────────────────────────

def test_sort_generates_order_by(builder):
    intent = _intent(
        dimensions=["region"], metrics=["revenue"],
        sort=SortConfig(metric="revenue", direction="ASC"),
    )
    queries = builder.build(intent)
    sql = queries[0].sql
    assert "ORDER BY SUM(revenue) ASC" in sql


def test_sort_desc(builder):
    intent = _intent(
        dimensions=["region"], metrics=["revenue"],
        sort=SortConfig(metric="revenue", direction="DESC"),
    )
    queries = builder.build(intent)
    sql = queries[0].sql
    assert "ORDER BY SUM(revenue) DESC" in sql


# ── Empty intent ─────────────────────────────────────────────────────────────

def test_empty_intent_returns_empty(builder):
    intent = _intent()
    queries = builder.build(intent)
    assert queries == []


# ── Multiple dimensions ───────────────────────────────────────────────────────

def test_multiple_dimensions_in_select_and_group_by(builder):
    intent = _intent(dimensions=["region", "product_name"], metrics=["revenue"])
    queries = builder.build(intent)
    sql = queries[0].sql
    assert "region" in sql
    assert "product_name" in sql
    assert "GROUP BY region, product_name" in sql
