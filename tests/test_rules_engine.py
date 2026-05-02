"""Tests for BusinessRulesEngine — trigger evaluation and action application."""
import pytest
from pathlib import Path
from engine.metadata.loader import MetadataLoader
from engine.metadata.registry import MetadataRegistry
from engine.rules.engine import BusinessRulesEngine
from models.intent import ParsedIntent, HavingConfig, HavingCondition

FIXTURES_DIR = str(Path(__file__).parent / "fixtures")


@pytest.fixture
def setup():
    config = MetadataLoader(FIXTURES_DIR).load()
    registry = MetadataRegistry(config)
    engine = BusinessRulesEngine(config.rules, registry)
    return engine, registry


def _intent(**kwargs) -> ParsedIntent:
    defaults = dict(
        dimensions=[],
        metrics=[],
        formula_metrics=[],
        filters={},
    )
    defaults.update(kwargs)
    return ParsedIntent(**defaults)


# ── Keyword trigger ───────────────────────────────────────────────────────────

def test_keyword_trigger_fires(setup):
    engine, _ = setup
    intent = _intent(metrics=["revenue"])
    result = engine.enrich(intent, "highlight the top products")
    assert "highlight_adds_margin" in result.applied_rules


def test_keyword_trigger_case_insensitive(setup):
    engine, _ = setup
    intent = _intent(metrics=["revenue"])
    result = engine.enrich(intent, "HIGHLIGHT underperforming regions")
    assert "highlight_adds_margin" in result.applied_rules


def test_keyword_trigger_no_match(setup):
    engine, _ = setup
    intent = _intent(metrics=["revenue"])
    result = engine.enrich(intent, "show me revenue by region")
    assert "highlight_adds_margin" not in result.applied_rules


# ── ensure_formula action ─────────────────────────────────────────────────────

def test_ensure_formula_added_to_intent(setup):
    engine, _ = setup
    intent = _intent(metrics=["revenue"])
    result = engine.enrich(intent, "highlight the underperforming products")
    assert "profit_margin_pct" in result.formula_metrics


def test_ensure_formula_not_duplicated(setup):
    engine, _ = setup
    intent = _intent(metrics=["revenue"], formula_metrics=["profit_margin_pct"])
    result = engine.enrich(intent, "highlight the underperforming products")
    assert result.formula_metrics.count("profit_margin_pct") == 1


# ── set_sort action ───────────────────────────────────────────────────────────

def test_sort_rule_sets_direction(setup):
    engine, _ = setup
    intent = _intent(metrics=["revenue"])
    result = engine.enrich(intent, "show top 5 regions by revenue")
    assert result.sort is not None
    assert result.sort.direction == "DESC"
    assert result.sort.metric == "revenue"


# ── metric_category trigger ───────────────────────────────────────────────────

def test_metric_category_trigger_fires_for_cumulative(setup):
    engine, _ = setup
    intent = _intent(metrics=["mtd_revenue"])
    result = engine.enrich(intent, "show MTD revenue by region")
    assert "cumulative_comparison_adds_growth_pct" in result.applied_rules
    assert "mtd_growth_pct" in result.formula_metrics


def test_metric_category_trigger_does_not_fire_for_absolute(setup):
    engine, _ = setup
    intent = _intent(metrics=["revenue"])
    result = engine.enrich(intent, "show revenue by region")
    assert "cumulative_comparison_adds_growth_pct" not in result.applied_rules


# ── Multiple rules fire ───────────────────────────────────────────────────────

def test_multiple_rules_can_fire(setup):
    engine, _ = setup
    intent = _intent(metrics=["mtd_revenue"])
    result = engine.enrich(intent, "highlight top performing regions by MTD revenue")
    # keyword rules + metric_category rule should all fire
    assert "highlight_adds_margin" in result.applied_rules
    assert "top_performing_sort_desc" in result.applied_rules
    assert "cumulative_comparison_adds_growth_pct" in result.applied_rules


# ── applied_rules populated ───────────────────────────────────────────────────

def test_applied_rules_populated_on_intent(setup):
    engine, _ = setup
    intent = _intent(metrics=["revenue"])
    result = engine.enrich(intent, "show revenue by region")
    assert isinstance(result.applied_rules, list)
