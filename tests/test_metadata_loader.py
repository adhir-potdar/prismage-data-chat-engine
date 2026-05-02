"""Tests for MetadataLoader and MetadataValidator."""
import pytest
from pathlib import Path
from engine.metadata.loader import MetadataLoader
from engine.metadata.validator import MetadataValidator

CONFIG_DIR = str(Path(__file__).parent.parent / "config" / "metadata")


def test_loader_loads_dimensions():
    config = MetadataLoader(CONFIG_DIR).load()
    assert len(config.dimensions) > 0
    names = [d.name for d in config.dimensions]
    assert "region" in names
    assert "product_name" in names


def test_loader_loads_metrics():
    config = MetadataLoader(CONFIG_DIR).load()
    assert len(config.metrics) > 0
    names = [m.name for m in config.metrics]
    assert "revenue" in names
    assert "mtd_revenue" in names


def test_loader_metric_categories():
    config = MetadataLoader(CONFIG_DIR).load()
    cats = {m.category.value for m in config.metrics}
    assert "absolute" in cats
    assert "average" in cats
    assert "cumulative" in cats
    assert "percentage" in cats
    assert "formula" in cats


def test_loader_loads_formulas():
    config = MetadataLoader(CONFIG_DIR).load()
    assert len(config.formulas) > 0
    names = [f.name for f in config.formulas]
    assert "profit_margin_pct" in names
    assert "revenue_run_rate" in names


def test_loader_loads_tables():
    config = MetadataLoader(CONFIG_DIR).load()
    assert len(config.tables) > 0
    names = [t.name for t in config.tables]
    assert "orders" in names


def test_loader_loads_business_rules():
    config = MetadataLoader(CONFIG_DIR).load()
    assert config.domain_context
    assert len(config.rules) > 0
    assert len(config.metric_hints) > 0
    assert len(config.having_patterns) > 0


def test_validator_passes_on_valid_config():
    config = MetadataLoader(CONFIG_DIR).load()
    MetadataValidator(config).validate()   # should not raise


def test_validator_catches_broken_formula_ref():
    config = MetadataLoader(CONFIG_DIR).load()
    config.metrics[0].formula_ref = "nonexistent_formula"
    config.metrics[0].category = type("C", (), {"value": "percentage"})()
    with pytest.raises(ValueError, match="unknown formula"):
        MetadataValidator(config).validate()
