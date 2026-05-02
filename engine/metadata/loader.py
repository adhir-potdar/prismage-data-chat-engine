"""
MetadataLoader — reads JSON config files and returns typed model objects.
"""
from __future__ import annotations
import json
from pathlib import Path

from models.metadata import (
    Dimension, Metric, Formula, Table,
    BusinessRule, MetricHint, HavingPatternDef, RuleTrigger, RuleAction,
)


class MetadataConfig:
    """Container for all metadata loaded from a config directory."""

    def __init__(self):
        self.dimensions: list[Dimension] = []
        self.metrics: list[Metric] = []
        self.formulas: list[Formula] = []
        self.tables: list[Table] = []
        self.domain_context: str = ""
        self.metric_hints: list[MetricHint] = []
        self.having_patterns: list[HavingPatternDef] = []
        self.rules: list[BusinessRule] = []


class MetadataLoader:
    """
    Loads all metadata JSON files from a config directory into typed objects.

    Expected directory layout:
        config_dir/
            metadata/
                dimensions.json
                metrics.json
                formulas.json
                tables.json
                business_rules.json

    Usage:
        loader = MetadataLoader("config/metadata")
        config = loader.load()
    """

    def __init__(self, config_dir: str):
        self.config_dir = Path(config_dir)

    def load(self) -> MetadataConfig:
        config = MetadataConfig()
        config.dimensions = self._load_dimensions()
        config.metrics = self._load_metrics()
        config.formulas = self._load_formulas()
        config.tables = self._load_tables()
        rules_data = self._load_business_rules()
        config.domain_context = rules_data.get("domain_context", "")
        config.metric_hints = [MetricHint(**h) for h in rules_data.get("metric_hints", [])]
        config.having_patterns = [HavingPatternDef(**p) for p in rules_data.get("having_patterns", [])]
        config.rules = [
            BusinessRule(
                name=r["name"],
                description=r.get("description"),
                trigger=RuleTrigger(**r["trigger"]),
                actions=[RuleAction(**a) for a in r.get("actions", [])],
            )
            for r in rules_data.get("rules", [])
        ]
        return config

    # ── Private helpers ──────────────────────────────────────────────────────

    def _read(self, filename: str) -> dict:
        path = self.config_dir / filename
        with open(path) as f:
            return json.load(f) or {}

    def _load_dimensions(self) -> list[Dimension]:
        data = self._read("dimensions.json")
        return [Dimension(**d) for d in data.get("dimensions", [])]

    def _load_metrics(self) -> list[Metric]:
        data = self._read("metrics.json")
        return [Metric(**m) for m in data.get("metrics", [])]

    def _load_formulas(self) -> list[Formula]:
        data = self._read("formulas.json")
        return [Formula(**f) for f in data.get("formulas", [])]

    def _load_tables(self) -> list[Table]:
        data = self._read("tables.json")
        return [Table(**t) for t in data.get("tables", [])]

    def _load_business_rules(self) -> dict:
        return self._read("business_rules.json")
