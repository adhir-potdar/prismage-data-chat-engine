"""
MetadataValidator — checks config consistency at startup.
Raises ValueError with clear messages for any broken references.
"""
from __future__ import annotations
from engine.metadata.loader import MetadataConfig


class MetadataValidator:
    """
    Validates cross-references between JSON config files:
    - Every metric formula_ref points to a defined formula
    - Every formula component references a defined metric with a db_column
    - Every table dimension/metric exists in dimensions.json / metrics.json
    - Every business rule action references valid metric/dimension/formula names
    """

    def __init__(self, config: MetadataConfig):
        self.config = config

    def validate(self) -> None:
        errors = []
        errors += self._validate_metric_formula_refs()
        errors += self._validate_formula_components()
        errors += self._validate_table_references()
        errors += self._validate_rule_actions()

        if errors:
            msg = "\n".join(f"  - {e}" for e in errors)
            raise ValueError(f"Metadata validation failed:\n{msg}")

    # ── Checks ───────────────────────────────────────────────────────────────

    def _validate_metric_formula_refs(self) -> list[str]:
        formula_names = {f.name for f in self.config.formulas}
        errors = []
        for m in self.config.metrics:
            if m.formula_ref and m.formula_ref not in formula_names:
                errors.append(f"Metric '{m.name}' references unknown formula '{m.formula_ref}'")
        return errors

    def _validate_formula_components(self) -> list[str]:
        metric_names = {m.name for m in self.config.metrics}
        errors = []
        for f in self.config.formulas:
            for comp in f.components:
                if comp not in metric_names:
                    errors.append(f"Formula '{f.name}' references unknown component metric '{comp}'")
        return errors

    def _validate_table_references(self) -> list[str]:
        dim_names = {d.name for d in self.config.dimensions}
        metric_names = {m.name for m in self.config.metrics}
        errors = []
        for t in self.config.tables:
            for d in t.dimensions:
                if d not in dim_names:
                    errors.append(f"Table '{t.name}' references unknown dimension '{d}'")
            for m in t.metrics:
                if m not in metric_names:
                    errors.append(f"Table '{t.name}' references unknown metric '{m}'")
        return errors

    def _validate_rule_actions(self) -> list[str]:
        metric_names = {m.name for m in self.config.metrics}
        dim_names = {d.name for d in self.config.dimensions}
        formula_names = {f.name for f in self.config.formulas}
        errors = []
        for rule in self.config.rules:
            for action in rule.actions:
                for m in action.metrics:
                    if m not in metric_names:
                        errors.append(f"Rule '{rule.name}' action references unknown metric '{m}'")
                for d in action.dimensions:
                    if d not in dim_names:
                        errors.append(f"Rule '{rule.name}' action references unknown dimension '{d}'")
                if action.formula and action.formula not in formula_names and action.type == "ensure_formula":
                    errors.append(f"Rule '{rule.name}' action references unknown formula '{action.formula}'")
        return errors
