"""
BusinessRulesEngine — evaluates triggers and applies actions to enrich ParsedIntent.
Runs between Stage 1 (parser) and Stage 2 (query builder).
"""
from __future__ import annotations
import logging
from models.metadata import BusinessRule, MetricCategory
from models.intent import ParsedIntent, OutputHint, SortConfig
from engine.metadata.registry import MetadataRegistry

logger = logging.getLogger(__name__)


class BusinessRulesEngine:
    """
    Iterates over all rules in business_rules.json.
    For each rule whose trigger matches the intent + question, applies the
    configured actions to the ParsedIntent in place.

    Returns the enriched ParsedIntent with applied_rules populated.
    """

    def __init__(self, rules: list[BusinessRule], registry: MetadataRegistry):
        self.rules = rules
        self.registry = registry

    def enrich(self, intent: ParsedIntent, question: str) -> ParsedIntent:
        applied = []
        for rule in self.rules:
            if self._evaluate_trigger(rule.trigger, intent, question):
                for action in rule.actions:
                    self._apply_action(action, intent)
                applied.append(rule.name)
                logger.debug(f"Rule applied: {rule.name}")

        intent.applied_rules = applied
        return intent

    # ── Trigger evaluation ───────────────────────────────────────────────────

    def _evaluate_trigger(self, trigger, intent: ParsedIntent, question: str) -> bool:
        t = trigger.type

        if t == "keyword":
            q = question.lower()
            return any(p.lower() in q for p in trigger.phrases)

        elif t == "having_pattern":
            if not intent.having:
                return False
            type_match = intent.having.type == trigger.having_type
            if not trigger.polarity:
                return type_match
            return type_match and intent.having.polarity == trigger.polarity

        elif t == "formula_requested":
            return trigger.formula in intent.formula_metrics

        elif t == "metric_present":
            return any(m in intent.metrics or m in intent.formula_metrics
                       for m in trigger.metrics)

        elif t == "metric_category":
            for m_name in intent.metrics:
                cat = self.registry.get_metric_category(m_name)
                if cat and cat.value == trigger.category:
                    return True
            return False

        elif t == "compound":
            return all(
                self._evaluate_trigger(
                    type("T", (), c)(), intent, question
                )
                for c in trigger.conditions
            )

        elif t == "any_of":
            return any(
                self._evaluate_trigger(
                    type("T", (), c)(), intent, question
                )
                for c in trigger.conditions
            )

        return False

    # ── Action handlers ──────────────────────────────────────────────────────

    def _apply_action(self, action, intent: ParsedIntent) -> None:
        t = action.type

        if t == "ensure_metrics":
            for m in action.metrics:
                if m not in intent.metrics and m not in intent.formula_metrics:
                    cat = self.registry.get_metric_category(m)
                    if cat in (MetricCategory.PERCENTAGE, MetricCategory.FORMULA):
                        intent.formula_metrics.append(m)
                    else:
                        intent.metrics.append(m)

        elif t == "ensure_dimensions":
            for d in action.dimensions:
                if d not in intent.dimensions:
                    intent.dimensions.append(d)

        elif t == "ensure_formula":
            if action.formula not in intent.formula_metrics:
                intent.formula_metrics.append(action.formula)

        elif t == "create_formula":
            self.registry.register_formula(
                name=action.name,
                expression=action.expression,
                components=action.components,
                label=action.label or action.name,
            )
            if action.name not in intent.formula_metrics:
                intent.formula_metrics.append(action.name)
            intent.inline_formulas.append({
                "name": action.name,
                "expression": action.expression,
                "components": action.components,
            })

        elif t == "override_formula":
            self.registry.override_formula(
                name=action.formula,
                expression=action.expression,
                components=action.components,
                label=action.label or action.formula,
            )
            intent.formula_overrides[action.formula] = {
                "expression": action.expression,
                "components": action.components,
            }

        elif t == "set_output_hint":
            intent.output_hints = OutputHint(
                format=action.format or "table",
                always_programmatic_enumeration=action.always_programmatic_enumeration,
                columns=action.columns,
                primary_columns=action.primary_columns,
                secondary_columns=action.secondary_columns,
            )

        elif t == "set_channel":
            intent.channels = action.channels

        elif t == "set_sort":
            intent.sort = SortConfig(
                metric=action.metric,
                direction=action.direction or "DESC",
            )
