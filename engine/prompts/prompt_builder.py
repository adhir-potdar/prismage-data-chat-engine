"""
PromptBuilder — injects metadata and domain context into prompt templates.
Bridges PromptLibrary (raw templates) and the pipeline stages (runtime values).
"""
from __future__ import annotations
from jinja2 import Template
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from engine.prompts.prompt_library import PromptLibrary
from engine.metadata.registry import MetadataRegistry
from engine.metadata.loader import MetadataConfig


class PromptBuilder:
    """
    Renders Jinja2 prompt templates with live metadata values.

    Templates use {{ variable }} syntax (Jinja2) to reference:
      - domain_context      from business_rules.json
      - dimensions_list     rendered from registry
      - metrics_list        rendered from registry
      - formulas_list       rendered from registry
      - metric_hints        rendered from business_rules.json
      - few_shot_examples   rendered from registry + rules
    """

    def __init__(self, library: PromptLibrary, registry: MetadataRegistry, config: MetadataConfig):
        self.library = library
        self.registry = registry
        self.config = config

    def build_parser_prompt(self, question: str, format_instructions: str = "") -> ChatPromptTemplate:
        from datetime import date
        raw_system = self.library.get_system_template("question_parser")
        system = Template(raw_system).render(
            domain_context=self.config.domain_context,
            dimensions_list=self.registry.render_dimensions(),
            metrics_list=self.registry.render_metrics(),
            formulas_list=self.registry.render_formulas(),
            metric_hints=self._render_metric_hints(),
            few_shot_examples=self._render_few_shot_examples(),
            current_date=date.today().strftime("%Y%m%d"),
        )
        if format_instructions:
            system = system + "\n\n## Strict Output Requirement\nYou MUST always respond with valid JSON matching the schema below, even for meta-questions. Never respond with natural language.\n" + format_instructions
        # Escape literal braces so LangChain does not treat JSON in the
        # rendered system prompt as Python format-string variables.
        system = system.replace("{", "{{").replace("}", "}}")
        return ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "{question}"),
        ])

    def build_response_prompt(
        self,
        data_context: str,
        task_description: str,
        context_line: str,
    ) -> ChatPromptTemplate:
        raw_system = self.library.get_system_template("nl_response")
        system = Template(raw_system).render(
            domain_context=self.config.domain_context,
            data_context=data_context,
            task_description=task_description,
        )
        system = system.replace("{", "{{").replace("}", "}}")
        return ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", context_line),
        ])

    def build_fallback_prompt(self, question: str, db_schema: str, db_dialect: str = "SQL") -> ChatPromptTemplate:
        raw_system = self.library.get_system_template("query_generator")
        system = Template(raw_system).render(
            domain_context=self.config.domain_context,
            domain_name="retail",
            db_schema=db_schema,
            db_dialect=db_dialect,
            dimensions="",
            metrics="",
            filters="",
            date_range="",
            default_limit=100,
        )
        system = system.replace("{", "{{").replace("}", "}}")
        return ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "{question}"),
        ])

    # ── Private ──────────────────────────────────────────────────────────────

    def _render_metric_hints(self) -> str:
        lines = []
        for h in self.config.metric_hints:
            if h.maps_to_having:
                m = h.maps_to_having
                op = m.get("operator", "?")
                m1 = m.get("metric1", "")
                m2 = m.get("metric2", "")
                lines.append(
                    f'  "{h.phrase}" → having: metric_comparison, '
                    f'conditions: [{{"metric1": "{m1}", "operator": "{op}", "metric2": "{m2}"}}]'
                )
            elif h.polarity:
                lines.append(f'  "{h.phrase}" → polarity: {h.polarity}, operator: {h.operator}')
        return "\n".join(lines)

    def _render_few_shot_examples(self) -> str:
        return """
Example 1:
  Q: "Show top 5 products by revenue this month"
  → {"dimensions": ["product_name"], "metrics": ["revenue"], "formula_metrics": [],
     "having": null, "filters": {}, "date_range": null,
     "sort": {"metric": "revenue", "direction": "DESC"}, "limit": 5, "confidence": 0.95}

Example 2:
  Q: "Which sales reps are below target this month?"
  → {"dimensions": ["sales_rep"], "metrics": ["revenue", "target_revenue"], "formula_metrics": ["target_achievement_pct"],
     "having": {"type": "metric_comparison", "polarity": "negative",
                "conditions": [{"metric1": "revenue", "operator": "<", "metric2": "target_revenue"}]},
     "filters": {}, "date_range": null, "sort": null, "limit": null, "confidence": 0.92}

Example 3:
  Q: "Show MTD revenue growth by region for Q1 2025"
  → {"dimensions": ["region"], "metrics": ["mtd_revenue", "prev_mtd_revenue"], "formula_metrics": ["mtd_growth_pct"],
     "having": null, "filters": {},
     "date_range": {"start": "20250101", "end": "20250331"},
     "sort": {"metric": "mtd_revenue", "direction": "DESC"}, "limit": null, "confidence": 0.90}
"""
