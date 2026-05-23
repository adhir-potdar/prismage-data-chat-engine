"""
Stage 4 — NL Responder
Generates a natural language answer from query execution results.
Uses programmatic enumeration as LLM input — never raw markdown tables.
"""
from __future__ import annotations
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from models.intent import ParsedIntent
from models.query import ExecutionResult
from engine.prompts.prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)


class NLResponder:
    """
    Stage 4: ExecutionResult + ParsedIntent → natural language answer string

    Key design:
    - programmatic_enumeration is always used as data_context (not raw table)
    - task_description is overridden to prevent LLM re-filtering pre-validated data
    - output_hints from business rules control table vs narrative format
    """

    def __init__(self, llm: BaseChatModel, prompt_builder: PromptBuilder):
        self.llm = llm
        self.prompt_builder = prompt_builder
        self.output_parser = StrOutputParser()

    def respond(
        self,
        execution_result: ExecutionResult,
        intent: ParsedIntent,
        question: str,
    ) -> tuple[str, str | None, str | None]:
        """
        Returns (full_answer, summary, detail).

        full_answer — complete LLM response text
        summary     — text after "QUICK SUMMARY:" up to the next section
        detail      — text after "ANALYSIS:"
        """
        if not execution_result.programmatic_enumeration or \
                execution_result.programmatic_enumeration == "No data returned.":
            msg = "No data was found for your query. Please refine your question or check the filters."
            return msg, None, None

        # Apply display row limit from intent before passing to LLM
        display_limit = intent.limit or 20
        data_context = self._apply_row_limit(
            execution_result.programmatic_enumeration, display_limit
        )
        total_rows = execution_result.total_rows

        task_description = self._build_task_description(total_rows, intent)
        context_line = self._build_context_line(question, intent)

        prompt = self.prompt_builder.build_response_prompt(
            data_context=data_context,
            task_description=task_description,
            context_line=context_line,
        )

        try:
            chain = prompt | self.llm | self.output_parser
            full_answer: str = chain.invoke({})
        except Exception as e:
            logger.error(f"Stage 4 LLM error: {e}")
            full_answer = data_context  # graceful fallback

        summary, detail = self._extract_sections(full_answer)
        return full_answer, summary, detail

    # ── Private ──────────────────────────────────────────────────────────────

    def _apply_row_limit(self, enumeration: str, limit: int) -> str:
        """
        Cap the enumeration to at most `limit` numbered entries per section.
        Entries are lines that start with whitespace + a digit (e.g. "  1. col: val").
        """
        lines = enumeration.splitlines()
        output: list[str] = []
        entry_count = 0
        truncated = False

        for line in lines:
            stripped = line.lstrip()
            # Section header (### PRIMARY ...)
            if stripped.startswith("###"):
                entry_count = 0
                truncated = False
                output.append(line)
                continue

            is_entry = stripped and stripped[0].isdigit() and "." in stripped[:4]
            if is_entry:
                entry_count += 1
                if entry_count > limit:
                    if not truncated:
                        output.append(f"  [... truncated at {limit} rows]")
                        truncated = True
                    continue

            output.append(line)

        return "\n".join(output)

    def _extract_sections(self, text: str) -> tuple[str | None, str | None]:
        """
        Parse 'QUICK SUMMARY: ...\n\nANALYSIS:\n...' from LLM output.
        Returns (summary_text, analysis_text) or (None, None) if not found.
        """
        summary: str | None = None
        detail: str | None = None

        if "QUICK SUMMARY:" in text:
            after_qs = text.split("QUICK SUMMARY:", 1)[1]
            # Summary ends at the next blank line or ANALYSIS:
            for separator in ["\n\nANALYSIS:", "\nANALYSIS:", "\n\n"]:
                if separator in after_qs:
                    summary = after_qs.split(separator, 1)[0].strip()
                    break
            else:
                summary = after_qs.strip()

        if "ANALYSIS:" in text:
            detail = text.split("ANALYSIS:", 1)[1].strip()

        return summary, detail

    # ── Private ──────────────────────────────────────────────────────────────

    def _build_task_description(self, total_rows: int, intent: ParsedIntent) -> str:
        if total_rows == 0:
            return "No results were returned. Inform the user clearly."

        # Programmatic mode: override task to prevent LLM re-filtering
        return (
            f"The {total_rows} entries listed above are the COMPLETE, PRE-VALIDATED results "
            f"already filtered by SQL. Provide business insights about ALL of them. "
            f"DO NOT re-evaluate, re-filter, or question whether entries qualify. "
            f"Describe patterns, highlight notable entries, and summarise overall findings."
        )

    def _build_context_line(self, question: str, intent: ParsedIntent) -> str:
        # Suppress the original question in the prompt to prevent LLM
        # re-applying its own filter semantics from the question text.
        # Only suppress if rules actually enriched the intent with useful fields —
        # a rule may fire on a generic question without adding any dimensions/metrics,
        # in which case we still want the LLM to see the original question.
        has_enrichment = bool(intent.applied_rules) and bool(
            intent.metrics or intent.dimensions or intent.formula_metrics
        )
        if has_enrichment:
            return "Context: The following are pre-filtered sales data entries. Provide business insights and analysis."
        return f"User Question: {question}"
