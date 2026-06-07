"""
ChatbotChain — LCEL pipeline composing all four stages.
Supports metadata-driven path and LangChain fallback path.
All stages are traced automatically via LangSmith when LANGSMITH_API_KEY is set.
"""
from __future__ import annotations
import logging
import time
from langsmith import traceable
from models.intent import ParsedIntent
from models.query import ChatResponse, ExecutionResult
from engine.pipeline.question_parser import QuestionParser
from engine.pipeline.query_builder import QueryBuilderStage
from engine.pipeline.query_executor import QueryExecutor
from engine.pipeline.nl_responder import NLResponder

logger = logging.getLogger(__name__)


class ChatbotChain:
    """
    Orchestrates the full pipeline:

        Question
          │
          ▼ QuestionParser (Stage 1 — LLM + PydanticOutputParser)
          │
          ├─ confidence ≥ threshold
          │       ▼ QueryBuilderStage (Stage 2 — metadata-driven, no LLM)
          │
          └─ confidence < threshold
                  ▼ LangChain SQL Chain fallback
          │
          ▼ QueryExecutor (Stage 3 — LangChain QuerySQLDataBaseTool)
          │
          ▼ NLResponder (Stage 4 — LLM + programmatic enumeration)
          │
          ▼ ChatResponse
    """

    def __init__(
        self,
        parser: QuestionParser,
        query_builder: QueryBuilderStage,
        executor: QueryExecutor,
        responder: NLResponder,
        fallback_chain=None,   # LangChain create_sql_query_chain instance
    ):
        self.parser = parser
        self.query_builder = query_builder
        self.executor = executor
        self.responder = responder
        self.fallback_chain = fallback_chain

    # Meta-question phrases — intercepted before invoking the LLM
    _META_PHRASES = [
        "what data", "what tables", "what metrics", "what can you",
        "what can i ask", "what information do you", "what do you know",
        "what is available", "data available", "what topics",
        "what questions", "how can you help", "what are you",
        "help me understand", "what kind of questions",
    ]

    @traceable(name="prismage_answer")
    def answer(self, question: str, verbose: bool = False) -> ChatResponse:
        """
        Run the full pipeline and return a ChatResponse.

        verbose=True prints step-by-step progress to stdout as each stage runs,
        matching the old engine terminal output format (STEP 1 → 4 with timing).
        """
        try:
            # Short-circuit meta/help questions before invoking the LLM.
            if self._is_meta_question(question):
                return ChatResponse(
                    question=question,
                    answer=(
                        "I'm a data query assistant. I can answer questions about your data — "
                        "for example: revenue by region, top products by sales, performance vs target, "
                        "MTD/YTD trends, customer satisfaction, and more.\n\n"
                        "Try asking something like:\n"
                        "  • 'Show revenue by region'\n"
                        "  • 'Top 5 products by sales this month'\n"
                        "  • 'Which sales reps are below target?'\n"
                        "  • 'MTD growth by customer segment'"
                    ),
                    success=True,
                )

            # ── STEP 1: Parse question + Build SQL queries ────────────────────
            t1 = time.time()
            intent, use_fallback = self.parser.parse(question)
            if use_fallback and self.fallback_chain:
                queries, used_fallback = self._run_fallback(question, intent)
            else:
                queries = self.query_builder.build(intent, question)
                used_fallback = False
            t1_elapsed = time.time() - t1

            if verbose:
                n = len(queries)
                label = "query" if n == 1 else "queries"
                print(f"\n📊 STEP 1: BUILDING SQL QUERIES")
                print(f"   ✅ Generated {n} separate {label} ({t1_elapsed:.2f}s)")
                for i, q in enumerate(queries, 1):
                    print(f"      Query {i}: {q.table} ({len(q.sql)} chars)")

            if not queries:
                if self._is_empty_intent(intent):
                    answer = (
                        "I'm a data query assistant. I can answer questions about your "
                        "data — for example: revenue by region, top products by sales, "
                        "performance vs target, or MTD growth. "
                        "Try asking something like: 'Show revenue by region' or "
                        "'What are the top 5 products by sales?'"
                    )
                else:
                    answer = "I could not build a query for your question. Please try rephrasing."
                return ChatResponse(question=question, answer=answer, success=False)

            # ── STEP 2: Execute queries ───────────────────────────────────────
            if verbose:
                n = len(queries)
                label = "QUERY" if n == 1 else "QUERIES"
                print(f"\n📊 STEP 2: EXECUTING {n} SQL {label}")

            def _on_query(i, total, q):
                if verbose:
                    print(f"   Executing query {i + 1}/{total}: {q.table}...")

            t2 = time.time()
            execution_result: ExecutionResult = self.executor.execute(
                queries, progress_callback=_on_query
            )
            t2_elapsed = time.time() - t2
            execution_result.used_fallback = used_fallback
            execution_result.applied_rules = intent.applied_rules

            # ── Re-sort merged results by intent.sort ─────────────────────────
            # ResultMerger rebuilds row order alphabetically by join key, discarding
            # the SQL ORDER BY. Re-apply the intent's sort direction here so that
            # intent.limit slices the correct top/bottom rows.
            if intent.sort:
                sort_metric = intent.sort.metric
                sort_desc = intent.sort.direction.upper() == "DESC"
                for r in execution_result.query_results:
                    if not (r.success and r.rows):
                        continue
                    # Find column: exact name or first column prefixed by metric name
                    sort_col = sort_metric if sort_metric in r.columns else next(
                        (c for c in r.columns if c.startswith(sort_metric + "_")), None
                    )
                    if sort_col:
                        r.rows.sort(
                            key=lambda row, sc=sort_col: (
                                row.get(sc) is None, row.get(sc) or 0
                            ),
                            reverse=sort_desc,
                        )

            # ── Apply intent.limit post-merge ─────────────────────────────────
            # SQL always fetches DEFAULT_LIMIT rows; trim each channel group to
            # limit after within-channel merges (value+volume → 1 group/channel).
            # Primary and secondary always stay as separate result groups.
            if intent.limit:
                good = [r for r in execution_result.query_results if r.success and r.rows]
                for r in good:
                    if len(r.rows) > intent.limit:
                        r.rows = r.rows[:intent.limit]
                        r.row_count = intent.limit

            good = [r for r in execution_result.query_results if r.success and r.rows]
            execution_result.total_rows = sum(r.row_count for r in good)
            execution_result.merged_group_count = len(good)
            execution_result.programmatic_enumeration = (
                self.executor.build_programmatic_enumeration(good)
            )

            if verbose:
                n_ok = sum(1 for r in execution_result.query_results if r.success)
                print(f"   ✅ Executed {n_ok}/{len(queries)} queries successfully ({t2_elapsed:.2f}s)")
                print(f"   Total rows from all queries: {execution_result.raw_row_count}")

            # ── STEP 3: Merge results ─────────────────────────────────────────
            # Merge happens inside execute(); we just report the outcome.
            if verbose:
                g = execution_result.merged_group_count
                print(f"\n📊 STEP 3: MERGING RESULTS ({g} group(s))")
                print(f"   ✅ Merged into {execution_result.total_rows} total rows "
                      f"across {g} group(s) (0.00s)")

            # ── STEP 4: Format tabular output + generate NL response ──────────
            if verbose:
                print(f"\n📊 STEP 4: FORMATTING TABULAR OUTPUT & GENERATING RESPONSE")

            t4 = time.time()
            successful = [r for r in execution_result.query_results if r.success and r.rows]

            # Time per-group tabular formatting (fast, but shows the groups)
            for i, result in enumerate(successful, 1):
                t4a = time.time()
                # (actual table building happens in main() from query_results data)
                t4a_elapsed = time.time() - t4a
                if verbose:
                    lbl = result.query.channel or result.query.table
                    print(f"   📝 Formatting group {i}/{len(successful)} ({lbl})... "
                          f"✅ ({result.row_count} rows, {t4a_elapsed:.2f}s)")

            # NL response (the slow LLM call)
            t4b = time.time()
            answer, summary, detail = self.responder.respond(
                execution_result, intent, question
            )
            t4b_elapsed = time.time() - t4b
            t4_elapsed = time.time() - t4

            if verbose:
                print(f"   🤖 Generating natural language response... ✅ ({t4b_elapsed:.2f}s)")
                print(f"   ⏱️  Total Step 4 time: {t4_elapsed:.2f}s")

            # Serialise query results for the CLI tabular display.
            # Filter columns to only those the question asked for (display_metrics),
            # stripping any context metrics added by business rules.
            display_cols = self._display_columns(intent, successful)
            query_results_data = [
                {
                    "table": r.query.table,
                    "channel": r.query.channel,
                    "columns": display_cols.get(i, r.columns),
                    "rows": [
                        {k: v for k, v in row.items() if k in display_cols.get(i, r.columns)}
                        for row in r.rows
                    ],
                    "row_count": r.row_count,
                }
                for i, r in enumerate(successful)
            ]

            return ChatResponse(
                question=question,
                answer=answer,
                summary=summary,
                detail=detail,
                tabular=execution_result.programmatic_enumeration,
                success=True,
                sql_queries=[{"sql": q.sql, "table": q.table, "channel": q.channel} for q in queries],
                total_rows=execution_result.total_rows,
                used_fallback=used_fallback,
                applied_rules=intent.applied_rules,
                step_timings={
                    "step1_s": round(t1_elapsed, 2),
                    "step2_s": round(t2_elapsed, 2),
                    "step3_s": 0.0,
                    "step4_s": round(t4_elapsed, 2),
                },
                raw_query_count=len(queries),
                raw_row_count=execution_result.raw_row_count,
                merged_group_count=execution_result.merged_group_count,
                query_results=query_results_data,
            )

        except Exception as e:
            logger.exception(f"ChatbotChain error: {e}")
            return ChatResponse(
                question=question,
                answer="An internal error occurred. Please try again.",
                success=False,
                error=str(e),
            )

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _display_columns(self, intent: ParsedIntent, results: list) -> dict:
        """
        Return a per-result-index dict of column names to show in the tabular display.

        Only dimension columns and question-requested metrics (with any _val/_vol suffix)
        are included. If display_metrics is empty (no filtering configured), all columns
        are returned unchanged.
        """
        if not intent.display_metrics:
            return {}

        display_set = set(intent.display_metrics)

        result_cols: dict[int, list[str]] = {}
        for i, r in enumerate(results):
            kept = []
            for col in r.columns:
                # Always keep dimension-like (string) columns
                sample = r.rows[0].get(col) if r.rows else None
                if isinstance(sample, str) or sample is None:
                    kept.append(col)
                    continue
                # Keep metric col if its base name (strip _val/_vol suffix) is in display_metrics
                base = col
                for sfx in ("_val", "_vol"):
                    if col.endswith(sfx):
                        base = col[: -len(sfx)]
                        break
                if base in display_set:
                    kept.append(col)
            result_cols[i] = kept

        return result_cols

    def _is_meta_question(self, question: str) -> bool:
        """True when the question is a help/meta query with no data intent."""
        q = question.lower().strip()
        return any(phrase in q for phrase in self._META_PHRASES)

    def _is_empty_intent(self, intent: ParsedIntent) -> bool:
        """True when the intent has no queryable fields — likely a meta/general question."""
        return not (
            intent.dimensions
            or intent.metrics
            or intent.formula_metrics
            or intent.filters
            or intent.having
        )

    # ── Fallback ─────────────────────────────────────────────────────────────

    @traceable(name="prismage_fallback_sql")
    def _run_fallback(self, question: str, intent: ParsedIntent):
        """Run LangChain SQL chain as fallback and wrap output into BuiltQuery list."""
        from models.query import BuiltQuery
        logger.info("Using LangChain SQL chain fallback.")
        try:
            sql = self.fallback_chain.invoke({"question": question})
            return [BuiltQuery(sql=sql, table="fallback", channel="fallback")], True
        except Exception as e:
            logger.error(f"Fallback chain error: {e}")
            return [], True
