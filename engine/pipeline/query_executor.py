"""
Stage 3 — Query Executor
Runs SQL queries against the database using LangChain's QuerySQLDataBaseTool.
Builds programmatic row enumeration for safe LLM consumption.
"""
from __future__ import annotations
import logging
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from models.query import BuiltQuery, QueryResult, ExecutionResult
from engine.pipeline.result_merger import ResultMerger

logger = logging.getLogger(__name__)


class QueryExecutor:
    """
    Stage 3: list[BuiltQuery] → ExecutionResult

    Uses LangChain's QuerySQLDataBaseTool for execution.
    Builds programmatic_enumeration — a structured text listing of ALL rows
    that is passed to Stage 4 instead of raw markdown tables, preventing
    the LLM from seeing blank cells or generating "(not available)" responses.
    """

    def __init__(self, db: SQLDatabase):
        self.db = db
        self.sql_tool = QuerySQLDataBaseTool(db=db)
        self._merger = ResultMerger()

    def execute(self, queries: list[BuiltQuery]) -> ExecutionResult:
        raw_results = []
        for query in queries:
            result = self._run_one(query)
            raw_results.append(result)

        # Merge results from the same channel that span multiple tables
        results = self._merge_by_channel(raw_results)

        total_rows = sum(r.row_count for r in results if r.success)
        enumeration = self._build_programmatic_enumeration(results)

        return ExecutionResult(
            query_results=results,
            total_rows=total_rows,
            programmatic_enumeration=enumeration,
        )

    # ── Merge helper ─────────────────────────────────────────────────────────

    def _merge_by_channel(self, results: list[QueryResult]) -> list[QueryResult]:
        """
        Group results by channel. If more than one successful result exists for
        the same channel, merge them via ResultMerger. Otherwise pass through.
        """
        from collections import defaultdict
        groups: dict[str, list[QueryResult]] = defaultdict(list)
        for r in results:
            groups[r.query.channel].append(r)

        merged: list[QueryResult] = []
        for channel, group in groups.items():
            successful = [r for r in group if r.success and r.rows]
            if len(successful) > 1:
                merged.append(self._merger.merge(successful))
            else:
                merged.extend(group)

        return merged

    # ── Private ──────────────────────────────────────────────────────────────

    def _run_one(self, query: BuiltQuery) -> QueryResult:
        try:
            raw = self.sql_tool.run(query.sql)
            rows, columns = self._parse_tool_output(raw)
            return QueryResult(
                query=query,
                columns=columns,
                rows=rows,
                row_count=len(rows),
                success=True,
            )
        except Exception as e:
            logger.error(f"SQL execution error on {query.table}: {e}")
            return QueryResult(query=query, error=str(e), success=False)

    def _parse_tool_output(self, raw: str) -> tuple[list[dict], list[str]]:
        """Parse LangChain SQL tool string output into rows + column list."""
        import ast
        try:
            data = ast.literal_eval(raw)
            if not data:
                return [], []
            columns = list(data[0].keys()) if isinstance(data[0], dict) else []
            return data, columns
        except Exception:
            return [], []

    def _build_programmatic_enumeration(self, results: list[QueryResult]) -> str:
        """
        Build a structured text enumeration of all result rows.
        This replaces raw markdown tables in the LLM prompt, guaranteeing
        every row is visible and blank cells are simply omitted.
        """
        sections = []
        global_idx = 1

        for result in results:
            if not result.success or not result.rows:
                continue

            table_label = result.query.channel.upper()
            lines = [f"### {table_label} ({result.row_count} rows)"]

            for row in result.rows:
                parts = [f"{global_idx}."]
                for col, val in row.items():
                    if val is not None and val != "":
                        parts.append(f"{col}: {val}")
                lines.append("  " + " | ".join(parts))
                global_idx += 1

            sections.append("\n".join(lines))

        return "\n\n".join(sections) if sections else "No data returned."
