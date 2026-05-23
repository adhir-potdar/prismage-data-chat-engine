"""
Stage 3 — Query Executor
Runs SQL queries against the database using LangChain's QuerySQLDataBaseTool.
Builds programmatic row enumeration for safe LLM consumption.
"""
from __future__ import annotations
import logging
from langchain_community.utilities import SQLDatabase
from langchain_community.tools import QuerySQLDatabaseTool as QuerySQLDataBaseTool
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

    def execute(self, queries: list[BuiltQuery], progress_callback=None) -> ExecutionResult:
        """
        Execute all queries and merge within-channel results.

        Within-channel merge: multiple sub-table queries sharing the same
        channel are outer-joined into a single group per channel.
        Cross-channel merging (keeping each channel as a separate result group)
        is left to the caller (ChatbotChain) after intent.limit is applied.

        progress_callback(i, total, query) is called before each query runs.
        """
        raw_results = []
        for i, query in enumerate(queries):
            if progress_callback:
                progress_callback(i, len(queries), query)
            result = self._run_one(query)
            raw_results.append(result)

        raw_row_count = sum(r.row_count for r in raw_results if r.success)

        # Merge within each channel: sub-table variants sharing the same channel
        # tag are outer-joined into a single group per channel.
        results = self._merge_by_channel(raw_results)

        merged_group_count = sum(1 for r in results if r.success and r.rows)
        total_rows = sum(r.row_count for r in results if r.success)
        enumeration = self.build_programmatic_enumeration(results)

        return ExecutionResult(
            query_results=results,
            total_rows=total_rows,
            programmatic_enumeration=enumeration,
            raw_row_count=raw_row_count,
            merged_group_count=merged_group_count,
        )

    # ── Merge helper ─────────────────────────────────────────────────────────

    def _merge_across_channels(self, results: list[QueryResult]) -> list[QueryResult]:
        """
        Merge all channel groups into a single combined result.

        Only fires when there is exactly ONE successful group per distinct channel
        (i.e. the within-channel merge already reduced each channel to a single
        group).  This guarantees clean outer-join column naming.
        """
        good = [r for r in results if r.success and r.rows]
        failed = [r for r in results if not r.success or not r.rows]

        if len(good) <= 1:
            return results

        # Guard: require exactly one group per channel to avoid suffix collisions
        channels = [r.query.channel for r in good]
        if len(channels) != len(set(channels)):
            # More than one group shares a channel name — skip cross-channel merge
            return results

        combined = self._merger.merge(good)
        return [combined] + failed

    def _merge_by_channel(self, results: list[QueryResult]) -> list[QueryResult]:
        """
        Group results by (label, channel) and merge within each group.

        Tables sharing the same channel tag are outer-joined on shared dimension
        columns. When sub-table variants carry distinct metric suffixes (e.g.
        cymtd_val vs cymtd_vol), the merge produces clean side-by-side columns
        without collisions.

        The label is set when hierarchy expansion produces separate query groups
        per dimension level (e.g. "zsm", "rsm", "asm", "so" for a sales_person
        query). Unlabeled queries use an empty-string key and behave as before.
        """
        from collections import defaultdict

        groups: dict[tuple, list[QueryResult]] = defaultdict(list)
        for r in results:
            key = (r.query.label or "", r.query.channel)
            groups[key].append(r)

        merged: list[QueryResult] = []
        for (_label, _channel), group in groups.items():
            successful = [r for r in group if r.success and r.rows]
            if len(successful) > 1:
                merged.append(self._merger.merge(successful))
            else:
                merged.extend(group)

        return merged

    # ── Private ──────────────────────────────────────────────────────────────

    def _run_one(self, query: BuiltQuery) -> QueryResult:
        try:
            from sqlalchemy import text
            with self.db._engine.connect() as conn:
                cursor = conn.execute(text(query.sql))
                columns = list(cursor.keys())
                rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
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

    def build_programmatic_enumeration(self, results: list[QueryResult]) -> str:
        """
        Build a structured text enumeration of all result rows.
        This replaces raw markdown tables in the LLM prompt, guaranteeing
        every row is visible and blank cells are simply omitted.

        When results carry a label (hierarchy-expanded queries), they are
        grouped by label and each channel sub-section is nested under it.
        Unlabeled results are formatted with just the channel as the header.
        """
        from collections import defaultdict

        # Separate labeled (hierarchy-expanded) from unlabeled results
        labeled: dict[str, list[QueryResult]] = defaultdict(list)
        unlabeled: list[QueryResult] = []
        for r in results:
            if r.success and r.rows:
                if r.query.label:
                    labeled[r.query.label].append(r)
                else:
                    unlabeled.append(r)

        sections = []
        global_idx = 1

        # Labeled groups: one section per level (e.g. ZSM, RSM, ASM, SO)
        for level, level_results in labeled.items():
            total = sum(r.row_count for r in level_results)
            lines = [f"### {level.upper()} ({total} rows total)"]
            for r in level_results:
                lines.append(f"  [{r.query.channel.upper()}]")
                for row in r.rows:
                    parts = [f"{global_idx}."]
                    for col, val in row.items():
                        if val is not None and val != "":
                            parts.append(f"{col}: {val}")
                    lines.append("    " + " | ".join(parts))
                    global_idx += 1
            sections.append("\n".join(lines))

        # Unlabeled groups: standard channel header
        for result in unlabeled:
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
