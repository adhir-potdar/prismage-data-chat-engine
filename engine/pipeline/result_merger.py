"""
ResultMerger — merges multiple QueryResult objects via full-outer join on shared
dimension columns. Used when the same question spans multiple DB tables (e.g.
primary and secondary channels, or sales_rep and product tables).

Generic — zero domain-specific logic. Join key is determined by inspecting
which columns appear in ALL result sets.
"""
from __future__ import annotations
import logging
from models.query import QueryResult

logger = logging.getLogger(__name__)


class ResultMerger:
    """
    Merges a list of QueryResult objects into a single merged QueryResult.

    Algorithm:
    1. Find columns that appear in ALL successful result sets (join keys).
    2. Build a lookup dict per result: join_key_tuple → row.
    3. Full-outer join: iterate all unique join keys across all results;
       for each key, assemble one merged row with columns from all results.
       Columns from later results get a suffix based on their channel label
       if they would collide with columns from an earlier result.
    4. Return a synthetic QueryResult with channel="merged".
    """

    def merge(self, results: list[QueryResult]) -> QueryResult:
        successful = [r for r in results if r.success and r.rows]

        if not successful:
            return QueryResult(
                query=results[0].query if results else _empty_query(),
                success=False,
                error="No successful query results to merge.",
            )

        if len(successful) == 1:
            return successful[0]

        # Identify dimension columns common across all result sets (join keys)
        join_keys = self._find_join_keys(successful)

        if not join_keys:
            # No common columns — concatenate instead of join
            logger.warning("No common columns found for merge; concatenating results.")
            return self._concatenate(successful)

        return self._outer_join(successful, join_keys)

    # ── Private ──────────────────────────────────────────────────────────────

    def _find_join_keys(self, results: list[QueryResult]) -> list[str]:
        """Return columns that appear in EVERY result set (potential join keys)."""
        col_sets = [set(r.columns) for r in results]
        common = col_sets[0]
        for s in col_sets[1:]:
            common = common & s
        # Metric columns are aggregated — only use dimension-like columns as keys.
        # Heuristic: keep columns whose values are strings in the first result.
        def _is_dimension_col(col: str, rows: list[dict]) -> bool:
            if not rows:
                return True
            sample = rows[0].get(col)
            return isinstance(sample, str) or sample is None

        return [c for c in sorted(common)
                if _is_dimension_col(c, results[0].rows)]

    def _outer_join(self, results: list[QueryResult], join_keys: list[str]) -> QueryResult:
        """Full outer join on join_keys; metric columns get channel-based suffixes on conflict."""
        # Build per-result lookup: key_tuple → row
        lookups: list[dict[tuple, dict]] = []
        for r in results:
            lk: dict[tuple, dict] = {}
            for row in r.rows:
                key = tuple(row.get(k) for k in join_keys)
                lk[key] = row
            lookups.append(lk)

        # Collect all unique keys (full outer)
        all_keys: set[tuple] = set()
        for lk in lookups:
            all_keys |= lk.keys()

        # Determine output column order and suffixing strategy
        merged_columns = list(join_keys)
        seen_cols: set[str] = set(join_keys)
        col_aliases: list[dict[str, str]] = []  # per-result: original_col → output_col

        for idx, r in enumerate(results):
            suffix = f"_{r.query.channel}" if r.query.channel != "unknown" else f"_{idx}"
            alias_map: dict[str, str] = {}
            for col in r.columns:
                if col in join_keys:
                    alias_map[col] = col
                    continue
                if col not in seen_cols:
                    alias_map[col] = col
                    seen_cols.add(col)
                    merged_columns.append(col)
                else:
                    out_col = f"{col}{suffix}"
                    alias_map[col] = out_col
                    if out_col not in seen_cols:
                        seen_cols.add(out_col)
                        merged_columns.append(out_col)
            col_aliases.append(alias_map)

        # Build merged rows
        merged_rows: list[dict] = []
        for key in sorted(all_keys, key=lambda k: [str(x) for x in k]):
            merged_row: dict = {}
            # Fill join key columns
            for jk, jv in zip(join_keys, key):
                merged_row[jk] = jv
            # Fill metric columns from each result
            for idx, lk in enumerate(lookups):
                row = lk.get(key, {})
                alias_map = col_aliases[idx]
                for orig_col, out_col in alias_map.items():
                    if orig_col in join_keys:
                        continue
                    merged_row[out_col] = row.get(orig_col)
            merged_rows.append(merged_row)

        from models.query import BuiltQuery
        merged_query = BuiltQuery(sql="[merged]", table="merged", channel="merged")
        return QueryResult(
            query=merged_query,
            columns=merged_columns,
            rows=merged_rows,
            row_count=len(merged_rows),
            success=True,
        )

    def _concatenate(self, results: list[QueryResult]) -> QueryResult:
        """Fallback: stack all rows from all results when no join keys exist."""
        all_cols: list[str] = []
        seen: set[str] = set()
        for r in results:
            for c in r.columns:
                if c not in seen:
                    all_cols.append(c)
                    seen.add(c)

        all_rows = []
        for r in results:
            all_rows.extend(r.rows)

        from models.query import BuiltQuery
        merged_query = BuiltQuery(sql="[concatenated]", table="concatenated", channel="merged")
        return QueryResult(
            query=merged_query,
            columns=all_cols,
            rows=all_rows,
            row_count=len(all_rows),
            success=True,
        )


def _empty_query():
    from models.query import BuiltQuery
    return BuiltQuery(sql="", table="", channel="")
