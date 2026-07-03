"""
Pydantic models for query building output and execution results.
"""
from __future__ import annotations
from typing import Any, Optional
from pydantic import BaseModel, Field


class BuiltQuery(BaseModel):
    """A single SQL query produced by Stage 2 for one table."""
    sql: str
    table: str
    channel: str           # "primary" | "secondary" | "transactional" | etc.
    label: Optional[str] = None


class QueryResult(BaseModel):
    """Raw result returned by Stage 3 for one BuiltQuery."""
    query: BuiltQuery
    columns: list[str] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)
    row_count: int = 0
    error: Optional[str] = None
    success: bool = True
    col_types: dict[str, str] = Field(default_factory=dict)  # column → Trino type string


class ExecutionResult(BaseModel):
    """Aggregated results from all queries in a single request."""
    query_results: list[QueryResult] = Field(default_factory=list)
    total_rows: int = 0
    programmatic_enumeration: Optional[str] = None
    used_fallback: bool = False
    applied_rules: list[str] = Field(default_factory=list)
    raw_row_count: int = 0        # total rows across all queries before channel merge
    merged_group_count: int = 0   # number of result groups after merge


class ChatResponse(BaseModel):
    """Final response object returned by the engine to the caller."""
    question: str
    answer: str                        # full response (summary + analysis combined)
    summary: Optional[str] = None      # one-line QUICK SUMMARY extracted from answer
    detail: Optional[str] = None       # ANALYSIS section extracted from answer
    tabular: Optional[str] = None      # programmatic row enumeration (for display)
    success: bool = True
    error: Optional[str] = None
    sql_queries: list[dict] = Field(default_factory=list)
    total_rows: int = 0
    used_fallback: bool = False
    applied_rules: list[str] = Field(default_factory=list)
    # Step timing and result data for verbose CLI output
    step_timings: dict = Field(default_factory=dict)
    raw_query_count: int = 0
    raw_row_count: int = 0
    merged_group_count: int = 0
    query_results: list = Field(default_factory=list)   # [{channel, columns, rows, row_count, col_types}]
    vega_lite_spec: Optional[Any] = None                # Vega-Lite v5 spec dict (set when --chart)
