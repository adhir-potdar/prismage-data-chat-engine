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


class ExecutionResult(BaseModel):
    """Aggregated results from all queries in a single request."""
    query_results: list[QueryResult] = Field(default_factory=list)
    total_rows: int = 0
    programmatic_enumeration: Optional[str] = None
    used_fallback: bool = False
    applied_rules: list[str] = Field(default_factory=list)


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
