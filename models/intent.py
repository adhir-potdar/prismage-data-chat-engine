"""
Pydantic models for the parsed question intent produced by Stage 1
and enriched by the Business Rules Engine before Stage 2.
"""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class HavingCondition(BaseModel):
    metric1: str
    operator: str  # "<" | ">" | "<=" | ">="
    metric2: str


class HavingConfig(BaseModel):
    type: str          # "vs_average" | "gap_to_target" | "metric_comparison"
    polarity: Optional[str] = None   # "positive" | "negative" | None
    conditions: list[HavingCondition] = Field(default_factory=list)


class DateRange(BaseModel):
    start: str   # "YYYYMMDD"
    end: str     # "YYYYMMDD"


class SortConfig(BaseModel):
    metric: str
    direction: str = "DESC"   # "ASC" | "DESC"


class OutputHint(BaseModel):
    format: str = "narrative"         # "narrative" | "table" | "comparison_table" | "side_by_side_table"
    always_programmatic_enumeration: bool = True
    columns: list[str] = Field(default_factory=list)
    primary_columns: list[str] = Field(default_factory=list)
    secondary_columns: list[str] = Field(default_factory=list)


class ParsedIntent(BaseModel):
    """
    Structured output of Stage 1 (question parser).
    Enriched by the Business Rules Engine before entering Stage 2.
    """
    # Core intent from Stage 1
    dimensions: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    formula_metrics: list[str] = Field(default_factory=list)
    having: Optional[HavingConfig] = None
    filters: dict[str, str] = Field(default_factory=dict)
    date_range: Optional[DateRange] = None
    sort: Optional[SortConfig] = None
    limit: Optional[int] = None
    confidence: float = 1.0

    # Enriched by Business Rules Engine
    channels: list[str] = Field(default_factory=list)
    output_hints: OutputHint = Field(default_factory=OutputHint)
    applied_rules: list[str] = Field(default_factory=list)
    inline_formulas: list[dict] = Field(default_factory=list)
    formula_overrides: dict[str, dict] = Field(default_factory=dict)
