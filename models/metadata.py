"""
Pydantic models for metadata loaded from JSON config files.
"""
from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class MetricCategory(str, Enum):
    ABSOLUTE = "absolute"
    AVERAGE = "average"
    CUMULATIVE = "cumulative"
    PERCENTAGE = "percentage"
    FORMULA = "formula"


class AggregateFunction(str, Enum):
    SUM = "SUM"
    AVG = "AVG"
    COUNT = "COUNT"


class Dimension(BaseModel):
    name: str
    aliases: list[str] = Field(default_factory=list)
    db_column: str
    table_affinity: list[str] = Field(default_factory=list)
    hierarchy_name: Optional[str] = None
    hierarchy_level: Optional[int] = None


class Metric(BaseModel):
    name: str
    aliases: list[str] = Field(default_factory=list)
    db_column: Optional[str] = None
    aggregate_fn: Optional[AggregateFunction] = None
    category: MetricCategory
    formula_ref: Optional[str] = None
    table_affinity: list[str] = Field(default_factory=list)


class Formula(BaseModel):
    name: str
    display: str
    expression: str
    components: list[str] = Field(default_factory=list)
    runtime_vars: list[str] = Field(default_factory=list)
    window: bool = False
    notes: Optional[str] = None


class Table(BaseModel):
    name: str
    channel: str
    date_column: Optional[str] = None
    description: Optional[str] = None
    dimensions: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)


class MetricHint(BaseModel):
    phrase: str
    maps_to_having: Optional[dict] = None
    polarity: Optional[str] = None
    operator: Optional[str] = None


class HavingPatternDef(BaseModel):
    type: str
    description: str
    sql_template: str
    multi_condition_join: str = "AND"


class RuleTrigger(BaseModel):
    type: str
    phrases: list[str] = Field(default_factory=list)
    having_type: Optional[str] = None
    polarity: Optional[str] = None
    formula: Optional[str] = None
    metrics: list[str] = Field(default_factory=list)
    category: Optional[str] = None
    conditions: list[dict] = Field(default_factory=list)


class RuleAction(BaseModel):
    type: str
    metrics: list[str] = Field(default_factory=list)
    dimensions: list[str] = Field(default_factory=list)
    formula: Optional[str] = None
    name: Optional[str] = None
    expression: Optional[str] = None
    components: list[str] = Field(default_factory=list)
    label: Optional[str] = None
    format: Optional[str] = None
    channels: list[str] = Field(default_factory=list)
    metric: Optional[str] = None
    direction: Optional[str] = None
    always_programmatic_enumeration: bool = False
    primary_columns: list[str] = Field(default_factory=list)
    secondary_columns: list[str] = Field(default_factory=list)
    columns: list[str] = Field(default_factory=list)


class BusinessRule(BaseModel):
    name: str
    description: Optional[str] = None
    trigger: RuleTrigger
    actions: list[RuleAction] = Field(default_factory=list)
