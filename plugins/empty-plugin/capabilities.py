"""
Plugin capability overrides — optional.

If your plugin needs to change how the engine builds certain SQL clauses,
subclass EngineCapabilities here and override the relevant methods.

The loader automatically detects this file, finds the first subclass of
EngineCapabilities, and injects it into the query builder.
If this file is absent or no subclass is found, the engine uses its defaults.

CURRENTLY OVERRIDABLE METHODS
------------------------------
build_date_filter(table, date_col, date_range, table_meta) -> str
    Called when the user specifies an explicit date range.
    Default: WHERE date_col BETWEEN 'start' AND 'end'
    Override when: your table stores periodic snapshots and you always want
    the latest snapshot within the range rather than all rows.
    Example: Haldiram — WHERE time_key = (SELECT MAX(time_key) FROM table
                                          WHERE time_key >= start AND <= end)

build_snapshot_filter(table, date_col, table_meta) -> str
    Called when no date range is given and date_mode = "snapshot" on the table.
    Default: WHERE date_col = (SELECT MAX(date_col) FROM table)
    Override when: you need a different "latest row" strategy.

ADDING NEW OVERRIDABLE ENGINE BEHAVIOURS
-----------------------------------------
1. Add a method with a sensible default to engine/capabilities/base.py
2. Call self.capabilities.<method>() at the appropriate point in the engine
3. Override the method here in your plugin's capabilities.py

USAGE
-----
Uncomment the class below, rename it, and override the methods you need.
Delete methods you do not need to override — they fall through to the base defaults.
"""
from engine.capabilities.base import EngineCapabilities


# class MyPluginCapabilities(EngineCapabilities):
#
#     def build_date_filter(self, table, date_col, date_range, table_meta=None):
#         """
#         Example: use MAX(date_col) within range for snapshot tables.
#         """
#         if table_meta and table_meta.date_mode == "snapshot":
#             return (
#                 f"{date_col} = ("
#                 f"SELECT MAX({date_col}) FROM {table} "
#                 f"WHERE {date_col} >= '{date_range.start}' "
#                 f"AND {date_col} <= '{date_range.end}'"
#                 f")"
#             )
#         return super().build_date_filter(table, date_col, date_range, table_meta)
