"""
DatabaseAdapter — wraps SQLAlchemy for connection management.
Returns a LangChain SQLDatabase instance usable by QuerySQLDataBaseTool.
"""
from __future__ import annotations
from langchain_community.utilities import SQLDatabase


def create_database(connection_string: str, include_tables: list[str] | None = None) -> SQLDatabase:
    """
    Create a LangChain SQLDatabase from a SQLAlchemy connection string.

    Examples:
        create_database("postgresql://user:pass@localhost/sales")
        create_database("sqlite:///data/retail.db")
        create_database("mysql+pymysql://user:pass@host/db")
    """
    kwargs = {}
    if include_tables:
        kwargs["include_tables"] = include_tables
    return SQLDatabase.from_uri(connection_string, **kwargs)
