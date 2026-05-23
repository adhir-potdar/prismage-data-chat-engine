"""
DatabaseAdapter — wraps SQLAlchemy for connection management.
Returns a LangChain SQLDatabase instance usable by QuerySQLDataBaseTool.
"""
from __future__ import annotations
import warnings
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
    # Suppress SAWarning from pgvector columns — SQLAlchemy does not recognise
    # the 'vector' type but it is harmless for our query-only usage.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Did not recognize type 'vector'")
        return SQLDatabase.from_uri(connection_string, **kwargs)
