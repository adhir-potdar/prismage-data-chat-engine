#!/usr/bin/env python3
"""
Delete Namespace Table

Usage:
    python delete_namespace.py <namespace_name>
    python delete_namespace.py revenue_mgmt
    python delete_namespace.py backup_revenue_mgmt
"""

import sys
from dynamic_embeddings.database.connection import DatabaseConnection
from sqlalchemy import text


def delete_namespace(namespace: str, force: bool = False):
    """Delete a namespace table."""

    table_name = f'embeddings_{namespace}'

    print(f"🗑️  DELETE NAMESPACE")
    print("=" * 60)
    print(f"   Namespace: {namespace}")
    print(f"   Table: {table_name}")
    print()

    db = DatabaseConnection()

    # Check if table exists
    with db.get_session() as session:
        result = session.execute(text("""
            SELECT EXISTS(
                SELECT 1 FROM information_schema.tables
                WHERE table_name = :table_name
                AND table_schema = 'public'
            )
        """), {"table_name": table_name})

        exists = result.scalar()

        if not exists:
            print(f"❌ Table '{table_name}' does not exist")
            return False

        # Get row count
        count_result = session.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
        count = count_result.scalar()
        print(f"   Current rows: {count:,}")

    # Confirm deletion
    if not force:
        print()
        print(f"⚠️  WARNING: This will permanently delete the table and all {count:,} embeddings!")
        response = input(f"   Type 'yes' to confirm deletion: ").strip().lower()

        if response != 'yes':
            print("❌ Deletion cancelled")
            return False

    # Delete the table
    try:
        with db.get_session() as session:
            session.execute(text(f"DROP TABLE {table_name} CASCADE"))
            session.commit()
            print()
            print(f"✅ Successfully deleted table: {table_name}")
            return True
    except Exception as e:
        print()
        print(f"❌ Error deleting table: {e}")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python delete_namespace.py <namespace_name> [--force]")
        print()
        print("Examples:")
        print("  python delete_namespace.py revenue_mgmt")
        print("  python delete_namespace.py backup_revenue_mgmt --force")
        sys.exit(1)

    namespace = sys.argv[1]
    force = '--force' in sys.argv or '-f' in sys.argv

    success = delete_namespace(namespace, force)
    sys.exit(0 if success else 1)
