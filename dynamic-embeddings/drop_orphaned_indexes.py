#!/usr/bin/env python
"""
Drop Orphaned Indexes Script

Drops all indexes that belong to non-existent tables. This cleans up
orphaned indexes left behind from failed table creations or schema changes.

Usage:
    python drop_orphaned_indexes.py [--namespace NAMESPACE] [--dry-run]
"""

import argparse
import logging
from sqlalchemy import create_engine, text
from dynamic_embeddings.database.connection import DatabaseConnection

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_orphaned_indexes(engine, namespace=None):
    """Find all indexes that belong to non-existent tables.

    Args:
        engine: SQLAlchemy engine
        namespace: Optional namespace filter (e.g., 'revenue_mgmt')

    Returns:
        List of tuples (indexname, tablename)
    """
    with engine.connect() as conn:
        # Find all indexes
        if namespace:
            pattern = f"%{namespace}%"
            query = text("""
                SELECT i.indexname, i.tablename
                FROM pg_indexes i
                WHERE i.indexname LIKE :pattern
                ORDER BY i.indexname
            """)
            indexes = conn.execute(query, {"pattern": pattern}).fetchall()
        else:
            query = text("""
                SELECT i.indexname, i.tablename
                FROM pg_indexes i
                WHERE i.schemaname = 'public'
                ORDER BY i.indexname
            """)
            indexes = conn.execute(query).fetchall()

        # Check which indexes belong to non-existent tables
        orphaned = []
        for indexname, tablename in indexes:
            table_exists = conn.execute(text(
                "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = :tablename)"
            ), {"tablename": tablename}).scalar()

            if not table_exists:
                orphaned.append((indexname, tablename))

        return orphaned


def drop_orphaned_indexes(engine, orphaned_indexes, dry_run=False):
    """Drop orphaned indexes.

    Args:
        engine: SQLAlchemy engine
        orphaned_indexes: List of (indexname, tablename) tuples
        dry_run: If True, only print what would be dropped
    """
    if not orphaned_indexes:
        logger.info("✅ No orphaned indexes found!")
        return

    logger.info(f"Found {len(orphaned_indexes)} orphaned indexes")

    if dry_run:
        logger.info("🔍 DRY RUN - Would drop the following indexes:")
        for indexname, tablename in orphaned_indexes:
            logger.info(f"   - {indexname} (table: {tablename})")
        return

    logger.info("🗑️  Dropping orphaned indexes...")
    dropped_count = 0
    failed_count = 0

    with engine.connect() as conn:
        for indexname, tablename in orphaned_indexes:
            try:
                conn.execute(text(f"DROP INDEX IF EXISTS {indexname}"))
                conn.commit()
                logger.info(f"   ✅ Dropped: {indexname}")
                dropped_count += 1
            except Exception as e:
                logger.error(f"   ❌ Failed to drop {indexname}: {e}")
                failed_count += 1

    logger.info(f"\n{'='*60}")
    logger.info(f"✅ Successfully dropped: {dropped_count} indexes")
    if failed_count > 0:
        logger.warning(f"⚠️  Failed to drop: {failed_count} indexes")
    logger.info(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description='Drop orphaned indexes for embeddings tables'
    )
    parser.add_argument(
        '--namespace',
        help='Filter indexes by namespace (e.g., revenue_mgmt)',
        default=None
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be dropped without actually dropping'
    )
    args = parser.parse_args()

    logger.info("🔍 ORPHANED INDEX CLEANUP")
    logger.info("="*60)

    if args.namespace:
        logger.info(f"   Namespace filter: {args.namespace}")
    else:
        logger.info("   Namespace filter: ALL")

    if args.dry_run:
        logger.info("   Mode: DRY RUN (no changes will be made)")
    else:
        logger.info("   Mode: LIVE (indexes will be dropped)")

    logger.info("="*60)
    logger.info("")

    # Get database connection
    db_conn = DatabaseConnection()

    # Find orphaned indexes
    logger.info("🔍 Scanning for orphaned indexes...")
    orphaned = find_orphaned_indexes(db_conn.engine, namespace=args.namespace)

    # Drop orphaned indexes
    drop_orphaned_indexes(db_conn.engine, orphaned, dry_run=args.dry_run)

    if args.dry_run:
        logger.info("\n💡 Run without --dry-run to actually drop the indexes")


if __name__ == "__main__":
    main()
