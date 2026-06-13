#!/usr/bin/env python
"""
Drop All Indexes by Pattern

Simple script to drop all indexes matching a pattern.

Usage:
    python drop_indexes_by_pattern.py revenue_mgmt [--dry-run]
"""

import argparse
import logging
from sqlalchemy import create_engine, text
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Drop all indexes by pattern')
    parser.add_argument('pattern', help='Pattern to match (e.g., revenue_mgmt)')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be dropped without dropping')
    args = parser.parse_args()

    db_url = os.getenv('DATABASE_URL', 'postgresql://localhost:5432/vectordb')
    engine = create_engine(db_url)

    pattern = f"%{args.pattern}%"

    logger.info("="*60)
    logger.info(f"DROP INDEXES BY PATTERN")
    logger.info("="*60)
    logger.info(f"  Pattern: {pattern}")
    logger.info(f"  Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    logger.info("="*60)
    logger.info("")

    with engine.connect() as conn:
        # Find all indexes matching pattern
        indexes = conn.execute(text("""
            SELECT indexname, tablename
            FROM pg_indexes
            WHERE schemaname = 'public'
            AND (indexname LIKE :pattern OR tablename LIKE :pattern)
            ORDER BY indexname
        """), {"pattern": pattern}).fetchall()

        logger.info(f"Found {len(indexes)} indexes matching pattern")
        logger.info("")

        if not indexes:
            logger.info("✅ No indexes found!")
            return

        if args.dry_run:
            logger.info("Would drop the following indexes:")
            for indexname, tablename in indexes:
                logger.info(f"  - {indexname} (table: {tablename})")
            logger.info("")
            logger.info("Run without --dry-run to actually drop them")
            return

        # Drop indexes
        dropped = 0
        failed = 0
        for indexname, tablename in indexes:
            try:
                conn.execute(text(f"DROP INDEX IF EXISTS {indexname}"))
                logger.info(f"  ✅ Dropped: {indexname}")
                dropped += 1
            except Exception as e:
                logger.error(f"  ❌ Failed: {indexname} - {e}")
                failed += 1

        conn.commit()

        logger.info("")
        logger.info("="*60)
        logger.info(f"✅ Dropped: {dropped}")
        if failed > 0:
            logger.info(f"❌ Failed: {failed}")
        logger.info("="*60)


if __name__ == "__main__":
    main()
