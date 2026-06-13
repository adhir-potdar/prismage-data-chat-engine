"""Namespace migration utilities for embeddings."""

import logging
from typing import Dict, Any, Optional
from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


class NamespaceMigration:
    """Handles migration between namespaces and legacy table support."""

    def __init__(self, database_url: str):
        """Initialize migration manager.

        Args:
            database_url: PostgreSQL connection URL
        """
        self.database_url = database_url
        self.engine = create_engine(database_url, echo=False)
        self.logger = logging.getLogger(__name__)

    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists.

        Args:
            table_name: Name of the table

        Returns:
            True if table exists
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT EXISTS(
                        SELECT 1 FROM information_schema.tables
                        WHERE table_name = :table_name
                        AND table_schema = 'public'
                    )
                """), {"table_name": table_name})
                return bool(result.scalar())
        except Exception as e:
            self.logger.error(f"Failed to check if table {table_name} exists: {e}")
            return False

    def get_table_row_count(self, table_name: str) -> int:
        """Get row count for a table.

        Args:
            table_name: Name of the table

        Returns:
            Number of rows
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                return result.scalar()
        except Exception as e:
            self.logger.error(f"Failed to get row count for {table_name}: {e}")
            return 0

    def migrate_namespace(
        self,
        source_namespace: str,
        target_namespace: str,
        mode: str = 'copy'
    ) -> Dict[str, Any]:
        """Migrate data from source namespace to target namespace.

        Args:
            source_namespace: Source table name or namespace (e.g., 'embeddings' or 'prod')
            target_namespace: Target namespace (e.g., 'default' or 'staging')
            mode: 'copy' (keep source) or 'move' (delete source)

        Returns:
            Migration report dictionary
        """
        try:
            # Determine source table name
            if source_namespace == 'embeddings':
                # Legacy table
                source_table = 'embeddings'
            else:
                source_table = f'embeddings_{source_namespace}'

            target_table = f'embeddings_{target_namespace}'

            self.logger.info(f"Starting migration: {source_table} -> {target_table} (mode: {mode})")

            # Validation
            if not self.table_exists(source_table):
                error_msg = f"Source table '{source_table}' does not exist"
                self.logger.error(error_msg)
                return {
                    'success': False,
                    'error': error_msg
                }

            if self.table_exists(target_table):
                error_msg = f"Target table '{target_table}' already exists"
                self.logger.error(error_msg)
                return {
                    'success': False,
                    'error': error_msg
                }

            if mode not in ['copy', 'move']:
                error_msg = f"Invalid mode '{mode}'. Must be 'copy' or 'move'"
                self.logger.error(error_msg)
                return {
                    'success': False,
                    'error': error_msg
                }

            # Get source row count
            source_count = self.get_table_row_count(source_table)
            self.logger.info(f"Source table has {source_count} rows")

            start_time = datetime.now()

            with self.engine.begin() as conn:
                # Create target table with same structure
                self.logger.info(f"Creating target table '{target_table}'...")
                conn.execute(text(f"""
                    CREATE TABLE {target_table} (LIKE {source_table} INCLUDING ALL)
                """))

                # Copy all data
                self.logger.info("Copying data...")
                conn.execute(text(f"""
                    INSERT INTO {target_table}
                    SELECT * FROM {source_table}
                """))

                # Verify row counts match (within transaction)
                target_count_result = conn.execute(text(f"SELECT COUNT(*) FROM {target_table}"))
                target_count = target_count_result.scalar()
                if source_count != target_count:
                    raise Exception(f"Row count mismatch: source={source_count}, target={target_count}")

                # Handle source table based on mode
                if mode == 'move':
                    backup_name = f"{source_table}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    self.logger.info(f"Renaming source table to '{backup_name}'...")
                    conn.execute(text(f"ALTER TABLE {source_table} RENAME TO {backup_name}"))
                    source_action = f"renamed to {backup_name}"
                else:
                    source_action = "kept unchanged"

            # Recreate vector index for target (outside transaction - optional operation)
            self.logger.info("Creating vector index for target table...")
            index_name = f"{target_table}_vector_idx"
            try:
                with self.engine.begin() as conn:
                    conn.execute(text(f"""
                        CREATE INDEX {index_name} ON {target_table}
                        USING ivfflat (embedding vector_cosine_ops)
                        WITH (lists = 100)
                    """))
                self.logger.info(f"Vector index created successfully")
            except Exception as idx_error:
                self.logger.warning(f"Failed to create vector index (optional): {idx_error}")
                self.logger.info("Migration succeeded without vector index - you can create it manually later")

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            result = {
                'success': True,
                'source_table': source_table,
                'target_table': target_table,
                'mode': mode,
                'rows_migrated': target_count,
                'duration_seconds': duration,
                'source_action': source_action,
                'timestamp': end_time.isoformat()
            }

            self.logger.info(f"Migration completed successfully: {result}")
            return result

        except Exception as e:
            self.logger.error(f"Migration failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'source_table': source_table if 'source_table' in locals() else source_namespace,
                'target_table': target_table if 'target_table' in locals() else target_namespace
            }

    def migrate_legacy_table(self, target_namespace: str = 'default') -> Dict[str, Any]:
        """Migrate legacy 'embeddings' table to namespace-based table.

        Args:
            target_namespace: Target namespace (default: 'default')

        Returns:
            Migration report dictionary
        """
        self.logger.info(f"Migrating legacy 'embeddings' table to 'embeddings_{target_namespace}'")
        return self.migrate_namespace('embeddings', target_namespace, mode='move')

    def copy_namespace(self, source: str, target: str) -> Dict[str, Any]:
        """Copy namespace data to another namespace.

        Args:
            source: Source namespace
            target: Target namespace

        Returns:
            Migration report dictionary
        """
        return self.migrate_namespace(source, target, mode='copy')

    def move_namespace(self, source: str, target: str) -> Dict[str, Any]:
        """Move namespace data to another namespace.

        Args:
            source: Source namespace
            target: Target namespace

        Returns:
            Migration report dictionary
        """
        return self.migrate_namespace(source, target, mode='move')

    def check_legacy_migration_needed(self) -> bool:
        """Check if legacy migration is needed.

        Returns:
            True if 'embeddings' table exists and 'embeddings_default' doesn't
        """
        has_legacy = self.table_exists('embeddings')
        has_default = self.table_exists('embeddings_default')

        return has_legacy and not has_default

    def auto_migrate_if_needed(self) -> Optional[Dict[str, Any]]:
        """Automatically migrate legacy table if needed.

        Returns:
            Migration report if migration was performed, None otherwise
        """
        if self.check_legacy_migration_needed():
            self.logger.info("Legacy migration needed - starting automatic migration")
            return self.migrate_legacy_table()
        else:
            self.logger.info("No legacy migration needed")
            return None
