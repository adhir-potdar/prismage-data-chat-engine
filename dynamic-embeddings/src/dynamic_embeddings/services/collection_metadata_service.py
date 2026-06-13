"""
Collection Metadata Service

Fast collection lookup using pre-built metadata table.
"""

from typing import List, Optional, Dict
from sqlalchemy import and_, or_
from datetime import datetime


class CollectionMetadataService:
    """Service for querying collection metadata."""

    def __init__(self, db_connection, namespace: str = 'default'):
        self.db_connection = db_connection
        self.namespace = namespace

        # Get namespace-specific metadata model
        from dynamic_embeddings.database.schema import get_collection_metadata_model
        self.MetadataModel = get_collection_metadata_model(namespace)

    def find_collections(
        self,
        dimension: str,
        time_granularity: Optional[str] = None,
        start_date: Optional[int] = None,
        end_date: Optional[int] = None,
        dimension_values: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Find collections matching criteria.

        Args:
            dimension: Dimension to search (e.g., 'property_geo', 'overall')
            time_granularity: Time granularity filter (e.g., 'mom', 'qoq')
            start_date: Filter collections overlapping this start date (YYYYMMDD)
            end_date: Filter collections overlapping this end date (YYYYMMDD)
            dimension_values: Filter collections containing these dimension values (e.g., ['APP', 'AMP'])

        Returns:
            List of matching collection metadata dicts
        """
        with self.db_connection.get_session() as session:
            query = session.query(self.MetadataModel)

            # Filter by dimension
            query = query.filter(self.MetadataModel.dimension == dimension)

            # Filter by time granularity (if specified)
            if time_granularity:
                query = query.filter(
                    self.MetadataModel.time_granularity == time_granularity.lower()
                )

            # Filter by date range (if specified)
            # User provides single date range, match against both collection periods
            if start_date and end_date:
                # Find collections where periods overlap with query range
                # Collection overlaps if:
                #   (collection_period1_start <= query_end) AND (collection_period1_end >= query_start)
                # OR
                #   (collection_period2_start <= query_end) AND (collection_period2_end >= query_start)

                query = query.filter(
                    or_(
                        and_(
                            self.MetadataModel.period1_start_date <= end_date,
                            self.MetadataModel.period1_end_date >= start_date
                        ),
                        and_(
                            self.MetadataModel.period2_start_date <= end_date,
                            self.MetadataModel.period2_end_date >= start_date
                        )
                    )
                )

            # Filter by dimension_values (if specified)
            # Only include collections that contain at least one of the requested dimension values
            if dimension_values:
                # Use JSONB ?| operator to check if array overlaps with provided values
                # The ?| operator expects a PostgreSQL text array on the right side
                from sqlalchemy import cast, func
                from sqlalchemy.dialects.postgresql import ARRAY
                from sqlalchemy.types import String as SQLString

                # Cast Python list to PostgreSQL ARRAY
                # dimension_values ?| ARRAY['APP', 'AMP']::text[]
                query = query.filter(
                    self.MetadataModel.dimension_values.op('?|')(
                        cast(dimension_values, ARRAY(SQLString))
                    )
                )

            # Execute query
            results = query.all()

            # Convert to dicts
            return [
                {
                    'collection_name': r.collection_name,
                    'dimension': r.dimension,
                    'time_granularity': r.time_granularity,
                    'dimension_values': r.dimension_values if hasattr(r, 'dimension_values') else None,
                    'period1_start_date': r.period1_start_date,
                    'period1_end_date': r.period1_end_date,
                    'period2_start_date': r.period2_start_date,
                    'period2_end_date': r.period2_end_date,
                    'total_embeddings': r.total_embeddings
                }
                for r in results
            ]

    def get_all_dimensions(self) -> List[str]:
        """Get list of all available dimensions."""
        with self.db_connection.get_session() as session:
            dimensions = session.query(
                self.MetadataModel.dimension
            ).distinct().all()

            return [d[0] for d in dimensions]

    def get_all_granularities(self) -> List[str]:
        """Get list of all available time granularities."""
        with self.db_connection.get_session() as session:
            granularities = session.query(
                self.MetadataModel.time_granularity
            ).distinct().all()

            return [g[0] for g in granularities]

    def get_date_range_for_dimension(
        self,
        dimension: str,
        time_granularity: Optional[str] = None
    ) -> Dict[str, int]:
        """Get min/max date range for a dimension."""
        from sqlalchemy import func

        with self.db_connection.get_session() as session:
            query = session.query(
                func.min(self.MetadataModel.period1_start_date),
                func.max(self.MetadataModel.period2_end_date)
            ).filter(self.MetadataModel.dimension == dimension)

            if time_granularity:
                query = query.filter(
                    self.MetadataModel.time_granularity == time_granularity.lower()
                )

            result = query.first()

            if result and result[0] and result[1]:
                return {
                    'min_date': result[0],
                    'max_date': result[1]
                }

            return {'min_date': None, 'max_date': None}
