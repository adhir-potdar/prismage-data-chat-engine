"""Engine configuration for dynamic chunking behavior."""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, validator


class ChunkingConfig(BaseModel):
    """Configuration for chunking behavior."""

    max_chunk_size_mb: float = Field(default=1.0, description="Maximum chunk size in MB")
    min_chunk_size_bytes: int = Field(default=100, description="Minimum chunk size in bytes")
    overlap_percentage: float = Field(default=0.1, description="Overlap between chunks (0.0-0.5)")
    enable_semantic_splitting: bool = Field(default=True, description="Enable semantic boundary detection")
    preserve_structure: bool = Field(default=True, description="Preserve JSON structure in metadata")

    @validator("overlap_percentage")
    def validate_overlap(cls, v: float) -> float:
        """Validate overlap percentage."""
        if not 0.0 <= v <= 0.5:
            raise ValueError("Overlap percentage must be between 0.0 and 0.5")
        return v


class StrategyConfig(BaseModel):
    """Configuration for strategy selection."""

    enable_auto_selection: bool = Field(default=True, description="Enable automatic strategy selection")
    fallback_strategy: str = Field(default="hierarchical_chunking", description="Fallback strategy name")
    confidence_threshold: float = Field(default=0.7, description="Minimum confidence for strategy selection")
    max_analysis_time_seconds: int = Field(default=30, description="Maximum time for analysis")
    enable_hybrid_strategies: bool = Field(default=True, description="Enable hybrid strategy combinations")

    @validator("confidence_threshold")
    def validate_confidence(cls, v: float) -> float:
        """Validate confidence threshold."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
        return v


class EmbeddingConfig(BaseModel):
    """Configuration for embedding generation."""

    model_name: str = Field(default="text-embedding-3-large", description="Embedding model name")
    dimensions: Optional[int] = Field(default=None, description="Embedding dimensions (if supported)")
    batch_size: int = Field(default=100, description="Batch size for embedding generation")
    max_retries: int = Field(default=3, description="Maximum retries for API calls")
    timeout_seconds: int = Field(default=60, description="API timeout in seconds")
    rate_limit_rpm: int = Field(default=3500, description="Rate limit requests per minute")

    @validator("batch_size")
    def validate_batch_size(cls, v: int) -> int:
        """Validate batch size."""
        if not 1 <= v <= 2048:
            raise ValueError("Batch size must be between 1 and 2048")
        return v


class EngineConfig(BaseModel):
    """Main configuration for the dynamic chunking engine."""

    # Core configurations
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)

    # Processing configuration
    enable_parallel_processing: bool = Field(default=True, description="Enable parallel processing")
    max_workers: int = Field(default=4, description="Maximum worker threads")
    enable_streaming: bool = Field(default=True, description="Enable streaming for large files")
    stream_chunk_size_mb: float = Field(default=10.0, description="Chunk size for streaming")

    # Analysis configuration
    analysis_rules: List[str] = Field(
        default=["size_based", "hierarchy_depth", "performance_metrics", "adtech_analytics"],
        description="Enabled analysis rules"
    )
    custom_analyzers: Dict[str, dict] = Field(
        default_factory=dict,
        description="Custom analyzer configurations"
    )

    # Caching configuration
    enable_caching: bool = Field(default=True, description="Enable result caching")
    cache_ttl_seconds: int = Field(default=3600, description="Cache TTL in seconds")
    cache_max_size: int = Field(default=1000, description="Maximum cache entries")

    # Monitoring configuration
    enable_metrics: bool = Field(default=True, description="Enable performance metrics")
    log_strategy_decisions: bool = Field(default=True, description="Log strategy decisions")
    enable_profiling: bool = Field(default=False, description="Enable performance profiling")

    @validator("max_workers")
    def validate_max_workers(cls, v: int) -> int:
        """Validate max workers."""
        if not 1 <= v <= 32:
            raise ValueError("Max workers must be between 1 and 32")
        return v

    @classmethod
    def from_adtech_analytics(cls) -> "EngineConfig":
        """Create optimized config for ad-tech analytics."""
        return cls(
            chunking=ChunkingConfig(
                max_chunk_size_mb=2.0,
                enable_semantic_splitting=True,
                preserve_structure=True
            ),
            strategy=StrategyConfig(
                enable_auto_selection=True,
                enable_hybrid_strategies=True,
                fallback_strategy="performance_semantic"
            ),
            analysis_rules=[
                "size_based",
                "hierarchy_depth",
                "performance_metrics",
                "adtech_analytics",
                "drill_down_pattern"
            ]
        )

    @classmethod
    def from_configuration_files(cls) -> "EngineConfig":
        """Create optimized config for configuration file processing."""
        return cls(
            chunking=ChunkingConfig(
                max_chunk_size_mb=0.5,
                enable_semantic_splitting=True,
                preserve_structure=True
            ),
            strategy=StrategyConfig(
                fallback_strategy="hierarchical_chunking",
                enable_hybrid_strategies=False
            ),
            analysis_rules=["size_based", "hierarchy_depth", "configuration_pattern"]
        )