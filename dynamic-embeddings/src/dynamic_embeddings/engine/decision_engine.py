"""Decision engine for automatically selecting optimal chunking strategies."""

from typing import Dict, Any, List, Tuple, Optional
from enum import Enum
import json
import logging
from pathlib import Path

from ..analyzers.content import ContentAnalyzer
from ..analyzers.structure import StructureAnalyzer
from ..config.analyzer_config import AnalyzerConfig


class ChunkingStrategy(Enum):
    """Available chunking strategies."""
    FLAT = "flat"
    HIERARCHICAL = "hierarchical"
    SEMANTIC = "semantic"
    DIMENSIONAL = "dimensional"
    HYBRID = "hybrid"


class DecisionEngine:
    """Engine for deciding optimal chunking strategy based on JSON analysis."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize decision engine with configuration.

        Args:
            config: Configuration dictionary. If None, uses default config.
        """
        self.config = config or AnalyzerConfig.get_default_config()
        self.content_analyzer = ContentAnalyzer(self.config)
        self.structure_analyzer = StructureAnalyzer()
        self.logger = logging.getLogger(__name__)

        # Decision thresholds and weights with defaults
        default_thresholds = {
            'max_flat_depth': 3,
            'max_flat_keys': 20,
            'min_hierarchical_depth': 4,
            'semantic_content_ratio': 0.3,
            'dimensional_array_ratio': 0.4,
            'large_document_size': 1000000  # 1MB in characters
        }

        default_weights = {
            'structure_weight': 0.4,
            'content_weight': 0.3,
            'size_weight': 0.2,
            'performance_weight': 0.1
        }

        # Merge config with defaults
        config_thresholds = self.config.get('decision_thresholds', {})
        config_weights = self.config.get('strategy_weights', {})

        self.decision_thresholds = {**default_thresholds, **config_thresholds}
        self.strategy_weights = {**default_weights, **config_weights}

    def decide_strategy(self, json_data: Dict[str, Any]) -> Tuple[ChunkingStrategy, Dict[str, Any]]:
        """Decide optimal chunking strategy for given JSON data.

        Args:
            json_data: JSON data to analyze

        Returns:
            Tuple of (chosen_strategy, decision_details)
        """
        self.logger.info("Starting strategy decision process")

        # Analyze structure
        structure_analysis = self.structure_analyzer.analyze(json_data)

        # Analyze content
        content_analysis = self.content_analyzer.analyze(json_data)

        # Calculate decision factors
        decision_factors = self._calculate_decision_factors(
            json_data, structure_analysis, content_analysis
        )

        # Apply decision rules
        strategy, confidence = self._apply_decision_rules(decision_factors)

        decision_details = {
            'chosen_strategy': strategy.value,
            'confidence': confidence,
            'decision_factors': decision_factors,
            'structure_analysis': structure_analysis.__dict__,
            'content_analysis': content_analysis.__dict__,
            'reasoning': self._generate_reasoning(strategy, decision_factors)
        }

        self.logger.info(f"Selected strategy: {strategy.value} with confidence: {confidence:.2f}")

        return strategy, decision_details

    def _calculate_decision_factors(
        self,
        json_data: Dict[str, Any],
        structure_analysis: Any,
        content_analysis: Any
    ) -> Dict[str, float]:
        """Calculate numerical factors for decision making."""

        # Size factors
        json_str = json.dumps(json_data)
        document_size = len(json_str)
        size_score = min(document_size / self.decision_thresholds['large_document_size'], 1.0)

        # Structure factors
        depth_score = min(structure_analysis.max_depth / 10.0, 1.0)
        complexity_score = min(structure_analysis.total_keys / 100.0, 1.0)
        hierarchy_score = 1.0 if structure_analysis.max_depth >= self.decision_thresholds['min_hierarchical_depth'] else 0.3

        # Array/dimensional factors
        array_ratio = structure_analysis.array_count / max(structure_analysis.total_keys, 1)
        dimensional_score = 1.0 if array_ratio >= self.decision_thresholds['dimensional_array_ratio'] else array_ratio

        # Content factors
        total_content = len(content_analysis.performance_metrics) + len(content_analysis.reasoning_content)
        content_density = total_content / max(structure_analysis.total_keys, 1)
        semantic_score = min(content_density, 1.0)

        # Domain specificity
        domain_confidence = 1.0 if content_analysis.domain_type != 'general' else 0.3

        # Performance factors
        flat_suitability = 1.0 if (
            structure_analysis.max_depth <= self.decision_thresholds['max_flat_depth'] and
            structure_analysis.total_keys <= self.decision_thresholds['max_flat_keys']
        ) else 0.2

        return {
            'size_score': size_score,
            'depth_score': depth_score,
            'complexity_score': complexity_score,
            'hierarchy_score': hierarchy_score,
            'dimensional_score': dimensional_score,
            'semantic_score': semantic_score,
            'domain_confidence': domain_confidence,
            'flat_suitability': flat_suitability,
            'array_ratio': array_ratio,
            'content_density': content_density,
            'document_size': document_size
        }

    def _apply_decision_rules(self, factors: Dict[str, float]) -> Tuple[ChunkingStrategy, float]:
        """Apply decision rules to select strategy."""

        strategy_scores = {}

        # Flat strategy scoring
        flat_score = (
            factors['flat_suitability'] * 0.6 +
            (1 - factors['complexity_score']) * 0.2 +
            (1 - factors['size_score']) * 0.2
        )
        strategy_scores[ChunkingStrategy.FLAT] = flat_score

        # Hierarchical strategy scoring
        hierarchical_score = (
            factors['hierarchy_score'] * 0.4 +
            factors['complexity_score'] * 0.3 +
            factors['depth_score'] * 0.3
        )
        strategy_scores[ChunkingStrategy.HIERARCHICAL] = hierarchical_score

        # Semantic strategy scoring
        semantic_score = (
            factors['semantic_score'] * 0.5 +
            factors['domain_confidence'] * 0.3 +
            factors['content_density'] * 0.2
        )
        strategy_scores[ChunkingStrategy.SEMANTIC] = semantic_score

        # Dimensional strategy scoring
        dimensional_score = (
            factors['dimensional_score'] * 0.6 +
            factors['array_ratio'] * 0.4
        )
        strategy_scores[ChunkingStrategy.DIMENSIONAL] = dimensional_score

        # Hybrid strategy scoring (for complex cases)
        hybrid_score = (
            factors['size_score'] * 0.3 +
            factors['complexity_score'] * 0.3 +
            factors['semantic_score'] * 0.2 +
            factors['dimensional_score'] * 0.2
        )
        strategy_scores[ChunkingStrategy.HYBRID] = hybrid_score

        # Select strategy with highest score
        best_strategy = max(strategy_scores.items(), key=lambda x: x[1])
        strategy, confidence = best_strategy

        # Apply minimum confidence threshold
        min_confidence = 0.4
        if confidence < min_confidence:
            # Fall back to hybrid for low confidence cases
            strategy = ChunkingStrategy.HYBRID
            confidence = max(confidence, min_confidence)

        return strategy, confidence

    def _generate_reasoning(self, strategy: ChunkingStrategy, factors: Dict[str, float]) -> str:
        """Generate human-readable reasoning for strategy selection."""

        reasoning_parts = []

        if strategy == ChunkingStrategy.FLAT:
            reasoning_parts.append(f"Document is relatively simple (depth: {factors['depth_score']:.1f}, complexity: {factors['complexity_score']:.1f})")
            reasoning_parts.append("Flat structure is suitable for direct key-value processing")

        elif strategy == ChunkingStrategy.HIERARCHICAL:
            reasoning_parts.append(f"Document has significant depth ({factors['depth_score']:.1f}) and complexity ({factors['complexity_score']:.1f})")
            reasoning_parts.append("Hierarchical chunking preserves structural relationships")

        elif strategy == ChunkingStrategy.SEMANTIC:
            reasoning_parts.append(f"High semantic content density ({factors['semantic_score']:.1f})")
            reasoning_parts.append(f"Strong domain specificity ({factors['domain_confidence']:.1f})")
            reasoning_parts.append("Content-aware chunking will preserve meaning")

        elif strategy == ChunkingStrategy.DIMENSIONAL:
            reasoning_parts.append(f"High array ratio ({factors['array_ratio']:.1f})")
            reasoning_parts.append("Document contains significant dimensional/tabular data")
            reasoning_parts.append("Array-aware chunking will preserve data relationships")

        elif strategy == ChunkingStrategy.HYBRID:
            reasoning_parts.append("Document exhibits multiple complexity factors")
            if factors['size_score'] > 0.5:
                reasoning_parts.append("Large document size requires multi-strategy approach")
            if factors['complexity_score'] > 0.7:
                reasoning_parts.append("High complexity benefits from adaptive chunking")
            reasoning_parts.append("Hybrid approach combines multiple strategies")

        return ". ".join(reasoning_parts) + "."

    def get_strategy_recommendations(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed strategy recommendations with alternatives."""

        strategy, details = self.decide_strategy(json_data)

        # Calculate scores for all strategies
        factors = details['decision_factors']
        all_scores = {}

        for strat in ChunkingStrategy:
            if strat == ChunkingStrategy.FLAT:
                score = factors['flat_suitability'] * 0.6 + (1 - factors['complexity_score']) * 0.4
            elif strat == ChunkingStrategy.HIERARCHICAL:
                score = factors['hierarchy_score'] * 0.5 + factors['complexity_score'] * 0.5
            elif strat == ChunkingStrategy.SEMANTIC:
                score = factors['semantic_score'] * 0.6 + factors['domain_confidence'] * 0.4
            elif strat == ChunkingStrategy.DIMENSIONAL:
                score = factors['dimensional_score'] * 0.7 + factors['array_ratio'] * 0.3
            elif strat == ChunkingStrategy.HYBRID:
                score = (factors['size_score'] + factors['complexity_score'] +
                        factors['semantic_score'] + factors['dimensional_score']) / 4

            all_scores[strat.value] = score

        # Sort alternatives
        sorted_strategies = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)

        return {
            'primary_recommendation': {
                'strategy': strategy.value,
                'confidence': details['confidence'],
                'reasoning': details['reasoning']
            },
            'alternative_strategies': [
                {'strategy': s, 'score': score, 'suitable_for': self._get_strategy_description(s)}
                for s, score in sorted_strategies[1:3]  # Top 2 alternatives
            ],
            'analysis_summary': {
                'document_size': factors['document_size'],
                'max_depth': details['structure_analysis']['max_depth'],
                'total_keys': details['structure_analysis']['total_keys'],
                'domain_type': details['content_analysis']['domain_type'],
                'has_arrays': details['structure_analysis']['array_count'] > 0,
                'semantic_content': len(details['content_analysis']['performance_metrics']) > 0
            }
        }

    def _get_strategy_description(self, strategy: str) -> str:
        """Get description of what each strategy is suitable for."""

        descriptions = {
            'flat': "Simple documents with shallow hierarchy and direct key-value pairs",
            'hierarchical': "Complex nested documents with deep structure and relationships",
            'semantic': "Content-rich documents with domain-specific terminology and reasoning",
            'dimensional': "Data-heavy documents with arrays, tables, and dimensional information",
            'hybrid': "Complex documents requiring multiple chunking approaches"
        }

        return descriptions.get(strategy, "General purpose chunking")

    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update engine configuration and reinitialize analyzers."""

        self.config.update(new_config)
        self.content_analyzer = ContentAnalyzer(self.config)

        if 'decision_thresholds' in new_config:
            self.decision_thresholds.update(new_config['decision_thresholds'])

        if 'strategy_weights' in new_config:
            self.strategy_weights.update(new_config['strategy_weights'])

        self.logger.info("Decision engine configuration updated")

    def get_strategy_config(self) -> Dict[str, Any]:
        """Get current strategy configuration."""
        return {
            'decision_thresholds': self.decision_thresholds,
            'strategy_weights': self.strategy_weights,
            'domain_patterns': list(self.config.get('domain_patterns', {}).keys()),
            'available_strategies': [strategy.value for strategy in ChunkingStrategy]
        }