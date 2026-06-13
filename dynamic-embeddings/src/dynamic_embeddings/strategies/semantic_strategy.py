"""Semantic chunking strategy for content-rich JSON documents."""

from typing import Dict, Any, List, Set
from .base_strategy import BaseChunkingStrategy, DocumentChunk, ChunkMetadata
import re


class SemanticChunkingStrategy(BaseChunkingStrategy):
    """Strategy for chunking based on semantic content and meaning."""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.min_semantic_content = self.config.get('min_semantic_content', 3)
        self.max_chunk_size = self.config.get('max_chunk_size', 2048)
        self.group_related_content = self.config.get('group_related_content', True)
        self.preserve_reasoning = self.config.get('preserve_reasoning', True)

        # Load semantic patterns from config
        self.performance_keywords = self.config.get('performance_keywords', [])
        self.reasoning_keywords = self.config.get('reasoning_keywords', [])
        self.domain_patterns = self.config.get('domain_patterns', {})

    def chunk(self, json_data: Dict[str, Any], source_id: str = "document") -> List[DocumentChunk]:
        """Chunk JSON data based on semantic content.

        Args:
            json_data: The JSON data to chunk
            source_id: Identifier for the source document

        Returns:
            List of semantically coherent chunks
        """
        chunks = []

        # Identify semantic regions
        semantic_regions = self._identify_semantic_regions(json_data)

        # Create chunks from semantic regions
        for i, region in enumerate(semantic_regions):
            chunk = self._create_semantic_chunk(region, source_id, i)
            if chunk and self._validate_chunk(chunk):
                chunks.append(chunk)

        # If no semantic regions found, fall back to structure-based chunking
        if not chunks:
            chunks = self._fallback_chunking(json_data, source_id)

        return chunks

    def _identify_semantic_regions(self, json_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify semantically coherent regions in the JSON data."""
        regions = []

        # Analyze the data structure
        content_nodes = self._find_content_nodes(json_data)

        # Group related content
        if self.group_related_content:
            grouped_nodes = self._group_related_nodes(content_nodes)
            regions.extend(grouped_nodes)
        else:
            regions.extend(content_nodes)

        # Merge small regions if they're semantically similar
        regions = self._merge_similar_regions(regions)

        return regions

    def _find_content_nodes(self, obj: Any, path: List[str] = None, nodes: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Find nodes that contain semantic content."""
        if path is None:
            path = []
        if nodes is None:
            nodes = []

        if isinstance(obj, dict):
            # Check if this dict contains semantic content
            semantic_score = self._calculate_semantic_score(obj)

            if semantic_score >= self.min_semantic_content:
                nodes.append({
                    'content': obj,
                    'path': path.copy(),
                    'semantic_score': semantic_score,
                    'content_type': self._classify_content_type(obj),
                    'keywords': self._extract_keywords(obj)
                })
            else:
                # Recurse into subdicts
                for key, value in obj.items():
                    self._find_content_nodes(value, path + [key], nodes)

        elif isinstance(obj, list):
            # Process array items
            for i, item in enumerate(obj):
                if isinstance(item, dict):
                    self._find_content_nodes(item, path + [f"[{i}]"], nodes)

        return nodes

    def _calculate_semantic_score(self, obj: Dict[str, Any]) -> float:
        """Calculate semantic richness score for an object."""
        score = 0.0

        if not isinstance(obj, dict):
            return score

        # Convert object to text for analysis
        text_content = self._extract_text_content(obj)
        text_lower = text_content.lower()

        # Score based on performance keywords
        performance_matches = sum(1 for kw in self.performance_keywords if kw.lower() in text_lower)
        score += performance_matches * 1.0

        # Score based on reasoning keywords
        reasoning_matches = sum(1 for kw in self.reasoning_keywords if kw.lower() in text_lower)
        score += reasoning_matches * 1.5  # Reasoning content is more valuable

        # Score based on domain-specific patterns
        for domain, patterns in self.domain_patterns.items():
            domain_matches = sum(1 for pattern in patterns if pattern.lower() in text_lower)
            score += domain_matches * 0.5

        # Score based on content richness
        if any(key in obj for key in ['description', 'reasoning', 'analysis', 'explanation', 'comment']):
            score += 2.0

        # Score based on narrative content (longer text values)
        for value in obj.values():
            if isinstance(value, str) and len(value) > 50:
                score += 1.0

        # Score based on presence of quantitative data with context
        if self._has_contextual_metrics(obj):
            score += 1.0

        return score

    def _classify_content_type(self, obj: Dict[str, Any]) -> str:
        """Classify the type of content in the object."""
        keys = list(obj.keys())
        values = list(obj.values())
        text_content = self._extract_text_content(obj).lower()

        # Reasoning content
        if any(key in ['reasoning', 'explanation', 'analysis', 'insight', 'conclusion'] for key in keys):
            return 'reasoning'

        # Performance metrics
        if any(re.search(r'\d+%|\d+\.\d+%|ratio|rate|percentage', str(v)) for v in values):
            return 'performance_metrics'

        # Descriptive content
        if any(key in ['description', 'comment', 'note', 'summary'] for key in keys):
            return 'descriptive'

        # Configuration/settings
        if any(key in ['config', 'setting', 'option', 'parameter'] for key in keys):
            return 'configuration'

        # Entity data
        if any(key in ['id', 'name', 'title', 'type'] for key in keys):
            return 'entity'

        # Temporal data
        if any(key in ['date', 'time', 'created', 'updated', 'period'] for key in keys):
            return 'temporal'

        return 'general'

    def _extract_keywords(self, obj: Dict[str, Any]) -> Set[str]:
        """Extract important keywords from the object."""
        keywords = set()
        text_content = self._extract_text_content(obj).lower()

        # Extract domain-specific keywords
        for kw in self.performance_keywords + self.reasoning_keywords:
            if kw.lower() in text_content:
                keywords.add(kw)

        # Extract domain pattern matches
        for domain, patterns in self.domain_patterns.items():
            for pattern in patterns:
                if pattern.lower() in text_content:
                    keywords.add(pattern)

        # Extract meaningful terms from keys
        for key in obj.keys():
            if len(key) > 3 and key.lower() not in ['data', 'info', 'item']:
                keywords.add(key.lower())

        return keywords

    def _group_related_nodes(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Group semantically related nodes together."""
        if len(nodes) <= 1:
            return nodes

        grouped = []
        used_indices = set()

        for i, node in enumerate(nodes):
            if i in used_indices:
                continue

            # Start a new group
            group = {
                'content': node['content'].copy(),
                'path': node['path'],
                'semantic_score': node['semantic_score'],
                'content_type': node['content_type'],
                'keywords': node['keywords'].copy(),
                'grouped_paths': [node['path']]
            }

            used_indices.add(i)

            # Find related nodes
            for j, other_node in enumerate(nodes[i+1:], start=i+1):
                if j in used_indices:
                    continue

                if self._are_nodes_related(node, other_node):
                    # Merge nodes
                    self._merge_nodes(group, other_node)
                    used_indices.add(j)

            grouped.append(group)

        return grouped

    def _are_nodes_related(self, node1: Dict[str, Any], node2: Dict[str, Any]) -> bool:
        """Check if two nodes are semantically related."""

        # Same content type
        if node1['content_type'] == node2['content_type'] and node1['content_type'] != 'general':
            return True

        # Shared keywords
        shared_keywords = node1['keywords'] & node2['keywords']
        if len(shared_keywords) >= 2:
            return True

        # Similar path structure
        path1 = node1['path']
        path2 = node2['path']
        if len(path1) > 0 and len(path2) > 0 and path1[0] == path2[0]:
            return True

        return False

    def _merge_nodes(self, group: Dict[str, Any], node: Dict[str, Any]) -> None:
        """Merge a node into an existing group."""
        # Merge content
        merged_content = {}
        merged_content.update(group['content'])

        # Create a namespace for the new content to avoid key conflicts
        node_namespace = f"node_{len(group['grouped_paths'])}"
        merged_content[node_namespace] = node['content']

        group['content'] = merged_content
        group['semantic_score'] += node['semantic_score']
        group['keywords'].update(node['keywords'])
        group['grouped_paths'].append(node['path'])

    def _merge_similar_regions(self, regions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge small regions that are similar."""
        if len(regions) <= 1:
            return regions

        merged = []
        used_indices = set()

        for i, region in enumerate(regions):
            if i in used_indices:
                continue

            current_size = len(str(region['content']))

            # If region is small, try to merge it
            if current_size < self.max_chunk_size // 2:
                best_match_idx = None
                best_score = 0

                for j, other_region in enumerate(regions[i+1:], start=i+1):
                    if j in used_indices:
                        continue

                    similarity = self._calculate_similarity(region, other_region)
                    other_size = len(str(other_region['content']))

                    if (similarity > 0.3 and
                        current_size + other_size <= self.max_chunk_size):
                        if similarity > best_score:
                            best_score = similarity
                            best_match_idx = j

                if best_match_idx is not None:
                    # Merge regions
                    other_region = regions[best_match_idx]
                    merged_region = {
                        'content': {
                            **region['content'],
                            f"merged_content_{best_match_idx}": other_region['content']
                        },
                        'path': region['path'],
                        'semantic_score': region['semantic_score'] + other_region['semantic_score'],
                        'content_type': region['content_type'],
                        'keywords': region['keywords'] | other_region['keywords']
                    }
                    merged.append(merged_region)
                    used_indices.add(best_match_idx)
                else:
                    merged.append(region)
            else:
                merged.append(region)

            used_indices.add(i)

        return merged

    def _calculate_similarity(self, region1: Dict[str, Any], region2: Dict[str, Any]) -> float:
        """Calculate similarity between two regions."""
        similarity = 0.0

        # Content type similarity
        if region1['content_type'] == region2['content_type']:
            similarity += 0.3

        # Keyword overlap
        keywords1 = region1['keywords']
        keywords2 = region2['keywords']
        if keywords1 and keywords2:
            overlap = len(keywords1 & keywords2)
            total = len(keywords1 | keywords2)
            similarity += (overlap / total) * 0.5

        # Path similarity
        path1 = region1['path']
        path2 = region2['path']
        if path1 and path2:
            common_prefix = 0
            for p1, p2 in zip(path1, path2):
                if p1 == p2:
                    common_prefix += 1
                else:
                    break
            path_similarity = common_prefix / max(len(path1), len(path2))
            similarity += path_similarity * 0.2

        return min(similarity, 1.0)

    def _create_semantic_chunk(
        self,
        region: Dict[str, Any],
        source_id: str,
        chunk_index: int
    ) -> DocumentChunk:
        """Create a semantic chunk from a region."""

        content = region['content']
        path_str = ".".join(region['path']) if region['path'] else "root"

        # Generate semantic tags
        semantic_tags = [f"content_type:{region['content_type']}"]
        semantic_tags.extend([f"keyword:{kw}" for kw in list(region['keywords'])[:5]])  # Limit keywords

        metadata = ChunkMetadata(
            chunk_id=self._generate_chunk_id(source_id, f"semantic_{chunk_index}"),
            source_path=path_str,
            chunk_type="semantic",
            depth_level=len(region['path']),
            size_bytes=self._calculate_chunk_size(content),
            key_count=self._count_keys(content),
            contains_arrays=self._contains_arrays(content),
            domain_tags=semantic_tags
        )

        # Add semantic context to content
        enhanced_content = content.copy()
        if self.preserve_reasoning:
            enhanced_content['_semantic_context'] = {
                'semantic_score': region['semantic_score'],
                'content_type': region['content_type'],
                'keywords': list(region['keywords'])
            }

        return DocumentChunk(content=enhanced_content, metadata=metadata)

    def _extract_text_content(self, obj: Any) -> str:
        """Extract all text content from an object."""
        text_parts = []

        if isinstance(obj, str):
            text_parts.append(obj)
        elif isinstance(obj, dict):
            for key, value in obj.items():
                text_parts.append(key)
                text_parts.append(self._extract_text_content(value))
        elif isinstance(obj, list):
            for item in obj:
                text_parts.append(self._extract_text_content(item))
        else:
            text_parts.append(str(obj))

        return " ".join(str(part) for part in text_parts if part)

    def _has_contextual_metrics(self, obj: Dict[str, Any]) -> bool:
        """Check if object contains metrics with contextual information."""
        has_metrics = False
        has_context = False

        for key, value in obj.items():
            # Check for metric patterns
            if isinstance(value, (int, float)) or (
                isinstance(value, str) and re.search(r'\d+%|\d+\.\d+', value)
            ):
                has_metrics = True

            # Check for contextual information
            if isinstance(value, str) and len(value) > 20:
                has_context = True

        return has_metrics and has_context

    def _fallback_chunking(self, json_data: Dict[str, Any], source_id: str) -> List[DocumentChunk]:
        """Fallback chunking when no semantic regions are found."""
        chunks = []

        # Simple top-level chunking
        if isinstance(json_data, dict):
            for i, (key, value) in enumerate(json_data.items()):
                if isinstance(value, (dict, list)) and value:
                    content = {key: value}
                    metadata = ChunkMetadata(
                        chunk_id=self._generate_chunk_id(source_id, f"fallback_{i}"),
                        source_path=key,
                        chunk_type="semantic_fallback",
                        depth_level=1,
                        size_bytes=self._calculate_chunk_size(content),
                        key_count=self._count_keys(content),
                        contains_arrays=self._contains_arrays(content)
                    )
                    chunk = DocumentChunk(content=content, metadata=metadata)
                    chunks.append(chunk)

        return chunks

    def get_strategy_name(self) -> str:
        """Get the name of this strategy."""
        return "semantic"