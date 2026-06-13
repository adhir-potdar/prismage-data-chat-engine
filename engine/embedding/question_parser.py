"""
Generic LLM-based question parser for embedding search.

Extracts: metrics, dimension combination, dimension values,
time granularities, and (optionally) date ranges.

All domain knowledge (metric names, dimension values, prompt template)
is injected from the plugin's schema_config and prompts_config.
"""
from __future__ import annotations
import json
import logging
import re
from datetime import datetime
from typing import Dict, List, Optional, Any

from engine.embedding.date_utils import parse_and_validate_date_range

logger = logging.getLogger(__name__)


class QuestionParser:
    """
    Parses a natural language question into structured extraction result.

    schema_config expected keys:
        time_granularities       (list[str])
        granularity_expansion    (dict[str, list[str]])
        dimensions.hierarchy     (list[str])
        date_extraction.enabled  (bool)
        date_extraction.prompt_if_no_dates (bool)
        date_extraction.confirm_low_confidence (bool)

    prompts_config expected keys:
        question_extraction  (str) — full prompt template with placeholders:
            {metrics_string}, {current_date}, {current_year}, {last_year}, {question}
    """

    def __init__(self, schema_config: Dict, prompts_config: Dict):
        self.schema = schema_config
        self.prompt_template: str = prompts_config.get('question_extraction', '')
        self.time_granularities: List[str] = schema_config.get('time_granularities', [])
        self.gran_expansion: Dict[str, List[str]] = {
            k.upper(): v for k, v in schema_config.get('granularity_expansion', {}).items()
        }
        date_cfg = schema_config.get('date_extraction', {})
        self.extract_dates: bool = date_cfg.get('enabled', True)
        self.prompt_if_no_dates: bool = date_cfg.get('prompt_if_no_dates', False)
        self.confirm_low_confidence: bool = date_cfg.get('confirm_low_confidence', False)

        # Metric list loaded externally and injected; start empty
        self._metric_names: List[str] = []

    def set_metrics(self, metric_names: List[str]) -> None:
        """Inject the list of valid metric names (loaded from plugin's kpi_metrics.csv)."""
        self._metric_names = sorted(metric_names)

    def parse(self, question: str, llm_service) -> Dict:
        """
        Parse question using LLM. Returns structured dict:
            success, metrics, dimension_combination, dimension_values,
            time_granularities, date_range (if enabled), raw_response (on error)
        """
        current_date = datetime.now().strftime('%Y%m%d')
        current_year = current_date[:4]
        last_year = str(int(current_year) - 1)
        metrics_string = ', '.join(self._metric_names) if self._metric_names else 'various metrics'

        prompt = self.prompt_template.format(
            metrics_string=metrics_string,
            metrics_count=len(self._metric_names),
            current_date=current_date,
            current_year=current_year,
            last_year=last_year,
            question=question,
        )

        try:
            result = llm_service.generate_answer(
                prompt=prompt,
                context='',
                query=question,
                temperature=0.1,
                max_tokens=500,
            )
        except Exception as exc:
            return {'success': False, 'error': str(exc)}

        if not result.get('success'):
            return {'success': False, 'error': result.get('error', 'LLM call failed')}

        extracted = self._parse_json_response(result['answer'])
        if extracted is None:
            return {
                'success': False,
                'error': 'Could not parse LLM response as JSON',
                'raw_response': result['answer'],
            }

        return {
            'success': True,
            'metrics': extracted.get('metrics', []),
            'dimension_combination': extracted.get('dimension_combination'),
            'dimension_values': extracted.get('dimension_values', {}),
            'time_granularities': extracted.get('time_granularities', []),
            'date_range': extracted.get('date_range', {}),
            'confidence': extracted.get('confidence', 'medium'),
        }

    def validate(self, extracted: Dict, normalize_fn=None) -> Dict:
        """
        Validate and normalize extracted info.

        Args:
            extracted:    result of parse()
            normalize_fn: optional callable(metric_str) -> canonical_name | None

        Returns:
            validation dict with keys:
                valid, metrics.valid, metrics.normalized, metrics.invalid,
                metrics.missing, dimensions.values, dimensions.combination,
                granularities.normalized, errors
        """
        result: Dict[str, Any] = {
            'valid': True,
            'metrics': {'valid': True, 'normalized': [], 'invalid': [], 'missing': False},
            'dimensions': {'values': {}, 'combination': None},
            'granularities': {'normalized': []},
            'errors': [],
        }

        # --- Metrics ---
        metrics = extracted.get('metrics', [])
        if not metrics:
            result['valid'] = False
            result['metrics']['valid'] = False
            result['metrics']['missing'] = True
            result['errors'].append('At least 1 metric is required')
        else:
            for metric in metrics:
                canonical = normalize_fn(metric) if normalize_fn else None
                if canonical:
                    if canonical not in result['metrics']['normalized']:
                        result['metrics']['normalized'].append(canonical)
                else:
                    result['metrics']['invalid'].append(metric)
                    result['metrics']['valid'] = False
                    result['valid'] = False

        # --- Dimensions (pass-through) ---
        result['dimensions']['values'] = extracted.get('dimension_values', {})
        result['dimensions']['combination'] = extracted.get('dimension_combination')

        # --- Granularities (normalize + expand) ---
        for gran in extracted.get('time_granularities', []):
            upper = gran.upper()
            expanded = self.gran_expansion.get(upper, [upper])
            for g in expanded:
                if g not in result['granularities']['normalized']:
                    result['granularities']['normalized'].append(g)

        return result

    def get_date_range(
        self,
        extracted: Dict,
        current_date_str: Optional[str] = None,
    ) -> Dict:
        """
        Extract and validate date range from parsed result.

        Returns dict with keys: start_date, end_date, status, error, confidence
        """
        if not self.extract_dates:
            return {'start_date': None, 'end_date': None, 'status': 'disabled'}

        if not current_date_str:
            current_date_str = datetime.now().strftime('%Y%m%d')

        date_info = extracted.get('date_range', {})
        start, end, status, error = parse_and_validate_date_range(date_info, current_date_str)

        return {
            'start_date': start,
            'end_date': end,
            'status': status,
            'error': error,
            'confidence': date_info.get('confidence', 'medium') if date_info else 'medium',
        }

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _parse_json_response(raw: str) -> Optional[Dict]:
        raw = raw.strip()
        # Try markdown code block first
        m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw, re.DOTALL)
        if m:
            json_str = m.group(1)
        else:
            s = raw.find('{')
            e = raw.rfind('}')
            if s < 0 or e <= s:
                return None
            json_str = raw[s:e + 1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None
