"""
Simple OpenAI LLM service for generating answers from prompt, context, and query.

This service focuses solely on making OpenAI API calls with the provided inputs.
"""

import os
import logging
from typing import Dict, Any, Optional
from openai import OpenAI
from datetime import datetime


class LLMService:
    """Simple OpenAI LLM service for answer generation."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        """Initialize the LLM service.

        Args:
            api_key: OpenAI API key (uses env var if None)
            model: OpenAI model to use
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key is required")

        self.model = model
        self.client = OpenAI(api_key=self.api_key)
        self.logger = logging.getLogger(__name__)

    def generate_answer(
        self,
        prompt: str,
        context: str,
        query: str,
        temperature: float = 0.1,
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """Generate answer using OpenAI with the provided prompt, context, and query.

        Args:
            prompt: The system prompt/instructions for the LLM
            context: Retrieved context from PGVector
            query: User's question
            temperature: Controls randomness (0.0-1.0)
            max_tokens: Maximum tokens in response

        Returns:
            Dictionary with answer and metadata
        """
        try:
            # Create the complete user message
            user_message = f"""CONTEXT:
{context}

QUERY: {query}"""

            # Make OpenAI API call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": prompt
                    },
                    {
                        "role": "user",
                        "content": user_message
                    }
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )

            answer = response.choices[0].message.content

            return {
                'success': True,
                'answer': answer,
                'model': self.model,
                'tokens_used': response.usage.total_tokens if response.usage else None,
                'prompt_tokens': response.usage.prompt_tokens if response.usage else None,
                'completion_tokens': response.usage.completion_tokens if response.usage else None,
                'timestamp': datetime.now().isoformat(),
                'temperature': temperature,
                'max_tokens': max_tokens
            }

        except Exception as e:
            self.logger.error(f"OpenAI API call failed: {e}")
            return {
                'success': False,
                'answer': None,
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'model': self.model
            }

    def get_service_info(self) -> Dict[str, Any]:
        """Get service configuration information."""
        return {
            'model': self.model,
            'api_configured': bool(self.api_key),
            'service_name': 'OpenAI LLM Service'
        }