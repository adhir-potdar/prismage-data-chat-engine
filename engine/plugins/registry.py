"""
PluginRegistry — holds all loaded plugin engines keyed by plugin name.
Provides a unified answer(plugin, question) interface.
"""
from __future__ import annotations
import logging
from models.query import ChatResponse
from engine.chains.chatbot_chain import ChatbotChain

logger = logging.getLogger(__name__)


class PluginRegistry:
    """
    Runtime registry of named plugin engines.

    Usage:
        registry = PluginRegistry()
        registry.register("haldiram-sales", chain)
        response = registry.answer("haldiram-sales", "Top 5 ASMs by cymtd?")
    """

    def __init__(self):
        self._engines: dict[str, ChatbotChain] = {}

    def register(self, name: str, chain: ChatbotChain) -> None:
        self._engines[name] = chain
        logger.info(f"Plugin registered: {name}")

    def get(self, name: str) -> ChatbotChain | None:
        return self._engines.get(name)

    def names(self) -> list[str]:
        return list(self._engines.keys())

    def answer(self, plugin: str, question: str) -> ChatResponse:
        chain = self._engines.get(plugin)
        if not chain:
            available = ", ".join(self._engines) or "(none loaded)"
            return ChatResponse(
                question=question,
                answer=f"Plugin '{plugin}' not found. Available plugins: {available}.",
                success=False,
                error="plugin_not_found",
            )
        return chain.answer(question)
