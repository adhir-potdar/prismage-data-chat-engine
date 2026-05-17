"""
PluginLoader — loads a single plugin from a plugin directory.

A plugin is a self-contained directory with:
    plugin.json       — manifest (name, version, config_dir, prompts_dir)
    config/metadata/  — 5 JSON metadata files
    config/prompts/   — prompt JSON files
    __init__.py       — optional; can expose plugin-level helpers
"""
from __future__ import annotations
import json
import logging
from pathlib import Path
from engine.chains.chatbot_chain import ChatbotChain

logger = logging.getLogger(__name__)


class PluginLoader:
    """
    Loads a plugin directory into a configured ChatbotChain.

    All engine wiring is delegated to api.chatbot.build_engine() so that
    plugin loading stays consistent with the standard engine factory and
    zero Haldiram-specific logic leaks into this file.
    """

    def load(
        self,
        plugin_dir: str,
        connection_string: str | None = None,
        llm_provider: str | None = None,
        llm_model: str | None = None,
        **kwargs,
    ) -> ChatbotChain:
        """
        Load a plugin from plugin_dir and return a fully-wired ChatbotChain.

        Args:
            plugin_dir:         absolute or relative path to the plugin directory
            connection_string:  SQLAlchemy DB URL (overrides DATABASE_URL env var)
            llm_provider:       "openai" or "anthropic" (overrides PRISMAGE_LLM_PROVIDER)
            llm_model:          model name (overrides PRISMAGE_LLM_MODEL)
            **kwargs:           any other kwargs forwarded to build_engine()
        """
        plugin_path = Path(plugin_dir).resolve()
        manifest = self._read_manifest(plugin_path)

        plugin_name = manifest.get("name", plugin_path.name)
        config_subdir = manifest.get("config_dir", "config/metadata")
        prompts_subdir = manifest.get("prompts_dir", "config/prompts")

        config_dir = str(plugin_path / config_subdir)
        prompts_dir = str(plugin_path / prompts_subdir)

        logger.info(f"Loading plugin '{plugin_name}' from {plugin_path}")

        # Defer to the standard engine factory — no plugin-specific logic here
        from api.chatbot import build_engine
        return build_engine(
            config_dir=config_dir,
            prompts_dir=prompts_dir,
            connection_string=connection_string,
            llm_provider=llm_provider,
            llm_model=llm_model,
            **kwargs,
        )

    # ── Private ──────────────────────────────────────────────────────────────

    def _read_manifest(self, plugin_path: Path) -> dict:
        manifest_path = plugin_path / "plugin.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"plugin.json not found in {plugin_path}. "
                "Every plugin directory must contain a plugin.json manifest."
            )
        with open(manifest_path) as f:
            return json.load(f)
