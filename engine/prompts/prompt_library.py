"""
PromptLibrary — loads prompt templates from local JSON files or LangSmith Hub.
"""
from __future__ import annotations
import json
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate


class PromptLibrary:
    """
    Loads prompt templates by stage name.

    source: "local"        → reads from prompts_dir/*.json
    source: "langsmith_hub"→ pulls from LangSmith Hub by hub_ref (requires LANGSMITH_API_KEY)
    """

    def __init__(self, prompts_dir: str):
        self.prompts_dir = Path(prompts_dir)
        self._cache: dict[str, dict] = {}

    def get(self, name: str) -> dict:
        """Return raw prompt config dict for a stage (e.g. 'question_parser')."""
        if name not in self._cache:
            self._cache[name] = self._load(name)
        return self._cache[name]

    def get_system_template(self, name: str) -> str:
        return self.get(name).get("system", "")

    # ── Private ──────────────────────────────────────────────────────────────

    def _load(self, name: str) -> dict:
        path = self.prompts_dir / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")

        with open(path) as f:
            config = json.load(f)

        if config.get("source") == "langsmith_hub":
            return self._load_from_hub(config["hub_ref"])

        return config

    def _load_from_hub(self, hub_ref: str) -> dict:
        try:
            from langchain import hub
            prompt = hub.pull(hub_ref)
            return {"system": prompt.messages[0].prompt.template if prompt.messages else ""}
        except Exception as e:
            raise RuntimeError(f"Failed to load prompt from LangSmith Hub ({hub_ref}): {e}")
