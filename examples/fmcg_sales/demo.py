"""
FMCG Sales domain demo — uses fmcg_sales config (primary/secondary channels).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from api.chatbot import build_engine

SAMPLE_QUESTIONS = [
    "Show top 5 products by CYMTD for primary and secondary",
    "Highlight the product and category under Anuj for target v/s achievement and LYMTD to CYMTD is both – ve in both Primary / Secondary and show me primary values and secondary values in tabular format",
    "Which RSM has highest current run rate?",
    "Show BTD by ZSM and product category",
    "for lowest performing asm and distributor under rsm Anuj highlight the lowest performing product and brand for CYMTD in tabular format",
    "Show required run rate by distributor",
    "Which brand has highest MTD growth percentage?",
    "Show YTD achievement percentage by RSM",
]

CONFIG_DIR = str(Path(__file__).parent / "config" / "metadata")
PROMPTS_DIR = str(Path(__file__).parent.parent.parent / "config" / "prompts")


def main():
    engine = build_engine(config_dir=CONFIG_DIR, prompts_dir=PROMPTS_DIR)
    for i, question in enumerate(SAMPLE_QUESTIONS, 1):
        print(f"\n{'='*70}\nQ{i}: {question}\n{'='*70}")
        response = engine.answer(question)
        print(response.answer)
        if response.applied_rules:
            print(f"  [rules: {', '.join(response.applied_rules)}]")


if __name__ == "__main__":
    main()
