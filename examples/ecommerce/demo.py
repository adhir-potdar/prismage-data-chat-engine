"""
E-commerce domain demo — uses the generic retail config in config/.
Runs sample questions and prints answers to stdout.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from api.chatbot import build_engine

SAMPLE_QUESTIONS = [
    "Show top 5 products by revenue this month",
    "Which sales reps are below target?",
    "Show MTD revenue growth by region",
    "Which product categories have the highest profit margin?",
    "Show average order value by customer segment",
    "Which regions are both below target and below last year MTD?",
    "Show revenue run rate by sales manager",
    "What is the gap to target for each region this month?",
    "Which customers have the lowest satisfaction score?",
    "Show YTD achievement percentage by sales rep",
]


CONFIG_DIR = str(Path(__file__).parent / "config" / "metadata")
PROMPTS_DIR = str(Path(__file__).parent.parent.parent / "config" / "prompts")


def main():
    engine = build_engine(config_dir=CONFIG_DIR, prompts_dir=PROMPTS_DIR)

    for i, question in enumerate(SAMPLE_QUESTIONS, 1):
        print(f"\n{'='*70}")
        print(f"Q{i}: {question}")
        print("="*70)
        response = engine.answer(question)
        print(response.answer)
        if response.used_fallback:
            print("  [fallback SQL chain used]")
        if response.applied_rules:
            print(f"  [rules: {', '.join(response.applied_rules)}]")


if __name__ == "__main__":
    main()
