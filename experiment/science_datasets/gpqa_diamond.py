"""
GPQA Diamond Dataset Handler

GPQA Diamond is a high-quality subset of GPQA containing the most
challenging and well-vetted graduate-level science questions.

Dataset: https://huggingface.co/datasets/Idavidrein/gpqa (config: gpqa_diamond)

The format is identical to the main GPQA dataset, so we reuse the same
formatter and evaluator.
"""

from .gpqa import gpqa_formatter, gpqa_evaluator, gpqa_scorer

# Alias for clarity
gpqa_diamond_formatter = gpqa_formatter
gpqa_diamond_evaluator = gpqa_evaluator
gpqa_diamond_scorer = gpqa_scorer

__all__ = [
    "gpqa_diamond_formatter",
    "gpqa_diamond_evaluator",
    "gpqa_diamond_scorer",
]
