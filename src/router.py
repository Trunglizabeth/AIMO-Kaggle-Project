"""
Simple Math Problem Router for Sprint 5.

Provides `MathRouter` which classifies problems into 'algebra' or 'geometry' (or 'other').
The default strategy is a lightweight heuristic; if an LLM engine is provided, it can be
used to improve classification in the future.
"""
from typing import Optional

GEOMETRY_KEYWORDS = {
    'triangle', 'circle', 'square', 'rectangle', 'angle', 'perimeter', 'area', 'volume', 'sphere', 'cylinder', 'cone', 'poly', 'isosceles', 'equilateral'
}

CLASSIFICATION_PROMPT = """You are a math problem classifier. Classify the following problem as exactly one of: algebra, geometry, or other. Answer with only the category word and no extra explanation.
"""


class MathRouter:
    def __init__(self, engine: Optional[object] = None):
        self.engine = engine

    def classify(self, problem_text: str) -> str:
        """Return one of: 'algebra', 'geometry', 'other'.

        If an LLM engine is available, use it to classify the problem text.
        Otherwise fall back to a lightweight keyword heuristic.
        """
        if self.engine is not None and hasattr(self.engine, 'generate_text'):
            try:
                raw = self.engine.generate_text(
                    problem_text,
                    temperature=0.0,
                    system_prompt=CLASSIFICATION_PROMPT,
                )
                normalized = raw.strip().lower()
                if 'geometry' in normalized:
                    return 'geometry'
                if 'algebra' in normalized:
                    return 'algebra'
                if 'other' in normalized:
                    return 'other'
            except Exception:
                pass

        return self._heuristic_classify(problem_text)

    def _heuristic_classify(self, problem_text: str) -> str:
        text = problem_text.lower()
        geom_hits = sum(1 for k in GEOMETRY_KEYWORDS if k in text)
        if geom_hits >= 1:
            return 'geometry'

        if any(k in text for k in ('solve', 'equation', 'integer', 'mod', 'remainder', 'prime', 'count', 'how many', 'digits', 'sum of')):
            return 'algebra'

        if len(text.split()) > 30 or any(ch in text for ch in ['+', '-', '*', '/', '^', '**']):
            return 'algebra'

        return 'other'
