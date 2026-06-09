from src.router import MathRouter


def test_math_router_falls_back_to_heuristic_without_engine():
    router = MathRouter()
    assert router.classify('Find the area of a circle with radius 5') == 'geometry'
    assert router.classify('Solve for x: 2*x + 3 = 11') == 'algebra'
    assert router.classify('Is the sky blue?') == 'other'


def test_math_router_uses_engine_classification():
    class FakeEngine:
        def generate_text(self, problem_text, temperature=None, system_prompt=None):
            assert system_prompt is not None
            assert 'classify' in system_prompt.lower()
            return 'geometry'

    router = MathRouter(engine=FakeEngine())
    assert router.classify('Find the sum of interior angles of a triangle') == 'geometry'


def test_math_router_handles_engine_errors_gracefully():
    class BrokenEngine:
        def generate_text(self, *args, **kwargs):
            raise RuntimeError('mock fail')

    router = MathRouter(engine=BrokenEngine())
    assert router.classify('Solve for x: 2*x + 3 = 11') == 'algebra'
