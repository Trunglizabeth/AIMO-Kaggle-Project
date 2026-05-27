import os
from src.engine_api import LLMEngineAPI


def test_mock_quadratic(monkeypatch):
    # Ensure mock mode is enabled
    monkeypatch.setenv('ALLOW_MOCK_LLM', 'true')
    if 'LLM_API_KEY' in os.environ:
        monkeypatch.delenv('LLM_API_KEY')

    engine = LLMEngineAPI()
    code = engine.generate_code('Find the sum of the roots of x^2 - 5*x + 6 = 0')
    assert 'def solve' in code
    assert 'x**2' in code or 'x^2' in code or 'sympy' in code


def test_mock_linear(monkeypatch):
    monkeypatch.setenv('ALLOW_MOCK_LLM', 'true')
    monkeypatch.delenv('LLM_API_KEY', raising=False)
    engine = LLMEngineAPI()
    code = engine.generate_code('Solve for x: 2*x + 3 = 11')
    assert 'def solve' in code
    assert 'sympy' in code or 'return' in code
