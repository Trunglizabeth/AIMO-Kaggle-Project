import os
import sys
import types
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


def test_explicit_mock_mode(monkeypatch):
    monkeypatch.delenv('LLM_API_KEY', raising=False)
    engine = LLMEngineAPI(mock_mode=True)
    code = engine.generate_code('Solve for x: 2*x + 3 = 11')
    assert 'def solve' in code


def test_generate_code_samples(monkeypatch):
    monkeypatch.setenv('ALLOW_MOCK_LLM', 'true')
    monkeypatch.delenv('LLM_API_KEY', raising=False)
    engine = LLMEngineAPI()
    outputs = engine.generate_code_samples('Find the sum of the roots of x^2 - 5*x + 6 = 0', n=3)
    assert isinstance(outputs, list)
    assert len(outputs) == 3
    assert all('def solve' in out for out in outputs)


def test_solve_with_majority_voting(monkeypatch):
    monkeypatch.delenv('LLM_API_KEY', raising=False)
    engine = LLMEngineAPI(mock_mode=True)
    answer = engine.solve_with_majority_voting('Solve for x: 2*x + 3 = 11', n=3)
    assert answer == 4


def test_generate_text_with_custom_system_prompt(monkeypatch):
    monkeypatch.delenv('LLM_API_KEY', raising=False)
    engine = LLMEngineAPI(mock_mode=True)
    result = engine.generate_text(
        'Find the area of a circle with radius 2',
        system_prompt='Classify the following problem as algebra, geometry, or other.',
    )
    assert result in {'algebra', 'geometry', 'other'}


def test_use_vllm_backend(monkeypatch):
    monkeypatch.delenv('LLM_API_KEY', raising=False)

    class DummyLocalEngine:
        def __init__(self, *args, **kwargs):
            pass

        def generate_batch(self, problem_texts, n, temperature, top_p, max_tokens):
            return [["```python\ndef solve():\n    return 2\n\nresult = solve()\n```"]]

    fake_module = types.ModuleType('src.vllm_engine')
    fake_module.LocalLLMEngine = DummyLocalEngine
    monkeypatch.setitem(sys.modules, 'src.vllm_engine', fake_module)

    engine = LLMEngineAPI(use_vllm=True, mock_mode=True)
    raw = engine.generate_code('Solve for x: 2*x + 3 = 11')
    assert 'def solve' in raw
    assert 'return 2' in raw


def test_run_evaluation_cli_import():
    from scripts.run_evaluation import main as run_eval_main
    assert callable(run_eval_main)
