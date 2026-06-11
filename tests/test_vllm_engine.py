import sys
import types

from src.vllm_engine import LocalLLMEngine


class FakeCompletion:
    def __init__(self, text):
        self.text = text


class FakeOutput:
    def __init__(self, outputs):
        self.outputs = outputs


class FakeLLM:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def generate(self, problem_texts, sampling_params):
        return [FakeOutput([FakeCompletion(f"```python\nresult = {i + 1}\n```\n") for i in range(sampling_params.n)]) for _ in problem_texts]


class FakeSamplingParams:
    def __init__(self, temperature, top_p, n, max_tokens, use_beam_search):
        self.temperature = temperature
        self.top_p = top_p
        self.n = n
        self.max_tokens = max_tokens
        self.use_beam_search = use_beam_search


def test_local_llm_engine_generate_batch(monkeypatch):
    fake_vllm = types.ModuleType('vllm')
    fake_vllm.LLM = FakeLLM
    fake_vllm.SamplingParams = FakeSamplingParams
    monkeypatch.setitem(sys.modules, 'vllm', fake_vllm)

    engine = LocalLLMEngine()
    outputs = engine.generate_batch(['Solve x+1=5'], n=3, temperature=0.5, top_p=0.9, max_tokens=50)

    assert isinstance(outputs, list)
    assert len(outputs) == 1
    assert len(outputs[0]) == 3
    assert outputs[0][0].strip().startswith('```python')
    assert 'result = 1' in outputs[0][0]
