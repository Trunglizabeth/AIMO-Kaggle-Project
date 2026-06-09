"""LLM engine API wrapper for Program-of-Thought AIMO solver.

Provides a `SYSTEM_PROMPT` and a lightweight `LLMEngineAPI` class that calls
an LLM (OpenAI-style or Google GenAI) to generate Python code solving math problems.

The wrapper prefers the `openai` package format and falls back to `google.generativeai`.
If neither client is available and mock mode is enabled via the environment variable
`ALLOW_MOCK_LLM=true`, `generate_code` will return deterministic mock code useful for
local development and demos.
"""

from typing import Optional
import os
import re
import time


# SYSTEM_PROMPT instructs the LLM to behave as an expert mathematical programmer
SYSTEM_PROMPT = """You are an elite mathematical problem solver and Python programmer.
Your task is to solve complex math problems by writing a Python script.

INSTRUCTIONS:
1. You must write a complete, standalone Python function named `solve()`.
2. DO NOT just return 0. You MUST implement the actual mathematical logic.
3. For Algebra/Calculus: Use the `sympy` library.
4. For Combinatorics/Counting/Number Theory: You are strongly encouraged to use standard Python features (loops, lists, sets, `itertools`, `math`) to brute-force or simulate the problem if an analytical solution is too complex.
5. The function `solve()` must return the final answer as a single integer.
6. If the final answer is a real number, round it or extract the required integer part as requested in the problem.
7. Wrap your code inside a standard markdown Python block (```python ... ```).
8. At the very end of your code, outside the function, call it and assign the output to a variable named `result`: `result = solve()`.

EXAMPLE PROBLEM:
How many integers between 10 and 100 have distinct digits?

EXAMPLE CODE:
```python
def solve():
    count = 0
    for i in range(10, 101):
        # Convert number to string to check for distinct digits
        if len(set(str(i))) == len(str(i)):
            count += 1
    return count

result = solve()
"""


class LLMEngineAPI:
    """Lightweight wrapper to call an LLM and return raw text responses.

    Initialization:
        - By default, the API key is read from `src.config.config.LLM_API_KEY` if that
          object exists, otherwise from the `LLM_API_KEY` environment variable.
        - To enable local mock mode, either set `ALLOW_MOCK_LLM=true` or pass
          `mock_mode=True` to the constructor.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        mock_mode: Optional[bool] = None,
        use_vllm: bool = False,
        vllm_model: Optional[str] = None,
        vllm_quantization: str = 'awq',
        vllm_gpu_memory_utilization: float = 0.9,
        vllm_temperature: float = 0.7,
        vllm_top_p: float = 0.95,
        vllm_max_tokens: int = 1024,
    ):
        # Prefer explicit API key
        if api_key:
            self.api_key = api_key
        else:
            # Try to import config safely to avoid import-time failures
            try:
                import importlib
                cfg_mod = importlib.import_module('src.config')
                if hasattr(cfg_mod, 'config'):
                    try:
                        self.api_key = cfg_mod.config.LLM_API_KEY
                    except Exception:
                        self.api_key = os.getenv('LLM_API_KEY', '')
                else:
                    self.api_key = os.getenv('LLM_API_KEY', '')
            except Exception:
                self.api_key = os.getenv('LLM_API_KEY', '')

        # Allow mock mode via explicit parameter or environment variable
        allow_mock_env = os.getenv('ALLOW_MOCK_LLM', 'false').lower() in ('1', 'true', 'yes')
        self.allow_mock = mock_mode if mock_mode is not None else allow_mock_env
        self.use_vllm = use_vllm
        self.vllm_model = vllm_model
        self.vllm_quantization = vllm_quantization
        self.vllm_gpu_memory_utilization = vllm_gpu_memory_utilization
        self.vllm_temperature = vllm_temperature
        self.vllm_top_p = vllm_top_p
        self.vllm_max_tokens = vllm_max_tokens

        # model name / settings
        try:
            from src import config as _cfg
            config_model = getattr(_cfg, 'MODEL_NAME', None) or (hasattr(_cfg, 'config') and getattr(_cfg.config, 'MODEL_NAME', None))
        except Exception:
            config_model = None

        default_model = 'Qwen/Qwen2.5-Math-7B-Instruct-AWQ' if self.use_vllm else 'gpt-4o-mini'
        self.model_name = model_name or vllm_model or config_model or default_model

        if self.use_vllm:
            self._provider = 'vllm'
            self._client = None
            try:
                from src.vllm_engine import LocalLLMEngine
                self._local_engine = LocalLLMEngine(
                    model=self.model_name,
                    quantization=self.vllm_quantization,
                    gpu_memory_utilization=self.vllm_gpu_memory_utilization,
                    temperature=self.vllm_temperature,
                    top_p=self.vllm_top_p,
                    max_tokens=self.vllm_max_tokens,
                )
            except Exception as e:
                if self.allow_mock:
                    self._provider = 'mock'
                    self._client = None
                    self._local_engine = None
                else:
                    raise
            return

        if not self.api_key and not self.allow_mock:
            raise ValueError('LLM API key not set. Set LLM_API_KEY in environment or set ALLOW_MOCK_LLM=true or mock_mode=True to enable mock mode.')

        # If no API key and mock mode requested, prefer mock provider and skip client imports
        if not self.api_key and self.allow_mock:
            self._provider = 'mock'
            self._client = None
            self._local_engine = None
            return

        self._provider = None
        self._client = None
        self._local_engine = None

        # Prepare an LLM client: prefer OpenAI, fall back to Google GenAI
        try:
            import openai
            openai.api_key = self.api_key
            self._client = openai
            self._provider = 'openai'
        except Exception:
            try:
                import google.generativeai as genai
                try:
                    genai.configure(api_key=self.api_key)
                except Exception:
                    try:
                        genai.api_key = self.api_key
                    except Exception:
                        pass
                self._client = genai
                self._provider = 'google'
            except Exception:
                if self.allow_mock:
                    self._provider = 'mock'
                    self._client = None
                else:
                    raise ImportError('No supported LLM client found. Install `openai` or `google-genai`.')

    def generate_code(self, problem_text: str, temperature: Optional[float] = None) -> str:
        """Send the SYSTEM_PROMPT and user's problem text to the LLM and return raw response text.

        If in mock mode, return deterministic code snippets for common problem patterns.
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem_text},
        ]

        # MOCK mode heuristics
        if self._provider == 'mock':
            pt = problem_text.strip()

            # Quadratic: detect x^2 or x**2
            if re.search(r"x\s*(?:\^|\*\*)\s*2", pt):
                nums = re.findall(r'([+-]?\d+)', pt)
                if len(nums) >= 3:
                    a, b, c = nums[0], nums[1], nums[2]
                else:
                    a, b, c = '1', '0', '0'
                return f"""```python
import sympy as sp

def solve():
    x = sp.symbols('x')
    sol = sp.solve(sp.Eq({a}*x**2 + ({b})*x + ({c}), 0), x)
    return int(sum(sol)) % 1000

result = solve()
```
"""

            # Linear equation ax + b = c
            m = re.search(r'([+-]?\d*)\s*\*?x\s*([+-]\s*\d+)?\s*=\s*([+-]?\d+)', pt)
            if m:
                a = m.group(1).strip() or '1'
                b = m.group(2).replace(' ', '') if m.group(2) else '+0'
                c = m.group(3)
                return f"""```python
import sympy as sp

def solve():
    x = sp.symbols('x')
    sol = sp.solve(sp.Eq({a}*x + ({b}), {c}), x)
    return int(sol[0]) % 1000

result = solve()
```
"""

            # Modular exponent: like '2**50 mod 1000' or '2^50 mod 1000'
            mmod = re.search(r'([0-9]+)\s*(?:\*\*|\^)\s*([0-9]+)\s*(?:mod|%|modulo)\s*([0-9]+)', pt, re.IGNORECASE)
            if mmod:
                base, exp, modv = mmod.groups()
                return f"""```python
def solve():
    return pow({base}, {exp}, {modv})

result = solve()
```
"""

            # Simple system detection (heuristic)
            if 'system of' in pt.lower() or '\n' in pt or ';' in pt:
                return """```python
import sympy as sp

def solve():
    x, y = sp.symbols('x y')
    sol = sp.solve([sp.Eq(x + y, 0), sp.Eq(x - y, 0)], (x, y))
    vals = list(sol.values()) if isinstance(sol, dict) else list(sol[0])
    return int(sum(vals)) % 1000

result = solve()
```
"""

            # Fallback mock
            return """```python
import sympy as sp

def solve():
    return 0

result = solve()
```
"""

        if self._provider == 'vllm':
            outputs = self._safe_vllm_generate_batch([
                problem_text
            ], n=1, temperature=temperature, top_p=self.vllm_top_p, max_tokens=self.vllm_max_tokens)
            # outputs is List[List[str]] -> return first completion text or empty string
            try:
                return outputs[0][0] if outputs and outputs[0] else ''
            except Exception:
                if self.allow_mock:
                    return """```python
import sympy as sp

def solve():
    # Mock fallback due to vLLM error
    return 0

result = solve()
```
"""
                raise

        # Prepare API call params
        client = self._client
        params = {
            'model': self.model_name,
            'messages': messages,
        }
        if temperature is not None:
            params['temperature'] = float(temperature)

        # Respect max tokens if defined
        try:
            from src import config as _cfg
            if hasattr(_cfg, 'config') and hasattr(_cfg.config, 'MAX_TOKENS'):
                params['max_tokens'] = int(_cfg.config.MAX_TOKENS)
        except Exception:
            pass

        # Call the selected provider
        try:
            if self._provider == 'openai':
                resp = client.ChatCompletion.create(**params)
                if resp and 'choices' in resp and len(resp['choices']) > 0:
                    message = resp['choices'][0].get('message', {})
                    content = message.get('content') if isinstance(message, dict) else None
                    if not content:
                        content = resp['choices'][0].get('text', '')
                    return content or ''
                return str(resp)

            elif self._provider == 'google':
                genai = client
                genai_kwargs = {
                    'model': self.model_name or 'models/text-bison-001',
                    'messages': messages,
                }
                if 'max_tokens' in params:
                    genai_kwargs['max_output_tokens'] = params['max_tokens']
                if temperature is not None:
                    genai_kwargs['temperature'] = float(temperature)

                resp = genai.chat.completions.create(**genai_kwargs)

                try:
                    if hasattr(resp, 'candidates') and len(resp.candidates) > 0:
                        cand = resp.candidates[0]
                        content = getattr(cand, 'content', None)
                        if content is None and isinstance(cand, dict):
                            content = cand.get('content') or cand.get('display') or cand.get('text')
                        if isinstance(content, (list, tuple)):
                            content = ''.join([seg.get('text', '') if isinstance(seg, dict) else str(seg) for seg in content])
                        return content or str(resp)
                except Exception:
                    pass

                if isinstance(resp, dict) and 'candidates' in resp and len(resp['candidates']) > 0:
                    cand = resp['candidates'][0]
                    if isinstance(cand, dict):
                        return cand.get('content', cand.get('text', '')) or str(resp)

                return str(resp)

        except Exception as e:
            # If mock mode is allowed, return a safe mock snippet instead of an error string
            if self.allow_mock:
                return """```python
import sympy as sp

def solve():
    # Mock fallback due to LLM API error
    return 0

result = solve()
```
"""
            return f'Error calling LLM API: {type(e).__name__}: {str(e)}'

    def _safe_vllm_generate_batch(self, problem_texts, n=1, temperature=None, top_p=None, max_tokens=None, retries: int = 1):
        """Call the LocalLLMEngine.generate_batch with defensive checks and optional retries.

        Returns a List[List[str]] where each inner list has up to `n` strings.
        If generation fails and `allow_mock` is True, falls back to mock outputs.
        """
        if not hasattr(self, '_local_engine') or self._local_engine is None:
            # Try to lazily initialize once
            try:
                from src.vllm_engine import LocalLLMEngine
                self._local_engine = LocalLLMEngine(
                    model=self.model_name,
                    quantization=self.vllm_quantization,
                    gpu_memory_utilization=self.vllm_gpu_memory_utilization,
                    temperature=self.vllm_temperature,
                    top_p=self.vllm_top_p,
                    max_tokens=self.vllm_max_tokens,
                )
            except Exception as e:
                if self.allow_mock:
                    # Return mock repeated outputs
                    mock = self.generate_code(problem_texts[0], temperature=temperature)
                    return [[mock] * n for _ in problem_texts]
                raise RuntimeError(f'Failed to initialize local vLLM engine: {e}') from e

        attempt = 0
        last_err = None
        while attempt <= retries:
            try:
                outputs = self._local_engine.generate_batch(
                    problem_texts,
                    n=n,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                )
                # Validate structure: expect list of lists
                if outputs is None:
                    raise RuntimeError('vLLM returned None outputs')
                if not isinstance(outputs, list):
                    # Try to coerce into expected format
                    outputs = [[str(outputs) for _ in range(n)]]
                # Ensure each entry is a list
                normalized = []
                for o in outputs:
                    if isinstance(o, list):
                        normalized.append([str(x) for x in o])
                    else:
                        normalized.append([str(o) for _ in range(n)])
                return normalized
            except Exception as e:
                last_err = e
                attempt += 1
                if attempt > retries:
                    break
                time.sleep(0.5 * attempt)

        # If we reach here, generation failed
        if self.allow_mock:
            mock = self.generate_code(problem_texts[0], temperature=temperature)
            return [[mock] * n for _ in problem_texts]
        raise RuntimeError(f'vLLM generation failed after {retries+1} attempts: {last_err}') from last_err

    def generate_code_samples(self, problem_text: str, n: int = 1, temperature: Optional[float] = None) -> list[str]:
        """Generate multiple raw outputs from the LLM for the same problem.

        This is useful for self-consistency / majority voting pipelines.
        """
        if n <= 0:
            return []

        if self._provider == 'mock':
            return [self.generate_code(problem_text, temperature=temperature) for _ in range(n)]

        if self._provider == 'vllm':
            outputs = self._safe_vllm_generate_batch([
                problem_text
            ], n=n, temperature=temperature, top_p=self.vllm_top_p, max_tokens=self.vllm_max_tokens)
            try:
                return outputs[0] if outputs and outputs[0] else []
            except Exception:
                if self.allow_mock:
                    return [self.generate_code(problem_text, temperature=temperature) for _ in range(n)]
                raise

        if self._provider == 'openai':
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": problem_text},
            ]
            params = {
                'model': self.model_name,
                'messages': messages,
                'n': n,
            }
            if temperature is not None:
                params['temperature'] = float(temperature)
            try:
                from src import config as _cfg
                if hasattr(_cfg, 'config') and hasattr(_cfg.config, 'MAX_TOKENS'):
                    params['max_tokens'] = int(_cfg.config.MAX_TOKENS)
            except Exception:
                pass

            resp = self._client.ChatCompletion.create(**params)
            outputs = []
            if resp and 'choices' in resp:
                for choice in resp['choices']:
                    message = choice.get('message', {})
                    content = message.get('content') if isinstance(message, dict) else None
                    if not content:
                        content = choice.get('text', '')
                    outputs.append(content or '')
            return outputs

        if self._provider == 'google':
            outputs = []
            for _ in range(n):
                outputs.append(self.generate_code(problem_text, temperature=temperature))
            return outputs

        return [self.generate_code(problem_text, temperature=temperature) for _ in range(n)]

    def solve_with_majority_voting(self, problem_text: str, n: Optional[int] = None, timeout: int = 5) -> int:
        """Run the full pipeline `n` times and return the majority-voted integer answer.

        - Calls `generate_code` to get raw LLM output
        - Extracts python code via `ResponseParser.extract_python_code`
        - Executes code with `PythonExecutor.execute_code`
        - Collects valid integer answers and returns the most common one.

        Args:
            problem_text (str): The math problem in plain text.
            n (Optional[int]): Number of samples to draw. If None, use config.MAJORITY_VOTING_N.
            timeout (int): Execution timeout (seconds) for each run.

        Returns:
            int: The majority-voted integer answer, or 0 as a fallback if no valid answers.
        """
        # Lazy imports to avoid module-level cycles and reduce startup cost
        from collections import Counter
        from src.parser import ResponseParser
        from src.executor import PythonExecutor
        try:
            from src import config as _cfg
            default_n = getattr(_cfg.config, 'MAJORITY_VOTING_N', 15)
        except Exception:
            default_n = 15

        runs = int(n) if n is not None else int(default_n)
        parser = ResponseParser()
        executor = PythonExecutor()

        def _to_int(val_str):
            try:
                return int(val_str)
            except Exception:
                try:
                    return int(float(val_str))
                except Exception:
                    return None

        votes = []

        if self._provider == 'vllm':
            try:
                raw_outputs = self._safe_vllm_generate_batch(
                    [problem_text],
                    n=runs,
                    temperature=self.vllm_temperature,
                    top_p=self.vllm_top_p,
                    max_tokens=self.vllm_max_tokens,
                )
                raw_list = raw_outputs[0] if raw_outputs else []
            except Exception:
                raw_list = []

            for raw in raw_list:
                code = parser.extract_python_code(raw)
                if not code:
                    continue

                out = executor.execute_code(code, timeout=timeout)
                if out == 'Timeout Error':
                    continue
                if isinstance(out, str) and out.startswith('Error'):
                    continue

                candidate = _to_int(out)
                if candidate is None:
                    continue
                try:
                    candidate = int(candidate) % 1000
                except Exception:
                    continue
                votes.append(candidate)
        else:
            for i in range(runs):
                raw = self.generate_code(problem_text)
                code = parser.extract_python_code(raw)
                if not code:
                    continue

                out = executor.execute_code(code, timeout=timeout)
                if out == 'Timeout Error':
                    continue
                if isinstance(out, str) and out.startswith('Error'):
                    continue

                candidate = _to_int(out)
                if candidate is None:
                    continue
                try:
                    candidate = int(candidate) % 1000
                except Exception:
                    continue
                votes.append(candidate)

        if not votes:
            return 0

        counter = Counter(votes)
        most_common, _ = counter.most_common(1)[0]
        return int(most_common)

