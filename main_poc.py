"""Proof-of-concept runner for the AIMO Program-of-Thought pipeline.

Steps:
- Send a math problem to the LLM engine
- Parse the raw LLM output to extract Python code
- Execute the extracted code in a sandboxed PythonExecutor
- Print each step's output for debugging/presentation

Note: On Windows, ensure this script is run under the `if __name__ == '__main__'` guard
because `PythonExecutor` uses multiprocessing.
"""

from src.engine_api import LLMEngineAPI
from src.parser import ResponseParser
from src.executor import PythonExecutor
import os
import sys


SAMPLE_PROBLEM = "Find the sum of the roots of the equation x^2 - 5*x + 6 = 0."


def run_pipeline(problem_text: str):
    # Step A: Call LLM (with graceful fallback to a mock response)
    try:
        # If no API key is set, enable mock mode via env var so the engine can run locally
        if not os.getenv('LLM_API_KEY'):
            os.environ.setdefault('ALLOW_MOCK_LLM', 'true')
        engine = LLMEngineAPI()
        raw = engine.generate_code(problem_text)
    except Exception as e:
        raw = (
            "```python\n"
            "import sympy as sp\n"
            "def solve():\n"
            "    x = sp.symbols('x')\n"
            "    sol = sp.solve(sp.Eq(x**2 - 5*x + 6, 0), x)\n"
            "    # sum of roots = 5\n"
            "    return int(sum([int(s) for s in sol])) % 1000\n"
            "\n"
            "result = solve()\n"
            "```"
        )
        print('[WARN] LLM engine unavailable; using mock response. Error:', str(e))

    print('\n--- RAW LLM OUTPUT ---')
    print(raw)

    # Step B: Extract python code
    code = ResponseParser.extract_python_code(raw)
    print('\n--- EXTRACTED CODE ---')
    if code:
        print(code)
    else:
        print('[ERROR] No python code extracted from LLM output.')
        return

    # Step C: Execute code safely
    executor = PythonExecutor()
    result = executor.execute_code(code, timeout=5)

    print('\n--- FINAL RESULT ---')
    print(result)


if __name__ == '__main__':
    # Allow overriding the problem via CLI
    if len(sys.argv) > 1:
        problem = ' '.join(sys.argv[1:])
    else:
        problem = SAMPLE_PROBLEM

    run_pipeline(problem)
