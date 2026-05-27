# src/prompts.py

# =====================
# SYSTEM PROMPT
# =====================
SYSTEM_PROMPT = """You are an expert Olympic mathematician and Python programmer.
When given a math problem, you must solve it by writing a Python function.

Rules:
1. Always write a function named `solve()` that returns the final integer answer.
2. Use sympy, math, or itertools if needed.
3. Do NOT explain anything outside the code block.
4. Always return an integer, never None.

Example output format:
```python
def solve():
    # your solution here
    return answer
```
"""

# =====================
# FEW-SHOT EXEMPLARS
# =====================
FEW_SHOT_EXAMPLES = [
    {
        "problem": "Find the sum of all integers from 1 to 100.",
        "solution": """
```python
def solve():
    return sum(range(1, 101))
```
"""
    },
    {
        "problem": "Find the number of prime numbers less than 50.",
        "solution": """
```python
def solve():
    from sympy import isprime
    return sum(1 for n in range(2, 50) if isprime(n))
```
"""
    },
    {
        "problem": "If x^2 - 5x + 6 = 0, find the product of all solutions.",
        "solution": """
```python
def solve():
    from sympy import symbols, solve
    x = symbols('x')
    solutions = solve(x**2 - 5*x + 6, x)
    result = 1
    for s in solutions:
        result *= s
    return int(result)
```
"""
    },
]


# =====================
# HÀM BUILD PROMPT
# =====================
def build_prompt(problem: str) -> list:
    """
    Nhận đề bài, trả về list messages đúng format để gọi API.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    # Thêm few-shot examples
    for example in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": example["problem"]})
        messages.append({"role": "assistant", "content": example["solution"]})

    # Thêm đề bài thực sự
    messages.append({"role": "user", "content": problem})

    return messages

