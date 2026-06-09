import pytest
from src.parser import ResponseParser


def test_extract_python_code_block():
    text = """
    Here's the code:
    ```python
    x = 1
    result = x + 2
    ```
    """
    code = ResponseParser.extract_python_code(text)
    assert 'result' in code
    assert 'x' in code


def test_extract_python_code_multiple_blocks():
    text = """
    ```python
    result = 1
    ```
    ```python
    # final
    result = 2
    ```
    """
    code = ResponseParser.extract_python_code(text)
    assert 'result=2' in code.replace(' ', '')


def test_extract_python_code_no_block_but_code_like():
    text = "def f():\n    return 5\nresult = f()"
    code = ResponseParser.extract_python_code(text)
    assert 'def f' in code


def test_extract_boxed_answer_simple():
    assert ResponseParser.extract_boxed_answer('Answer: \\boxed{42}') == '42'


def test_extract_boxed_answer_nested():
    assert ResponseParser.extract_boxed_answer('Result: \\boxed{\\frac{22}{7}}') == '\\frac{22}{7}'


def test_extract_numeric_answer_from_boxed():
    assert ResponseParser.extract_numeric_answer('Answer: \\boxed{42}') == '42'


def test_extract_numeric_answer_from_result_assignment():
    assert ResponseParser.extract_numeric_answer('result = 123') == '123'


def test_extract_numeric_answer_from_print():
    assert ResponseParser.extract_numeric_answer('print(84)') == '84'


def test_extract_numeric_answer_from_text():
    assert ResponseParser.extract_numeric_answer('The final answer is 77.') == '77'
