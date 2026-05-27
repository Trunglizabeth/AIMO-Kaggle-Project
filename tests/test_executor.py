from src.executor import PythonExecutor


def test_execute_simple():
    ex = PythonExecutor()
    code = "result = 2 + 3"
    out = ex.execute_code(code, timeout=3)
    assert out == '5'


def test_execute_timeout():
    ex = PythonExecutor()
    code = "while True: pass"
    out = ex.execute_code(code, timeout=1)
    assert out == 'Timeout Error'


def test_execute_error():
    ex = PythonExecutor()
    code = "result = 1/0"
    out = ex.execute_code(code, timeout=3)
    assert 'ZeroDivisionError' in out
