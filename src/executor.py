"""Safe Python code executor for evaluating LLM-generated math solutions."""

from multiprocessing import Process, Queue


def _run_code_worker(code_string: str, queue: Queue) -> None:
    """Module-level worker function to execute code in a subprocess.

    Defined at module scope so it is picklable on Windows spawn start method.
    """
    try:
        # Import inside worker to reduce pickling across processes
        import io as _io
        import math as _math
        import sympy as _sympy
        import sys as _sys
        from contextlib import redirect_stdout as _redirect_stdout

        exec_globals = {
            "math": _math,
            "sympy": _sympy,
        }

        stdout_buffer = _io.StringIO()
        with _redirect_stdout(stdout_buffer):
            exec(code_string, exec_globals)

        result = exec_globals.get("result", None)
        if result is not None:
            queue.put(("success", str(result)))
            return

        stdout_value = stdout_buffer.getvalue().strip()
        if stdout_value:
            queue.put(("success", stdout_value))
            return

        queue.put(("error", "No result returned from subprocess"))

    except SyntaxError as e:
        queue.put(("error", f"SyntaxError: {str(e)}"))
    except ZeroDivisionError as e:
        queue.put(("error", f"ZeroDivisionError: {str(e)}"))
    except TypeError as e:
        queue.put(("error", f"TypeError: {str(e)}"))
    except ValueError as e:
        queue.put(("error", f"ValueError: {str(e)}"))
    except NameError as e:
        queue.put(("error", f"NameError: {str(e)}"))
    except Exception as e:
        queue.put(("error", f"{type(e).__name__}: {str(e)}"))


import math  # keep module-level reference for documentation and type hints
import sympy


class PythonExecutor:
    """
    A safe and robust Python code executor for evaluating LLM-generated code.

    This executor runs code in a separate process with a strict timeout mechanism
    to prevent infinite loops from hanging the main system. It provides a sandboxed
    namespace with math and sympy libraries pre-injected for mathematical computations.

    Features:
        - Process-level isolation prevents infinite loops from affecting the main system
        - Strict timeout mechanism terminates runaway code
        - Pre-injected math and sympy libraries for mathematical operations
        - Comprehensive exception handling with detailed error messages
        - Extracts and returns the 'result' variable after successful execution

    Important:
        On Windows, ensure this class is instantiated within a `if __name__ == '__main__':`
        guard in your main script to avoid multiprocessing issues.

    Example:
        >>> executor = PythonExecutor()
        >>> code = '''
        ... import math
        ... x = 5
        ... result = x ** 2 + math.sqrt(x)
        ... '''
        >>> output = executor.execute_code(code, timeout=5)
        >>> print(output)  # Output: 28.236...
    """

    def execute_code(self, code_string: str, timeout: int = 5) -> str:
        """
        Execute Python code in a completely isolated process with timeout protection.

        The code execution environment includes:
            - `math` module: standard library math functions
            - `sympy` module: symbolic mathematics library
            - Full Python builtins

        Args:
            code_string (str): The Python code to execute. The code should assign
                              its final result to a variable named `result`.
            timeout (int): Maximum execution time in seconds. Default is 5 seconds.
                          If exceeded, the process is terminated and "Timeout Error"
                          is returned.

        Returns:
            str: Either the string representation of the `result` variable on success,
                 or an error message on failure or timeout.

        Error Message Format:
            - "Timeout Error": Code execution exceeded the specified timeout
            - "Error: {ExceptionType}: {message}": Exception occurred during execution
            - "Execution Error: ...": Process communication or other runtime errors
        """
        queue = Queue()

        # Create and start the isolated process using the module-level worker
        process = Process(target=_run_code_worker, args=(code_string, queue))
        process.start()

        # Wait for process completion with timeout
        process.join(timeout=timeout)

        # Check if process exceeded timeout
        if process.is_alive():
            # Process is still running - terminate it
            process.terminate()
            process.join(timeout=1)  # Give it 1 second to terminate gracefully

            # Force kill if graceful termination failed
            if process.is_alive():
                process.kill()
                process.join()

            return "Timeout Error"

        # Retrieve result from queue
        if queue.empty():
            return "Execution Error: No result returned from subprocess"

        status, message = queue.get()

        if status == "success":
            return message
        else:
            return f"Error: {message}"
