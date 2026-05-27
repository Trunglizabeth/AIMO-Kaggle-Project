"""Parser module for extracting structured information from LLM outputs."""

import re
from typing import Optional


class ResponseParser:
    """
    Robust parser for extracting structured information from LLM-generated text.

    This parser provides static methods to extract specific formats from raw LLM output,
    including Python code blocks and LaTeX-formatted answers. It handles edge cases
    like multiple code blocks, nested braces, and malformed formatting.

    Example:
        >>> text = '''
        ... Here's my solution:
        ... ```python
        ... x = 5
        ... result = x ** 2
        ... ```
        ... The answer is \\boxed{25}
        ... '''
        >>> code = ResponseParser.extract_python_code(text)
        >>> answer = ResponseParser.extract_boxed_answer(text)
    """

    @staticmethod
    def extract_python_code(text: str) -> str:
        """
        Extract Python code from markdown code blocks in LLM output.

        This method searches for standard markdown python code blocks (```python ... ```).
        If multiple blocks are found, the last one is extracted. If no block is found,
        the method attempts to detect if the text contains code-like content and returns
        it after stripping formatting, or returns an empty string if it doesn't appear
        to be code.

        Args:
            text (str): Raw text from LLM containing potential python code blocks.

        Returns:
            str: The extracted Python code (last block if multiple),
                 or an empty string if no code block is found or text doesn't look like code.

        Example:
            >>> text = '''
            ... Solution:
            ... ```python
            ... x = 10
            ... result = x * 2
            ... ```
            ... '''
            >>> code = ResponseParser.extract_python_code(text)
            >>> print(code)
            x = 10
            result = x * 2
        """
        # Pattern to match markdown python code blocks
        # Using DOTALL flag to match newlines with .
        pattern = r'```python\s*(.*?)\s*```'
        matches = re.findall(pattern, text, re.DOTALL)

        if matches:
            # Return the last code block if multiple blocks exist
            return matches[-1].strip()

        # No markdown code blocks found - try to detect if text looks like code
        # If the LLM returned an API error message, treat it as no-code
        error_indicators = [
            'Error calling LLM API',
            'APIRemovedInV1',
            'You tried to access openai.ChatCompletion',
        ]
        if any(e in cleaned_text for e in error_indicators):
            return ""
        # Remove common markdown formatting characters
        cleaned_text = text.strip()

        # Heuristics to detect if text looks like Python code
        code_indicators = [
            'def ',
            'class ',
            'import ',
            'from ',
            'result =',
            'return ',
            'for ',
            'while ',
            'if ',
            'try:',
            'except',
        ]

        # Check if text contains any code indicators
        if any(indicator in cleaned_text for indicator in code_indicators):
            return cleaned_text

        # Doesn't look like code, return empty string
        return ""

    @staticmethod
    def extract_boxed_answer(text: str) -> str:
        """
        Extract the answer from a LaTeX \\boxed{} tag in LLM output.

        This method uses regex to find and extract content inside LaTeX \\boxed{} tags.
        It handles nested curly braces by matching balanced braces, allowing for
        complex expressions like \\boxed{{x: 5, y: 10}} or \\boxed{\\frac{1}{2}}.

        Args:
            text (str): Raw text from LLM containing LaTeX formatting.

        Returns:
            str: The content inside the \\boxed{} tag (without the \boxed{} wrapper),
                 or an empty string if no \\boxed{} tag is found.

        Example:
            >>> text = "The final answer is \\boxed{42}"
            >>> answer = ResponseParser.extract_boxed_answer(text)
            >>> print(answer)
            42

            >>> text = "Result: \\boxed{\\frac{22}{7}}"
            >>> answer = ResponseParser.extract_boxed_answer(text)
            >>> print(answer)
            \\frac{22}{7}
        """
        # Pattern to match \boxed{...} with balanced brace handling
        # This pattern uses a recursive-like approach with re module limitations
        # We'll use a more robust method: find \boxed{ and manually parse balanced braces

        # First, try to find the \boxed{ pattern
        pattern = r'\\boxed\s*\{'
        match = re.search(pattern, text)

        if not match:
            # No \boxed{ found
            return ""

        # Start position after \boxed{
        start_pos = match.end()

        # Manually parse balanced braces starting from start_pos
        brace_count = 1
        end_pos = start_pos

        for i in range(start_pos, len(text)):
            char = text[i]

            # Skip escaped characters
            if i > 0 and text[i - 1] == '\\':
                continue

            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1

                if brace_count == 0:
                    end_pos = i
                    break

        # Extract the content between balanced braces
        if brace_count == 0:
            content = text[start_pos:end_pos].strip()
            return content
        else:
            # Unbalanced braces - return empty string or partial match
            return ""
