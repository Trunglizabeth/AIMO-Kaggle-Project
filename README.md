# AIMO-Kaggle-Project

AI Mathematical Olympiad (AIMO) solver pipeline using Program-of-Thought (PoT).

This repository provides a PoC pipeline that:

- Sends math problems to an LLM (OpenAI or Google GenAI)
- Extracts Python code from the LLM response
- Safely executes the generated code in a sandboxed subprocess

Quickstart

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. (Optional) Set your LLM API key in the environment:

PowerShell (Windows):

```powershell
$env:LLM_API_KEY = "your_api_key"
```

3. Run the proof-of-concept runner:

```bash
python main_poc.py
```

Files of interest

- `src/engine_api.py`: LLM wrapper with `LLMEngineAPI` and `SYSTEM_PROMPT`.
- `src/parser.py`: `ResponseParser` with methods to extract Python code and boxed LaTeX answers.
- `src/executor.py`: `PythonExecutor` for safe execution with strict timeouts.
- `main_poc.py`: End-to-end PoC runner that prints each step for debugging.

Testing

Run unit tests with:

```bash
pytest -q
```

Notes

- The `data/` folder is ignored by git and intended for large datasets.
- On Windows, multiprocessing requires guarding script entry with `if __name__ == '__main__':`.
