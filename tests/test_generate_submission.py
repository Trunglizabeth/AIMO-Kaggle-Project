import os
from pathlib import Path


def test_generate_submission_import():
    from scripts.generate_submission import main

    assert callable(main)
