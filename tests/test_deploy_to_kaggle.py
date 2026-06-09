import os
from pathlib import Path


def test_deploy_to_kaggle_import():
    from scripts.deploy_to_kaggle import main
    assert callable(main)


def test_find_wheel_files(tmp_path):
    from scripts.deploy_to_kaggle import find_wheel_files

    (tmp_path / 'vllm-1.whl').write_text('dummy')
    (tmp_path / 'sympy-1.whl').write_text('dummy')
    patterns = ['vllm-*.whl', 'sympy-*.whl']

    found = find_wheel_files(tmp_path, patterns)
    assert len(found['vllm-*.whl']) == 1
    assert len(found['sympy-*.whl']) == 1
    assert found['vllm-*.whl'][0].name == 'vllm-1.whl'
