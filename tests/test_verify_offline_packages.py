import tempfile
from pathlib import Path


def test_verify_offline_packages_import():
    from scripts.verify_offline_packages import main
    assert callable(main)


def test_find_wheel_files(tmp_path):
    from scripts.verify_offline_packages import find_wheel_files

    patterns = ['a-*.whl', 'b-*.whl']
    (tmp_path / 'a-1.whl').write_text('dummy')
    (tmp_path / 'b-2.whl').write_text('dummy')

    found = find_wheel_files(tmp_path, patterns)
    assert 'a-*.whl' in found
    assert 'b-*.whl' in found
    assert len(found['a-*.whl']) == 1
    assert len(found['b-*.whl']) == 1
