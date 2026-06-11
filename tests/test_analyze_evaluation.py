import pandas as pd
from pathlib import Path


def test_analyze_evaluation_import():
    from scripts.analyze_evaluation import summarize_report
    assert callable(summarize_report)


def test_categorize_error_types():
    from scripts.analyze_evaluation import categorize_error

    assert categorize_error('Timeout while executing') == 'Timeout'
    assert categorize_error('No code extracted from response') == 'No code extracted'
    assert categorize_error('Parsing predicted answer failed') == 'Parsing failure'
    assert categorize_error('Error: invalid syntax') == 'Execution error'
    assert categorize_error('Mock fallback due to missing model') == 'Mock fallback'
    assert categorize_error('Some other failure') == 'Other error'


def test_summarize_report(tmp_path):
    from scripts.analyze_evaluation import summarize_report

    df = pd.DataFrame([
        {'id': '1', 'problem': 'x^2=4', 'true_answer': '2', 'predicted_answer': '2', 'is_correct': True, 'error_log': ''},
        {'id': '2', 'problem': 'x+1=0', 'true_answer': '0', 'predicted_answer': None, 'is_correct': False, 'error_log': 'Timeout Error'},
    ])
    report_path = tmp_path / 'report.csv'
    df.to_csv(report_path, index=False)

    stats = summarize_report(report_path, top_errors=5, top_wrong=2)
    assert stats['total'] == 2
    assert stats['correct'] == 1
    assert stats['incorrect'] == 1
    assert stats['missing_predictions'] == 1
    assert stats['timeout_count'] == 1
    assert stats['error_categories']['Timeout'] == 1
    assert len(stats['top_wrong_examples']) == 1


def test_analyze_evaluation_save_outputs(tmp_path):
    from scripts.analyze_evaluation import summarize_report, save_markdown_summary

    df = pd.DataFrame([
        {'id': '1', 'problem': 'x^2=4', 'true_answer': '2', 'predicted_answer': '2', 'is_correct': True, 'error_log': ''},
        {'id': '2', 'problem': 'x+1=0', 'true_answer': '0', 'predicted_answer': None, 'is_correct': False, 'error_log': 'Timeout Error'},
    ])
    report_path = tmp_path / 'report.csv'
    json_path = tmp_path / 'summary.json'
    md_path = tmp_path / 'summary.md'
    df.to_csv(report_path, index=False)

    stats = summarize_report(report_path, top_errors=5, top_wrong=2)

    with json_path.open('w', encoding='utf-8') as f:
        import json
        json.dump(stats, f)

    save_markdown_summary(stats, md_path)

    assert json_path.exists()
    assert md_path.exists()
    content = md_path.read_text(encoding='utf-8')
    assert 'Evaluation Summary' in content
    assert '- Total examples: 2' in content
