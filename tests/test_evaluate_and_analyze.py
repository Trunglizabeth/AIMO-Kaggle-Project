from pathlib import Path


def test_evaluate_and_analyze_import():
    from scripts.evaluate_and_analyze import main

    assert callable(main)


def test_evaluate_and_analyze_calls_substeps(monkeypatch, tmp_path):
    from scripts import evaluate_and_analyze

    called = {
        'evaluate': False,
        'analyze': False,
        'markdown': False,
        'submit': False,
    }

    def fake_evaluate(**kwargs):
        called['evaluate'] = True

    def fake_summarize(report_path, top_errors, top_wrong):
        called['analyze'] = True
        return {
            'total': 0,
            'correct': 0,
            'incorrect': 0,
            'accuracy': 0.0,
            'missing_predictions': 0,
            'timeout_count': 0,
            'error_counts': {},
            'error_categories': {},
            'top_wrong_examples': [],
        }

    def fake_save_markdown_summary(stats, output_path):
        called['markdown'] = True

    def fake_generate_main(argv=None):
        called['submit'] = True
        assert '--csv' in argv
        assert '--output' in argv

    monkeypatch.setattr(evaluate_and_analyze.evaluate_module, 'evaluate', fake_evaluate)
    monkeypatch.setattr(evaluate_and_analyze.analyze_module, 'summarize_report', fake_summarize)
    monkeypatch.setattr(evaluate_and_analyze.analyze_module, 'save_markdown_summary', fake_save_markdown_summary)
    monkeypatch.setattr(evaluate_and_analyze.generate_module, 'main', fake_generate_main)

    report_path = tmp_path / 'evaluation_report.csv'
    output_summary = tmp_path / 'report_summary.json'
    output_markdown = tmp_path / 'report_summary.md'

    evaluate_and_analyze.main([
        '--csv', 'data/train.csv',
        '--report', str(report_path),
        '--output-summary', str(output_summary),
        '--output-markdown', str(output_markdown),
        '--generate-submission',
    ])

    assert called['evaluate']
    assert called['analyze']
    assert called['markdown']
    assert called['submit']
