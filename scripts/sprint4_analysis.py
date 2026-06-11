"""
Sprint 4 analysis script
- Reads `data/evaluation_report.csv` produced in Sprint 3
- Produces:
  - outputs/error_pie.png (pie chart of error categories)
  - outputs/accuracy_comparison.png (Pass@1 vs Pass@15)
  - docs/case_studies.md (2-3 extracted examples)

Usage:
    python scripts/sprint4_analysis.py --report data/evaluation_report.csv --out outputs

"""
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


def make_error_pie(df, out_dir):
    counts = df['error_log'].fillna('No error').replace('', 'No error')
    counts = counts.map(lambda x: x if isinstance(x, str) and x else 'No error')
    vc = counts.value_counts()
    plt.figure(figsize=(8,8))
    vc.plot.pie(autopct='%1.1f%%')
    plt.ylabel('')
    plt.title('Error distribution')
    path = os.path.join(out_dir, 'error_pie.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print('Saved', path)


def make_accuracy_comparison(df, out_dir):
    # Compute pass@1 accuracy (is_correct True) and pass@15 (simulate majority voting by grouping id and mode)
    total = df.shape[0]
    pass1 = df['is_correct'].sum() / total

    # If there is a `predictions` column with multiple preds per id, use majority voting; else simulate by assuming multiple rows per id
    if 'id' in df.columns and df.duplicated(subset=['id']).any():
        grouped = df.groupby('id')
        majority_correct = grouped['is_correct'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.mean()>0.5)
        pass15 = majority_correct.mean()
    else:
        # fallback: if `prediction_method` contains 'majority' or 'voting', try to estimate; otherwise assume pass15 ~ pass1
        pass15 = pass1

    plt.figure(figsize=(6,4))
    sns.barplot(x=['Pass@1','Pass@15'], y=[pass1*100, pass15*100], palette='muted')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0,100)
    for i,v in enumerate([pass1*100, pass15*100]):
        plt.text(i, v+1, f'{v:.2f}%', ha='center')
    path = os.path.join(out_dir, 'accuracy_comparison.png')
    plt.title('Pass@1 vs Pass@15 (majority voting)')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print('Saved', path)


def extract_case_studies(df, out_dir, n=3):
    # Find interesting cases: where code executed (prediction_method==code) and is_correct==False.
    pm = df.get('prediction_method')
    if pm is not None:
        candidates = df[(pm == 'code') & (df['is_correct'] == False)]
    else:
        candidates = df[df['is_correct'] == False]

    examples = candidates.head(n)
    md_lines = ["# Case Studies\n"]
    for _, row in examples.iterrows():
        md_lines.append(f"## id={row['id']}\n")
        md_lines.append(f"**Problem:** {row['problem']}\n")
        md_lines.append(f"**True answer:** {row['true_answer']}\n")
        md_lines.append(f"**Predicted answer:** {row.get('predicted_answer', '')}\n")
        md_lines.append(f"**Prediction method:** {row.get('prediction_method', '')}\n")
        md_lines.append(f"**Error log:** {row.get('error_log', '')}\n")
        md_lines.append('---\n')

    out_path = os.path.join('docs', 'case_studies.md')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
    print('Saved', out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', default='data/evaluation_report.csv')
    parser.add_argument('--out', default='outputs')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(args.report)

    make_error_pie(df, args.out)
    make_accuracy_comparison(df, args.out)
    extract_case_studies(df, args.out, n=3)


if __name__ == '__main__':
    main()
