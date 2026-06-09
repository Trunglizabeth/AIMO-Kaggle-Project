import argparse
import pandas as pd
from datasets import load_dataset
import uuid


def prepare_aime_dataset(
    output_path: str = "data/train.csv",
    dataset_name: str = "gneubig/aime-1983-2024",
    split: str = "train",
    count: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    print("Connecting to Hugging Face...")
    print(f"Downloading dataset {dataset_name} split={split}...")

    dataset = load_dataset(dataset_name, split=split)
    dataset = dataset.shuffle(seed=seed).select(list(range(count)))

    rows = []
    for item in dataset:
        problem_text = item.get("problem", item.get("Question", ""))
        answer_text = item.get("answer", item.get("Answer", ""))

        rows.append({
            "id": str(uuid.uuid4())[:8],
            "problem": problem_text,
            "answer": str(answer_text).strip(),
        })

    df = pd.DataFrame(rows)
    output_path = output_path or "data/train.csv"
    df.to_csv(output_path, index=False)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Prepare the AIME dataset for AIMO project.')
    parser.add_argument('--output', default='data/train.csv', help='Output CSV path.')
    parser.add_argument('--dataset', default='gneubig/aime-1983-2024', help='Hugging Face dataset identifier.')
    parser.add_argument('--split', default='train', help='Dataset split name.')
    parser.add_argument('--count', type=int, default=50, help='Number of examples to extract.')
    parser.add_argument('--seed', type=int, default=42, help='Shuffle random seed.')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        df = prepare_aime_dataset(
            output_path=args.output,
            dataset_name=args.dataset,
            split=args.split,
            count=args.count,
            seed=args.seed,
        )
        print(f"\n✓ Successfully downloaded AIME dataset with {len(df)} problems!")
        print(f"✓ Columns: {list(df.columns)}")
        print(f"✓ Saved to {args.output}")
    except Exception as e:
        print(f"✗ Error: {e}")