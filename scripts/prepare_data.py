import pandas as pd
from datasets import load_dataset
import uuid

def prepare_aime_dataset(output_path="data/train.csv"):
    print("Connecting to Hugging Face...")
    print("Downloading AIME 1983-2024 dataset...")
    
    # Tải bộ AIME (Không bị dính bản quyền như bộ MATH)
    dataset = load_dataset("gneubig/aime-1983-2024", split="train")
    
    # Trộn ngẫu nhiên và lấy 50 bài
    dataset = dataset.shuffle(seed=42).select(range(50))
    
    rows = []
    for item in dataset:
        # Lấy text câu hỏi và đáp án (tự động nhận diện tên cột)
        problem_text = item.get("problem", item.get("Question", ""))
        answer_text = item.get("answer", item.get("Answer", ""))
        
        rows.append({
            "id": str(uuid.uuid4())[:8],
            "problem": problem_text,
            "answer": str(answer_text).strip()
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return df

if __name__ == "__main__":
    try:
        df = prepare_aime_dataset()
        print(f"\n✓ Successfully downloaded AIME dataset with {len(df)} problems!")
        print(f"✓ Columns: {list(df.columns)}")
        print("✓ Saved to data/train.csv")
    except Exception as e:
        print(f"✗ Error: {e}")