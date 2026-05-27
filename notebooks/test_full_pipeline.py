# notebooks/test_full_pipeline.py
import sys
import re
import os
sys.path.append(".")

from dotenv import load_dotenv
from src.engine import get_engine
from src.prompts import build_prompt
from src.voting import majority_vote
import pandas as pd

load_dotenv()
engine = get_engine()

# =====================
# HÀM BÓC TÁCH CODE (tạm thời, chờ Trung xong parser.py)
# =====================
def extract_code(text):
    match = re.search(r"```python\s*(.*?)\s*```", text, re.DOTALL)
    return match.group(1).strip() if match else None

def run_code(code):
    if code is None:
        return "Error: No code"
    try:
        local_env = {}
        exec(code, local_env)
        return int(local_env["solve"]())
    except Exception as e:
        return f"Error: {e}"

# =====================
# PIPELINE CHÍNH
# =====================
def solve_problem(problem: str, n: int = 5) -> int:
    messages = build_prompt(problem)
    prompt_text = "\n".join([m["content"] for m in messages])

    # Sinh n outputs
    all_outputs = engine.generate([prompt_text], n=n)
    outputs = all_outputs[0]

    # Bóc tách và chạy code
    results = []
    for out in outputs:
        code = extract_code(out)
        result = run_code(code)
        results.append(result)

    # Majority voting
    return majority_vote(results)

# =====================
# TEST VỚI reference.csv
# =====================
df = pd.read_csv("data/raw/reference.csv")

# Chỉ test 3 bài đầu để tiết kiệm API
correct = 0
total = 3

print("🧪 Test pipeline với reference.csv\n")
for i in range(total):
    problem = df["problem"][i]
    true_answer = df["answer"][i]

    print(f"📝 Bài {i+1}: {problem[:80]}...")
    predicted = solve_problem(problem, n=5)

    is_correct = predicted == true_answer
    if is_correct:
        correct += 1

    print(f"  🎯 Dự đoán: {predicted} | Đúng: {true_answer} | {'✅' if is_correct else '❌'}")
    print()

print("=" * 50)
print(f"📊 Kết quả: {correct}/{total} = {correct/total*100:.0f}%")