# notebooks/test_engine.py
import sys
import os
sys.path.append(".")

from dotenv import load_dotenv
from src.engine import get_engine
from src.prompts import build_prompt
from src.voting import majority_vote

load_dotenv()

# Khởi tạo engine
engine = get_engine()

# Thử với 1 bài toán đơn giản
# Dùng N=5 thay vì 15 để tiết kiệm API call
problem = "Find the remainder when 2^100 is divided by 1000."

print(f"📝 Đề bài: {problem}")
print("=" * 50)

# Build prompt
messages = build_prompt(problem)
# Chuyển messages thành string cho GroqEngine
prompt_text = "\n".join([m["content"] for m in messages])

# Sinh 5 outputs
print("🤖 Đang sinh 5 outputs...")
all_outputs = engine.generate([prompt_text], n=5)
outputs = all_outputs[0]  # Lấy outputs của bài đầu tiên

print(f"\n📋 Các outputs nhận được:")
for i, out in enumerate(outputs):
    print(f"\n--- Output {i+1} ---")
    print(out)

print("\n" + "=" * 50)
print("✅ Xong! Tổng outputs:", len(outputs))