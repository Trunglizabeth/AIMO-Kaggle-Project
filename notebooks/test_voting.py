# notebooks/test_voting.py
import sys
sys.path.append(".")

from src.voting import majority_vote

# Test 1: Đa số đúng
results1 = [376, 376, 376, 999, 123, "Error: Timeout", 376, 999]
print("Test 1:", majority_vote(results1))  # Kỳ vọng: 376

# Test 2: Tất cả lỗi
results2 = ["Error: Syntax", "Error: Timeout", "Error: ZeroDivision"]
print("Test 2:", majority_vote(results2))  # Kỳ vọng: 0

# Test 3: Hòa nhau
results3 = [100, 100, 200, 200, "Error"]
print("Test 3:", majority_vote(results3))  # Kỳ vọng: 100 hoặc 200