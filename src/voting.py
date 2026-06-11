# src/voting.py
from statistics import mode
from collections import Counter
from src.config import config


def majority_vote(results: list) -> int:
    """
    Nhận list kết quả từ nhiều lần chạy code,
    trả về đáp án xuất hiện nhiều nhất.
    
    - Lọc bỏ các kết quả lỗi (string)
    - Nếu tất cả đều lỗi → trả về 0 (fallback)
    """
    # Lọc chỉ giữ kết quả là số nguyên
    valid = [r for r in results if isinstance(r, int)]

    print(f"  📊 Tổng: {len(results)} | Hợp lệ: {len(valid)} | Lỗi: {len(results) - len(valid)}")

    if not valid:
        print("  ⚠️ Tất cả đều lỗi! Trả về 0 (fallback)")
        return 0

    # Đếm tần suất
    counter = Counter(valid)
    most_common = counter.most_common(3)
    print(f"  🗳️ Top kết quả: {most_common}")

    # Lấy kết quả xuất hiện nhiều nhất
    final = most_common[0][0]
    return final