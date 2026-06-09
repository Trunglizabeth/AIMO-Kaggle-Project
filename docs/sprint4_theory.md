# Cơ sở lý thuyết — Sprint 4

Mục tiêu: giải thích vì sao chọn kiến trúc hiện tại cho pipeline giải toán số học bằng LLM, nêu hạn chế nền tảng của LLM với toán học, giới thiệu Program-of-Thought (PoT) và vai trò của KV Cache trong vLLM.

## 1. Hạn chế của LLM khi giải toán

- Sự không nhất quán số học: LLM là mô hình sinh ngôn ngữ học xác suất; nó tối ưu cho độ mượt văn bản, không đảm bảo tính chính xác số học. Lỗi gồm: làm tròn sai, mất/đổi dấu, sai trong các bước trung gian.
- Thiếu tính toán biểu thức chính xác: LLM không thực thi tính toán nội tại — nó dự đoán token, nên các phép tính phức tạp dễ bị sai; việc theo dõi trạng thái số học qua nhiều bước là yếu.
- Rủi ro hallucination: tạo ra biểu thức hoặc số không có cơ sở từ dữ liệu vào.
- Khó debug: output dạng ngôn ngữ kết hợp với latex/code cần tách, kiểm tra và thực thi để xác minh kết quả.

Kết luận: Để đạt độ chính xác cao cho bài toán dạng AIME, cần kết hợp LLM với môi trường tính toán (exec) để kiểm chứng kết quả.

## 2. Program-of-Thought (PoT)

- Ý tưởng: thay vì yêu cầu LLM trực tiếp trả lời "42", ta yêu cầu LLM sinh một chương trình (mã Python) mô tả các bước tính toán. Sau đó ta thực thi chương trình trong sandbox, thu kết quả thực tế.
- Lợi ích:
  - Chuyển trách nhiệm tính toán từ mô hình sang máy (deterministic execution).
  - Dễ dàng kiểm tra và tái sử dụng các bước trung gian.
  - Hỗ trợ voting/ensemble: chạy nhiều lần/đa mô hình, so sánh outputs.
- Rủi ro và biện pháp:
  - Mã không an toàn: dùng sandbox/subprocess với timeout, chroot-like restrictions (nếu cần), và giới hạn import.
  - Lỗi cú pháp: parser phải robust để tách code từ text và xử lý fallback (raw numeric extraction).

## 3. KV Cache và vLLM

- KV Cache (Key-Value cache) được vLLM dùng để lưu trữ trạng thái attention giữa các token, giúp phục vụ cho streaming/decoding tốc độ cao và reuse context cho multi-turn/beam search.
- Vai trò với pipeline:
  - Khi cần gọi LLM nhiều lần (ví dụ majority voting, hoặc incremental reasoning), KV cache giảm chi phí tính toán cho phần context không đổi.
  - Hữu ích khi fine-grained decoding hoặc khi tái sử dụng tiền đề chung (prompt template) cho nhiều ví dụ.
- Hạn chế: KV cache tăng bộ nhớ; cần cân bằng kích thước batch/sequence để tránh OOM.

## 4. Lý do chọn kiến trúc hiện tại

- PoT + sandbox execution: cho phép tách phần "suy luận" (mã) và "kiểm chứng" (thực thi) — tăng tính chính xác và minh bạch.
- Multiple prediction methods (code execution, raw answer fallback, majority voting): tăng robustess khi LLM thất bại ở một kênh.
- Dùng vLLM/OpenAI/GoogleGenAI adapter: duy trì khả năng thay thế engine, tận dụng KV cache khi dùng vLLM trong môi trường offline hoặc self-hosted.

---

Tiếp theo: sơ đồ pipeline (file riêng `docs/pipeline_diagram.mmd`).
