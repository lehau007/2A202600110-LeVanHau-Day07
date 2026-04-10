# Danh sách công việc Lab 7: Nền Tảng Dữ Liệu: Embedding & Vector Store

Dựa trên cấu trúc của bài lab, đây là danh sách các công việc chi tiết bạn cần thực hiện. Bài lab chia làm 2 giai đoạn (Phase 1: Cá nhân, Phase 2: Nhóm nhưng có phần việc cá nhân).

## Phase 1: Hoàn thành Core Coding (Làm cá nhân)

Mục tiêu của phần này là implement tất cả các TODO trong các file ở thư mục `src/`. Các test trong `tests/` hiện tại đang fail và mục tiêu là làm cho `pytest tests/ -v` pass toàn bộ.

- [ ] **Trong `src/chunking.py`:**
  - [ ] Implement `SentenceChunker`: Tách văn bản theo câu và nhóm thành các chunk.
  - [ ] Implement `RecursiveChunker`: Thử tách theo các separator theo thứ tự ưu tiên, đệ quy trên các đoạn văn bản quá dài.
  - [ ] Implement `compute_similarity`: Tính cosine similarity với cơ chế xử lý zero-magnitude.
  - [ ] Implement `ChunkingStrategyComparator`: Gọi 3 chiến lược chunking và tính toán các thống kê (stats).
- [ ] **Trong `src/store.py`:**
  - [ ] Implement `EmbeddingStore.__init__`: Khởi tạo store (in-memory hoặc ChromaDB).
  - [ ] Implement `EmbeddingStore.add_documents`: Embed và lưu trữ từng document.
  - [ ] Implement `EmbeddingStore.search`: Embed query và rank kết quả bằng dot product.
  - [ ] Implement `EmbeddingStore.get_collection_size`: Trả về số lượng document trong store.
  - [ ] Implement `EmbeddingStore.search_with_filter`: Lọc theo metadata trước khi search.
  - [ ] Implement `EmbeddingStore.delete_document`: Xóa toàn bộ chunk thuộc về một `doc_id`.
- [ ] **Trong `src/agent.py`:**
  - [ ] Implement `KnowledgeBaseAgent.answer`: Thực hiện RAG (Retrieve -> Xây dựng prompt -> Gọi LLM).

## Phase 2: So Sánh Retrieval Strategy (Làm việc Nhóm & Cá nhân)

Phần này yêu cầu làm việc với dữ liệu thật, chạy thử nghiệm các chiến lược chunking và báo cáo kết quả.

- [ ] **Exercise 1.1 & 1.2 (Warm-up):**
  - [ ] Trả lời câu hỏi lý thuyết về Cosine Similarity (Không dùng toán). Ghi vào Section 1 của Report.
  - [ ] Tính toán số lượng chunk và giải thích lý do dùng overlap. Ghi vào Section 1 của Report.
- [ ] **Exercise 3.0: Chuẩn bị tài liệu:**
  - [ ] Chọn một domain (ví dụ: FAQ, SOP, ...).
  - [ ] Thu thập 5-10 tài liệu (.txt hoặc .md) và đưa vào thư mục `data/`.
  - [ ] Thiết kế metadata schema (ít nhất 2 trường hữu ích). Ghi vào Section 2 của Report.
- [ ] **Exercise 3.1: Thiết kế Retrieval Strategy:**
  - [ ] Chạy baseline với `ChunkingStrategyComparator`.
  - [ ] Chọn và tinh chỉnh chiến lược (built-in với tham số tối ưu hoặc tự viết `CustomChunker`).
  - [ ] Ghi lại strategy của bạn vào Section 3 của Report.
- [ ] **Exercise 3.2: Chuẩn bị Benchmark Queries:**
  - [ ] Thống nhất 5 benchmark queries với "gold answers" (câu trả lời chuẩn). (Yêu cầu ít nhất 1 câu cần lọc metadata).
  - [ ] Ghi danh sách này vào Section 6 của Report.
- [ ] **Exercise 3.3: Cosine Similarity Predictions:**
  - [ ] Chọn 5 cặp câu.
  - [ ] Dự đoán cặp nào có similarity cao/thấp nhất trước khi chạy code.
  - [ ] Chạy `compute_similarity()` và ghi kết quả thực tế. Nhận xét sự bất ngờ nếu có. Ghi vào Section 5 của Report.
- [ ] **Exercise 3.4: Chạy Benchmark & So Sánh:**
  - [ ] Chạy 5 benchmark queries bằng strategy đã chọn, lấy kết quả top 3.
  - [ ] So sánh với các thành viên khác: Strategy nào tốt nhất? Tại sao? Metadata có giúp ích không?
  - [ ] Ghi kết quả vào Section 6 của Report.
- [ ] **Exercise 3.5: Failure Analysis:**
  - [ ] Xác định ít nhất 1 trường hợp thất bại (failure case) khi retrieval.
  - [ ] Phân tích lý do (chunk quá to/nhỏ, thiếu metadata...) và đề xuất cách cải thiện. Ghi vào Section 7 của Report.

## Tổng Kết và Nộp Bài

- [ ] Chạy `pytest tests/ -v` và đảm bảo 100% tests pass.
- [ ] Điền đầy đủ thông tin vào file báo cáo cá nhân: `report/REPORT.md`.
- [ ] Nộp thư mục `src/` (code cá nhân) và `report/REPORT.md`.