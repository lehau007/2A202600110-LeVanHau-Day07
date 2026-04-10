# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Lê Văn Hậu  
**Nhóm:** (cập nhật theo nhóm thực tế)  
**Ngày:** 10/4/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**  
Hai vector embedding có hướng gần nhau, tức là hai đoạn văn bản có nội dung ngữ nghĩa tương tự, dù có thể dùng từ khác nhau.

**Ví dụ HIGH similarity:**
- Câu A: Chí Phèo vừa đi vừa chửi.
- Câu B: Hắn vừa đi vừa chửi và say rượu.
- Lý do: cùng nhân vật, cùng hành vi và cùng ngữ cảnh.

**Ví dụ LOW similarity:**
- Câu A: Con mèo nằm ngủ trên mái nhà.
- Câu B: Hệ mặt trời có tám hành tinh.
- Lý do: khác chủ đề hoàn toàn.

**Vì sao cosine similarity thường phù hợp cho text embeddings hơn Euclidean distance?**  
Cosine similarity đo hướng vector thay vì độ lớn vector, nên ít bị ảnh hưởng bởi độ dài câu và phù hợp hơn khi so sánh nghĩa.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50:**  
num_chunks = ceil((10000 - 50) / (500 - 50)) = ceil(9950 / 450) = 23

**Nếu overlap = 100:**  
num_chunks = ceil((10000 - 100) / (500 - 100)) = ceil(9900 / 400) = 25

Overlap lớn hơn làm tăng số chunk nhưng giữ ngữ cảnh ở biên tốt hơn.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & lý do chọn

**Domain:** Văn học Việt Nam (truyện ngắn Nam Cao).  
Lý do: dữ liệu có chiều sâu ngữ nghĩa, phù hợp để kiểm tra retrieval theo nội dung và khả năng grounding.

### Data Inventory (bộ benchmark)

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | chi_pheo.txt | data nội bộ lab | 56,378 | source, extension, title, author, category, year |
| 2 | 1_bua_no.txt | data nội bộ lab | 15,811 | source, extension, title, author, category, year |
| 3 | quet_nha.txt | data nội bộ lab | 12,419 | source, extension, title, author, category, year |
| 4 | trang_sang.txt | data nội bộ lab | 15,929 | source, extension, title, author, category, year |
| 5 | tu_cach_mo.txt | data nội bộ lab | 11,769 | source, extension, title, author, category, year |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ | Vai trò |
|----------------|------|-------|---------|
| source | string | data/chi_pheo.txt | Truy vết nguồn chunk, dùng cho metadata filter |
| extension | string | .txt | Lọc theo loại file |
| title | string | Chí Phèo | Lọc theo tác phẩm |
| author | string | Nam Cao | Lọc theo tác giả |
| category | string | truyện ngắn | Lọc theo thể loại |
| year | string | 1941 | Lọc theo mốc thời gian |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy ChunkingStrategyComparator trên trang_sang.txt (chunk_size=400):

| Strategy | Chunk Count | Avg Length |
|----------|-------------|------------|
| fixed_size | 42 | 398.79 |
| sentence | 322 | 97.72 |
| recursive | 25 | 631.56 |

### So sánh Retrieval Strategy (benchmark thực tế)

Đã chạy script so sánh chiến lược chunking trên cùng 5 query benchmark:

| Strategy | Số chunk index | Retrieval hit rate | Avg keyword score |
|----------|----------------|--------------------|-------------------|
| fixed_size | 134 | 0.200 | 0.517 |
| sentence | 700 | 0.600 | 0.567 |
| recursive | 144 | 0.800 | 0.467 |

**Kết luận strategy:**
- Tốt nhất theo retrieval hit rate: recursive.
- Tốt nhất theo avg keyword score: sentence.

### Query cần metadata filtering (theo yêu cầu lab)

Đã thêm query mơ hồ và bắt buộc filter:
- Query: Nhân vật chính mở đầu truyện bằng hành động gì?
- Metadata filter: {"source": "data/chi_pheo.txt"}

Luồng benchmark và agent đã hỗ trợ filter qua search_with_filter.

---

## 4. My Approach — Cá nhân (10 điểm)

### Chunking

- Tích hợp chunking vào benchmark bằng build_chunk_documents.
- Mặc định benchmark dùng RecursiveChunker (chunk_size=1000, overlap=150).
- Có utility so sánh 3 chiến lược để chọn cấu hình retrieval phù hợp.

### EmbeddingStore

- add_documents lưu toàn bộ chunk + metadata.
- search trả về score và metadata.
- search_with_filter lọc metadata trước khi tính similarity.
- delete_document xóa toàn bộ chunk theo doc_id.

### KnowledgeBaseAgent

- answer(question, top_k, metadata_filter=None).
- Khi có metadata_filter, agent gọi search_with_filter để lấy context đúng phạm vi.

### Benchmark Pipeline

- Load 5 tài liệu.
- Enrich metadata domain (title/author/category/year).
- Chunk tài liệu trước khi add vào store.
- Chạy 5 benchmark cases (trong đó có 1 case dùng metadata filter).
- Chấm retrieval_hit_rate và avg_keyword_score.

### Test Results

- Tổng test pass: 44/44.
- Lệnh: python -m pytest -q

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|------------|------------|---------|--------------|-------|
| 1 | Chi Pheo vua di vua chui. | Han vua di vua chui va say ruou. | high | -0.0892 | Sai |
| 2 | Ba lao trong Mot bua no bi doi. | Trong Mot bua no, ba lao phai di xin an. | high | 0.1685 | Đúng |
| 3 | Dien suy nghi ve nghe thuat. | Dien cho rang nghe thuat phai gan voi dau kho doi song. | high | 0.0142 | Một phần |
| 4 | Con meo nam ngu tren mai nha. | He mat troi co tam hanh tinh. | low | -0.0547 | Đúng |
| 5 | Lo tu hien lanh thanh de tien. | Su khinh miet cua xa hoi lam Lo tha hoa. | high | -0.0948 | Sai |

**Nhận xét:** MockEmbedder là hash-based deterministic embedding nên không phản ánh ngữ nghĩa thật.

---

## 6. Results — Cá nhân (10 điểm)

### Benchmark Queries & Gold Answers

| # | Query | Gold Answer (tóm tắt) |
|---|-------|------------------------|
| 1 | Nhân vật chính mở đầu truyện bằng hành động gì? | Chí Phèo vừa đi vừa chửi (có filter source=chi_pheo) |
| 2 | Trong truyện Một bữa no, bà lão sống bằng cách nào khi tuổi già sức yếu? | Sống chật vật, xin ăn, làm việc vặt khi còn sức |
| 3 | Trong truyện Quét nhà, vì sao cha mẹ của Hồng hay cáu gắt? | Áp lực nghèo túng, nợ nần, căng thẳng sinh hoạt |
| 4 | Trong truyện Trăng sáng, Điền thay đổi quan niệm nghệ thuật thế nào? | Từ mơ mộng thoát ly sang nghệ thuật gắn đời sống |
| 5 | Trong truyện Tư cách mõ, Lộ bị biến đổi tính cách do tác động nào? | Bị khinh miệt, làm nhục, tha hóa dần |

### Kết quả benchmark thực tế (Gemini)

- LLM backend: gemini-3.1-flash-lite-preview
- num_docs_loaded: 5
- num_chunks_loaded: 144
- store_size: 144
- num_cases: 5
- chunking_strategy: RecursiveChunker
- retrieval_hit_rate: 0.8
- avg_keyword_score: 0.4667

| # | Retrieval hit? | Used metadata filter? | Keyword score |
|---|----------------|-----------------------|---------------|
| 1 | Yes | Yes | 0.3333 |
| 2 | Yes | No | 0.5000 |
| 3 | No | No | 0.0000 |
| 4 | Yes | No | 0.7500 |
| 5 | Yes | No | 0.7500 |

**Top-3 chứa đúng nguồn:** 4/5 query.

---

## 7. What I Learned (5 điểm — Demo)

### Failure case

- Query thất bại retrieval: Quét nhà (top-3 chưa chứa đúng nguồn trong lần chạy gần nhất).
- Nguyên nhân: nội dung nhiều truyện có motif xã hội gần nhau, dễ nhiễu semantic retrieval.
- Cải thiện: kết hợp metadata filter theo title hoặc author+title, và thêm bước rerank.

### Bài học chính

- Metadata filtering là bắt buộc khi query mơ hồ.
- Chunking strategy ảnh hưởng rõ rệt đến hit rate và chất lượng câu trả lời.
- Một metric đơn lẻ chưa đủ; cần nhìn đồng thời hit rate và quality answer.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 10 / 10 |
| Chunking strategy | Nhóm | 15 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 9 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 4 / 5 |
| **Tổng** | | **88 / 100 (theo kết quả hiện tại)** |
