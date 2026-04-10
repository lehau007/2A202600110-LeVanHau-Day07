# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Lê Văn Hậu  
**Nhóm:** (cập nhật theo nhóm thực tế)  
**Ngày:** 10/4/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**  
Hai vector embedding có hướng gần nhau, tức là hai đoạn văn bản có nội dung ngữ nghĩa tương tự, dù có thể khác từ vựng.

**Ví dụ HIGH similarity:**
- Câu A: Chí Phèo vừa đi vừa chửi.
- Câu B: Hắn vừa đi vừa chửi và say rượu.
- Lý do: Cùng mô tả một hành vi, cùng nhân vật, cùng bối cảnh ngữ nghĩa.

**Ví dụ LOW similarity:**
- Câu A: Con mèo nằm ngủ trên mái nhà.
- Câu B: Hệ mặt trời có tám hành tinh.
- Lý do: Khác hẳn chủ đề, không liên quan ngữ nghĩa.

**Vì sao cosine similarity thường phù hợp cho text embeddings hơn Euclidean distance?**  
Cosine similarity đo hướng thay vì độ lớn vector, nên bền vững hơn khi độ dài câu khác nhau. Điều này phù hợp với mục tiêu so sánh ý nghĩa hơn là so sánh độ dài biểu diễn.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50: số chunk?**  
Công thức:

num_chunks = ceil((doc_length - overlap) / (chunk_size - overlap))

Thay số:

num_chunks = ceil((10000 - 50) / (500 - 50)) = ceil(9950 / 450) = ceil(22.11) = 23

**Nếu overlap = 100 thì sao?**  
num_chunks = ceil((10000 - 100) / (500 - 100)) = ceil(9900 / 400) = ceil(24.75) = 25

Overlap tăng thì số chunk tăng. Đổi lại, ngữ cảnh ở biên chunk được giữ tốt hơn, giúp retrieval chính xác hơn.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & lý do chọn

**Domain:** Văn học Việt Nam (Nam Cao)  
Nhóm chọn domain này vì dữ liệu có chiều sâu nội dung, ngôn ngữ giàu ngữ cảnh, phù hợp để đánh giá retrieval theo chất lượng diễn giải thay vì chỉ keyword đơn giản.

### Data Inventory (bộ đã benchmark)

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | chi_pheo.txt | data nội bộ lab | 56,378 | source, extension |
| 2 | 1_bua_no.txt | data nội bộ lab | 15,811 | source, extension |
| 3 | quet_nha.txt | data nội bộ lab | 12,419 | source, extension |
| 4 | trang_sang.txt | data nội bộ lab | 15,929 | source, extension |
| 5 | tu_cach_mo.txt | data nội bộ lab | 11,769 | source, extension |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ | Vai trò |
|----------------|------|-------|---------|
| source | string | data\\chi_pheo.txt | Truy vết nguồn chunk để kiểm tra grounding |
| extension | string | .txt | Hỗ trợ lọc theo loại tài liệu |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy ChunkingStrategyComparator trên tài liệu trang_sang.txt với chunk_size=400:

| Strategy | Chunk Count | Avg Length |
|----------|-------------|------------|
| fixed_size | 42 | 398.79 |
| sentence | 322 | 97.72 |
| by_sentences | 322 | 97.72 |
| recursive | 25 | 631.56 |

### Strategy tôi chọn cho benchmark retrieval

Trong benchmark hiện tại, mỗi truyện được index như một document đầy đủ (doc-level indexing) để kiểm tra nhanh mức độ định vị tài liệu đúng nguồn trước khi tinh chỉnh chunking sâu hơn.

**Lý do chọn cách này cho vòng benchmark đầu:**
- Dễ kiểm soát đúng/sai theo nguồn truyện.
- Giảm nhiễu do nhiều siêu tham số chunking trong lần chạy chuẩn đầu tiên.
- Phù hợp mục tiêu xác nhận pipeline end-to-end: add dữ liệu, search, gọi LLM, chấm kết quả.

### So sánh với baseline

So với chunk-level retrieval, doc-level retrieval cho phép đánh giá nhanh hit-rate theo truyện. Tuy nhiên, độ chính xác chi tiết trong từng đoạn văn sẽ kém hơn khi câu hỏi yêu cầu đoạn rất cụ thể.

### So sánh với thành viên khác

Phần này sẽ cập nhật thêm khi tổng hợp đủ kết quả từ các thành viên chạy cùng bộ query trên strategy khác nhau.

---

## 4. My Approach — Cá nhân (10 điểm)

### Chunking Functions

- SentenceChunker: tách câu bằng regex, gom theo số câu tối đa và overlap theo câu.
- RecursiveChunker: ưu tiên tách theo cấu trúc lớn đến nhỏ, sau đó merge bằng sliding window.
- compute_similarity: cosine similarity với guard tránh chia cho 0.

### EmbeddingStore

- add_documents: lưu embedding + metadata cho từng document.
- search: trả về id, content, score, metadata; sắp xếp giảm dần theo score.
- search_with_filter: lọc metadata trước rồi mới tính similarity.
- delete_document: xóa toàn bộ record theo id.

### KnowledgeBaseAgent

- answer: retrieve top-k chunks/docs, tạo prompt dạng Context + Question, gọi LLM để trả lời.

### Benchmark Pipeline mới

Đã thêm pipeline benchmark riêng:
- Tải 5 truyện trong data.
- Add vào EmbeddingStore.
- Chạy 5 câu hỏi benchmark.
- Gọi LLM Gemini model: gemini-3.1-flash-lite-preview.
- Tính retrieval_hit_rate và avg_keyword_score.

### Test Results

- Tổng test pass: 43/43.
- Thời gian chạy gần nhất: 0.16s.

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

Sử dụng MockEmbedder để tính actual score:

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|------------|------------|---------|--------------|-------|
| 1 | Chi Pheo vua di vua chui. | Han vua di vua chui va say ruou. | high | -0.0892 | Sai |
| 2 | Ba lao trong Mot bua no bi doi. | Trong Mot bua no, ba lao phai di xin an. | high | 0.1685 | Đúng |
| 3 | Dien suy nghi ve nghe thuat. | Dien cho rang nghe thuat phai gan voi dau kho doi song. | high | 0.0142 | Một phần |
| 4 | Con meo nam ngu tren mai nha. | He mat troi co tam hanh tinh. | low | -0.0547 | Đúng |
| 5 | Lo tu hien lanh thanh de tien. | Su khinh miet cua xa hoi lam Lo tha hoa. | high | -0.0948 | Sai |

**Nhận xét:**  
MockEmbedder là deterministic hash embedding, không học ngữ nghĩa thật, nên điểm similarity có thể lệch trực giác con người.

---

## 6. Results — Cá nhân (10 điểm)

### Benchmark Queries & Gold Answers (đã dùng khi chạy benchmark)

| # | Query | Gold Answer (tóm tắt) |
|---|-------|------------------------|
| 1 | Trong Chí Phèo, nhân vật Chí Phèo mở đầu truyện bằng hành động gì? | Vừa đi vừa chửi trong trạng thái say rượu, bối cảnh làng Vũ Đại |
| 2 | Trong truyện Một bữa no, bà lão sống bằng cách nào khi tuổi già sức yếu? | Làm việc vặt, xin ăn, sống chật vật trong đói nghèo |
| 3 | Trong truyện Quét nhà, vì sao cha mẹ của Hồng hay cáu gắt trong giai đoạn khó khăn? | Áp lực nghèo túng, nợ nần, căng thẳng sinh hoạt |
| 4 | Trong truyện Trăng sáng, Điền thay đổi quan niệm nghệ thuật như thế nào? | Từ mơ mộng thoát ly sang nghệ thuật gắn với đời sống khổ đau |
| 5 | Trong truyện Tư cách mõ, Lộ bị biến đổi tính cách do những tác động xã hội nào? | Bị khinh miệt, làm nhục, tha hóa dần trong môi trường làng xã |

### Kết quả chạy benchmark thực tế (Gemini)

- LLM backend: gemini-3.1-flash-lite-preview
- num_docs_loaded: 5
- store_size: 5
- num_cases: 5
- retrieval_hit_rate: 0.8
- avg_keyword_score: 0.8167

| # | Retrieval hit? | Retrieved sources (top-3) | Keyword score |
|---|----------------|---------------------------|---------------|
| 1 | Yes | chi_pheo, trang_sang, tu_cach_mo | 0.3333 |
| 2 | Yes | quet_nha, tu_cach_mo, 1_bua_no | 0.7500 |
| 3 | Yes | trang_sang, quet_nha, tu_cach_mo | 1.0000 |
| 4 | No | 1_bua_no, chi_pheo, quet_nha | 1.0000 |
| 5 | Yes | trang_sang, tu_cach_mo, chi_pheo | 1.0000 |

**Top-3 chứa tài liệu đúng nguồn:** 4/5 query.

---

## 7. What I Learned (5 điểm — Demo)

### Failure case chính

- Query: Trăng sáng, Điền thay đổi quan niệm nghệ thuật như thế nào?
- Vấn đề: top-3 retrieval không chứa đúng nguồn trang_sang trong lần chạy này.
- Dù vậy, câu trả lời LLM vẫn đúng ý do kiến thức văn học tổng quát và ngữ cảnh từ tài liệu khác có nội dung gần.

### Nguyên nhân khả dĩ

- Đang index theo doc-level, chưa chunk theo đoạn để tách rõ các luận điểm nghệ thuật.
- Embedding backend hiện tại chưa chuyên cho tiếng Việt văn học dài.

### Hướng cải thiện

- Chuyển sang chunk-level indexing với RecursiveChunker.
- Bổ sung metadata theo tác phẩm/nhân vật/chủ đề để filter trước retrieval.
- Thử reranker sau bước vector search để tăng độ chính xác top-1.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 10 / 10 |
| Chunking strategy | Nhóm | 13 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 9 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 4 / 5 |
| **Tổng** | | **86 / 100 (tạm thời theo kết quả hiện có)** |
