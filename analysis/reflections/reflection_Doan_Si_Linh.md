# Báo cáo cá nhân
### Đoàn Sĩ Linh (AI Engineer)
### MSV: 2A202600363

## 1. Vai trò và mục tiêu cá nhân

Trong dự án Lab 14 - Evaluation Factory, tôi đảm nhiệm vai trò **AI Engineer** với các mục tiêu trọng tâm:

- Chuyển đổi Agent từ dạng giả lập sang một hệ thống RAG thực thụ có khả năng truy xuất và trả lời dựa trên tài liệu nội bộ.
- Xây dựng hệ thống đánh giá đa giám khảo (Multi-Model Judge) để tăng độ khách quan và tin cậy cho kết quả benchmark.
- Đảm bảo hệ thống có cơ chế fallback linh hoạt, không bị gián đoạn khi gặp lỗi API hoặc thiếu tài nguyên.

## 2. Phạm vi phụ trách chính

- **Phát triển Agent (`agent/main_agent.py`)**: Hiện thực hóa pipeline RAG gồm Retrieval (lexical scoring) và Generation (LLM + Heuristic Fallback).
- **Thiết kế Judge (`engine/llm_judge.py`)**: Xây dựng `MultiModelJudge` hỗ trợ chấm điểm song song, tính toán độ đồng thuận (agreement rate) và xử lý xung đột (conflict resolution).
- **Chuẩn hóa dữ liệu đầu ra**: Đảm bảo Agent trả về đầy đủ `retrieved_ids` và `metadata` (token usage, model, sources) phục vụ cho việc tính toán metrics.

## 3. Đóng góp kỹ thuật nổi bật

### 3.1 Pipeline RAG thực tế với Lexical Retrieval
- Thay thế các câu trả lời hardcode bằng logic truy xuất dựa trên độ tương quan token (_lexical scoring_).
- Triển khai tiền xử lý tài liệu: tách câu, chuẩn hóa ID, và lọc stopword để tối ưu hóa việc tìm kiếm từ khóa.
- Cơ chế chấm điểm tài liệu kết hợp giữa đếm token trùng lặp và tìm kiếm chuỗi khớp chính xác để tăng độ ưu tiên cho các đoạn văn bản chứa câu hỏi.

### 3.2 Hệ thống Multi-Model Judge và Tie-breaking
- Triển khai luồng đánh giá song song dùng đồng thời `gpt-4o` và `gpt-4o-mini` để đối chiếu kết quả.
- Thiết lập cơ chế **Tiebreaker**: Nếu điểm số giữa 2 model lệch nhau quá ngưỡng `conflict_threshold`, hệ thống sẽ tự động gọi model thứ 3 để phân xử, giúp giảm thiểu sai số của một model đơn lẻ.
- Tính toán `agreement_rate` để đo lường độ ổn định của tiêu chí đánh giá qua từng phiên chạy.

### 3.3 Cơ chế Fallback thông minh
- **Generation Fallback**: Khi không có API key hoặc LLM lỗi, Agent chuyển sang chế độ `extractive` - tự động trích xuất câu có độ tương quan cao nhất từ tài liệu để trả lời, thay vì trả về lỗi.
- **Judge Fallback**: Xây dựng `_heuristic_judge` dựa trên các thuật toán truyền thống (Overlap ratio) để đảm bảo pipeline benchmark vẫn chạy được và có số liệu tham khảo ngay cả khi offline.

### 3.4 Kiểm soát định kiến vị trí (Position Bias)
- Hiện thực hóa phương thức `check_position_bias` để kiểm tra xem Judge có bị thiên kiến khi ưu tiên câu trả lời đứng trước hay không thông qua việc tráo đổi vị trí A/B và so sánh kết quả.

## 4. Chiều sâu kỹ thuật đã áp dụng

- **Async Concurrency**: Sử dụng `asyncio.gather` để gọi đồng thời nhiều Judge/Agent, giảm đáng kể thời gian chạy benchmark trên bộ dữ liệu lớn.
- **Rubric-based Scoring**: Chấm điểm dựa trên 3 tiêu chí rõ ràng: `accuracy` (độ chính xác facts), `faithfulness` (độ trung thực với context), và `relevancy` (độ liên quan câu hỏi).
- **Metadata Instrumentation**: Ghi lại chi tiết `llm_mode` (api, fallback, no_context) và `tokens_used` để phục vụ phân tích chi phí và hiệu năng.

## 5. Vấn đề phát sinh và cách xử lý

- **Vấn đề 1**: Model Judge đôi khi trả về format JSON không chuẩn hoặc chứa markdown.
  - **Cách xử lý**: Ép kiểu đầu ra bằng `response_format={"type": "json_object"}` và viết hàm `_normalize_judge_payload` để rà soát, gán giá trị mặc định cho các trường bị thiếu hoặc sai kiểu dữ liệu.
- **Vấn đề 2**: Câu hỏi nằm ngoài phạm vi tài liệu (OOS) thường bị LLM "bịa" câu trả lời (hallucination).
  - **Cách xử lý**: Tinh chỉnh System Prompt yêu cầu Agent chỉ được dùng context và trả về "Out of scope" nếu không tìm thấy. Đồng thời ở phía Judge, bổ sung logic nhận diện OOS trong heuristic để chấm điểm chính xác hơn.

## 6. Kết quả đạt được (theo vai trò AI Engineer)

- **Agent hoàn thiện**: Đạt tỷ lệ **Hit rate 0.9667** trên bộ dữ liệu thật, chứng tỏ logic retrieval hoạt động hiệu quả.
- **Judge tin cậy**: Tỷ lệ đồng thuận giữa các giám khảo đạt **86.41%**, mức xung đột thấp, giúp kết quả đánh giá có trọng số thuyết phục.
- **Tự động hóa**: Hệ thống có khả năng tự xử lý lỗi API và duy trì dòng chảy dữ liệu liên tục qua các mode fallback.

## 7. Bài học cá nhân

- Việc xây dựng cơ chế Fallback không chỉ là để "chống cháy" mà còn giúp quá trình debug nhanh hơn khi không cần phụ thuộc hoàn toàn vào internet/API key.
- Đánh giá LLM bằng LLM (LLM-as-a-Judge) cần được giám soát bằng các metrics như `agreement_rate` để tránh tình trạng "judge ảo".
- Prompt engineering đóng vai trò quyết định trong việc giữ cho Agent tuân thủ grounding, đặc biệt là với các câu hỏi bẫy.

## 8. Kế hoạch cải thiện vòng tiếp theo

- Thử nghiệm **Semantic Retrieval** (Embedding) để thay thế Lexical hiện tại, nhằm xử lý các câu hỏi mang tính diễn đạt (paraphrasing).
- Mở rộng Multi-judge sang các model mã nguồn mở (như Llama 3) để tối ưu chi phí.
- Tối ưu hóa prompt của Judge để cung cấp `reasoning` chi tiết và có tính gợi ý sửa lỗi cao hơn.
