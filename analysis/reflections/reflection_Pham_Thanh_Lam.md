# Báo cáo Cá nhân (Individual Reflection) - Lab Day 14

**Họ và tên:** Phạm Thanh Lam  
**Vai trò:** Data Engineer (Nhóm Data)  
**Nhiệm vụ chính:** Giai đoạn 1 (GĐ1) - Thiết kế và xây dựng Golden Dataset & Hard Cases Guide.

---

## 1. Đóng góp kỹ thuật (Engineering Contribution)

Trong dự án này, tôi chịu trách nhiệm chính cho việc xây dựng "nền móng" của hệ thống đánh giá - đó là **Golden Dataset**. Các đóng góp cụ thể bao gồm:

### Xây dựng Script SDG (`data/synthetic_gen.py`)

- **Triển khai Async Pipeline:** Tôi đã sử dụng `AsyncOpenAI` để tối ưu tốc độ sinh dữ liệu. Thay vì sinh tuần tự, script chạy song song cho tất cả các tài liệu trong thư mục `docs/`, giúp tạo ra 62 test cases chỉ trong chưa đầy 30 giây.
- **Phân loại dữ liệu thông minh:** Tôi đã thiết kế prompt để ép Model sinh ra 5 loại câu hỏi khác nhau với độ khó tăng dần:
  - `fact-check`: Truy vấn thông tin trực tiếp (Chiếm ~60%).
  - `inference`: Đòi hỏi suy luận logic (Chiếm ~20%).
  - `adversarial`, `ambiguous`, `out-of-context`: Các trường hợp "phá hoại" để kiểm tra độ bền của Agent (Chiếm ~20%).
- **Cấu trúc hóa dữ liệu:** Đảm bảo mỗi case đều có `expected_retrieval_ids` chính xác để hỗ trợ tính toán Hit Rate và MRR ở GĐ2.

### Thiết kế Hard Cases Guide (`data/HARD_CASES_GUIDE.md`)

- Tôi đã soạn thảo bộ tài liệu hướng dẫn các kỹ thuật "Red Teaming" như **Prompt Injection**, **Goal Hijacking** và các trường hợp **Out-of-context** để nhóm có thể tiếp tục mở rộng bộ dữ liệu benchmark trong tương lai.

---

## 2. Độ sâu kỹ thuật (Technical Depth)

Qua quá trình thực hiện RAG Evaluation, tôi đã đúc kết được các kiến thức trọng tâm sau:

### Tầm quan trọng của Retrieval Metrics

- **MRR (Mean Reciprocal Rank):** Khác với Hit Rate (chỉ quan tâm có tìm thấy hay không), MRR đánh giá vị trí của tài liệu đúng trong danh sách trả về. Nếu tài liệu đúng nằm ở vị trí đầu tiên, điểm là 1, vị trí thứ hai là 0.5. MRR cao giúp giảm nhiễu cho LLM, từ đó giảm thiểu Hallucination.
- **Position Bias trong LLM Judge:** Tôi nhận thấy các Judge AI đôi khi có xu hướng thiên vị cho các câu trả lời ngắn hoặc các câu trả lời xuất hiện đầu tiên. Việc thiết kế Ground Truth rõ ràng trong Golden Set là cách tốt nhất để đo lường và hiệu chỉnh sự thiên vị này.

### Cân bằng Chi phí và Chất lượng (Cost vs Quality)

- Khi sinh data, tôi chọn `gpt-4o-mini` để tối ưu chi phí (giá rẻ hơn ~20 lần so với gpt-4o) nhưng vẫn đảm bảo chất lượng nhờ việc thiết kế System Prompt cực kỳ chi tiết với định dạng JSON strict mode.

---

## 3. Giải quyết vấn đề (Problem Solving)

**Thách thức:** Ban đầu, các câu hỏi do AI sinh ra thường quá dễ và chỉ lặp lại văn bản trong tài liệu.  
**Giải pháp:** Tôi đã áp dụng kỹ thuật **Few-shot Prompting** và mô tả cụ thể các loại "lỗi" mà tôi muốn AI tạo ra (như câu hỏi mập mờ hoặc câu hỏi không có trong context). Kết quả là bộ dữ liệu đã có các case như `case_011` hoặc `case_024` mà Agent bắt buộc phải trả lời "Tôi không tìm thấy thông tin này" thay vì tự bịa ra câu trả lời.

---

## 4. Tự đánh giá

- **Hoàn thành:** 100% khối lượng công việc GĐ1.
- **Tác động:** Bộ dữ liệu 62 cases đạt chuẩn đã giúp nhóm Backend và AI có cơ sở để tính toán chính xác các chỉ số Hit Rate (đạt 1.0) và MRR ngay lập tức.
