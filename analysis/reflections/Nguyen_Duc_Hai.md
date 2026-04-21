# Báo cáo Cá nhân (Individual Reflection) - Lab Day 14
**Sinh viên:** Nguyễn Đức Hải  
**MSSV:** 2A202600149  
**Vai trò:** Backend Engineer

---

## 1. Đóng góp kỹ thuật

Trong dự án này, tôi chịu trách nhiệm chính trong việc phát triển hạ tầng đánh giá (Evaluation Infrastructure), cụ thể là hai module lõi:

*   **Module `engine/retrieval_eval.py`:**
    *   Hoàn thiện logic tính toán **Hit Rate @ k** và **MRR (Mean Reciprocal Rank)** từ mức độ từng test-case đến trung bình toàn tập dữ liệu.
    *   Xây dựng cơ chế `evaluate_batch` hỗ trợ hai chế độ: đánh giá dựa trên tập dữ liệu tĩnh và đánh giá trực tiếp từ output của Agent theo thời gian thực.
    *   Phân loại trạng thái dữ liệu (Evaluated, Skipped, No Ground-Truth) để đảm bảo độ chính xác của metrics khi dữ liệu đầu vào không đồng nhất.

*   **Module `engine/runner.py`:**
    *   Phát triển **Async Benchmark Runner** chuyên nghiệp sử dụng `asyncio`.
    *   Triển khai cơ chế **Batch Concurrency** bằng `asyncio.Semaphore` để kiểm soát số lượng request song song đến LLM, giúp hệ thống không bị lỗi Rate Limit của nhà cung cấp (OpenAI/Anthropic).
    *   Thiết kế kiến trúc **Error Isolation**: Sử dụng try/except cục bộ cho từng test-case để đảm bảo rằng nếu một case bị lỗi (crash agent hoặc timeout), toàn bộ quá trình benchmark của 50+ cases vẫn hoàn thành mà không bị gián đoạn.
    *   Tích hợp tính năng **Cost & Performance Tracking**: Tự động tính toán tổng số Token đã sử dụng và Throughput (cases/giây).

## 2. Độ sâu kỹ thuật

Qua bài lab này, tôi đã áp dụng và làm sâu sắc thêm các kiến thức chuyên môn sau:

*   **Metrics Đánh giá Retrieval:** Hiểu rõ tại sao chỉ đánh giá câu trả lời (Generation) là chưa đủ. Việc sử dụng **MRR** giúp đo lường khả năng xếp hạng của Retriever (tài liệu đúng nằm ở vị trí càng cao thì điểm càng tốt), trong khi **Hit Rate** đo lường khả năng tìm thấy tài liệu trong top-k.
*   **Xử lý Bất đồng bộ Nâng cao:** Không chỉ sử dụng `asyncio.gather` đơn thuần, tôi đã áp dụng Semaphore để tạo ra "Backpressure control", một kỹ thuật quan trọng trong AI Engineering để tối ưu hóa hiệu năng mà vẫn giữ an toàn cho hệ thống.
*   **Trade-off giữa Chi phí và Chất lượng:** Qua việc tích hợp thông tin `scoring_mode` (Heuristic vs API) trong runner, tôi nhận thấy việc sử dụng Judge bằng mô hình nhỏ (gpt-4o-mini) hoặc Heuristic có thể giúp giảm tới 80% chi phí nhưng cần được calibrate (hiệu chuẩn) kỹ lưỡng để không làm giảm độ tin cậy.

## 3. Giải quyết vấn đề

Một trong những thách thức lớn nhất tôi gặp phải là việc **Isolate lỗi trong môi trường Async**. Ban đầu, nếu một task trong `gather` bị crash, nó sẽ làm hỏng kết quả của cả batch. 

**Giải pháp của tôi:**
1.  Bọc mỗi case trong một wrapper function với try/except riêng.
2.  Trả về một "Error Record" thay vì raise Exception.
3.  Sử dụng `return_exceptions=True` trong `asyncio.gather` như một lớp bảo vệ thứ hai.
Kết quả là hệ thống chạy cực kỳ ổn định, cho phép báo cáo chính xác tỷ lệ lỗi (Error Rate) thay vì dừng chương trình đột ngột.

## 4. Tự đánh giá

*   **Mức độ hoàn thành:** 100% các yêu cầu Expert Task được giao.
*   **Điểm mạnh:** Code sạch, có docstring đầy đủ, xử lý lỗi chặt chẽ và tối ưu hóa tốt cho việc chạy benchmark số lượng lớn.
*   **Điểm cần cải thiện:** Có thể tích hợp thêm cơ chế tự động Retry (Exponential Backoff) cho các request LLM bị timeout thay vì chỉ đánh dấu lỗi.

**Cam kết:** Mọi mã nguồn trong module `engine/` do tôi phụ trách đều đã được kiểm thử độc lập và sẵn sàng để Tech Lead tích hợp vào pipeline chung của nhóm.
