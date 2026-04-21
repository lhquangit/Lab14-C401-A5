# Reflection Cá nhân - 

### Le Hong Quan (Tech Lead / Integrator)
### MSV: 2A202600097

## 1. Vai trò và mục tiêu cá nhân

Trong bài lab này, tôi đảm nhiệm vai trò **Tech Lead/Integrator** với mục tiêu chính là:

- Đồng bộ các module thành một pipeline đánh giá chạy được end-to-end.
- Đảm bảo kết quả benchmark có thể kiểm chứng bằng số liệu (retrieval, judge, regression gate).
- Giữ chất lượng tích hợp ổn định khi nhóm chỉnh sửa song song nhiều file.

## 2. Phạm vi phụ trách chính

- Tích hợp luồng chạy tổng trong `main.py` (V1 vs V2, ghi `summary.json`, `benchmark_results.json`, release gate).
- Chuẩn hóa logic chấm điểm đa giám khảo trong `engine/llm_judge.py` (agreement/conflict/error/scoring_mode).
- Rà soát và hiệu chỉnh engine đánh giá trong `engine/retrieval_eval.py` và `engine/runner.py`.
- Cập nhật kiểm tra nộp bài trong `check_lab.py` để phản ánh đúng schema output mới.
- Hoàn thiện báo cáo phân tích lỗi hệ thống và bổ sung trực quan so sánh model.

## 3. Đóng góp kỹ thuật nổi bật

### 3.1 Tích hợp Benchmark end-to-end

- Tách rõ 2 luồng: baseline (V1) và candidate (V2), sau đó so sánh bằng regression metrics.
- Dùng `BenchmarkRunner.aggregate_results()` làm nguồn metrics thống nhất để tránh lệch số giữa module.
- Bổ sung các chỉ số vận hành vào summary: `avg_score`, `hit_rate`, `mrr`, `agreement_rate`, `conflict_rate`, `error_rate`, `counts`, `cost`.

### 3.2 Thiết kế Release Gate theo nguyên tắc sản phẩm

- Áp dụng gate đa điều kiện thay vì chỉ nhìn 1 điểm trung bình.
- Kết hợp cả ngưỡng tuyệt đối và ngưỡng delta so với baseline.
- Kết quả thực tế: hệ thống **BLOCK RELEASE** vì `delta_score=-0.0615` thấp hơn ngưỡng `-0.05`, dù các chỉ số khác đạt.

### 3.3 Củng cố độ tin cậy của Judge

- Dùng multi-judge (`gpt-4o` + `gpt-4o-mini`) và gọi tiebreaker khi xung đột.
- Theo dõi `agreement_rate`, `conflict_rate`, `judge_errors` thay vì chỉ dùng `final_score`.
- Thêm fallback heuristic khi không có API key để pipeline không gãy.

### 3.4 Đồng bộ báo cáo và kiểm thử nộp bài

- Chuẩn hóa `check_lab.py` theo output schema hiện tại để phát hiện lỗi nộp bài sớm.
- Viết `analysis/failure_analysis.md` dựa trên dữ liệu thật trong reports, không dùng template giả.
- Tạo script `analysis/plot_model_comparison.py` để trực quan hóa V1/V2 cho phân tích regression.

## 4. Chiều sâu kỹ thuật đã áp dụng

- **MRR**: đo thứ hạng tài liệu đúng đầu tiên để đánh giá chất lượng retrieval sâu hơn hit-rate.
- **Agreement/Conflict**: dùng làm tín hiệu độ ổn định của chấm điểm, giảm rủi ro lệ thuộc một judge.
- **Regression Gate**: bảo vệ chất lượng release bằng tiêu chí định lượng, tránh “cảm giác tốt hơn”.
- **Trade-off cost vs quality**: theo dõi tokens/case và đề xuất cấu hình model tiết kiệm hơn khi cần.

## 5. Vấn đề phát sinh và cách tôi xử lý

- Vấn đề 1: Dữ liệu benchmark và summary có lúc lệch nhau do schema/nguồn tổng hợp khác nhau.
Cách xử lý: thống nhất về một nguồn aggregate, cập nhật lại luồng ghi report.
- Vấn đề 2: Nhánh merge bị conflict ở file report JSON.
Cách xử lý: resolve conflict theo phiên bản benchmark mới nhất, đảm bảo file hợp lệ và đọc được.
- Vấn đề 3: Điểm trung bình thấp dù retrieval cao.
Cách xử lý: phân tích fail cases cho thấy lỗi chính ở answering/routing (nhiều câu `"Out of scope."` sai ngữ cảnh), từ đó ưu tiên sửa logic fallback.

## 6. Kết quả đạt được (theo run hiện tại)

- Tổng cases: **61**
- Pass/Fail: **33/28**
- Avg score: **3.3687**
- Hit rate: **0.9667**
- MRR: **0.9083**
- Agreement rate: **0.8641**
- Error rate: **0.0**
- Quyết định gate: **BLOCK RELEASE** (fail `delta_score`)

## 7. Bài học cá nhân ở vai trò Tech Lead/Integrator

- “Tích hợp đúng” quan trọng không kém “thuật toán tốt”; sai schema hoặc sai nguồn tổng hợp có thể làm sai toàn bộ kết luận.
- Regression gate giúp đội ra quyết định khách quan, tránh đẩy bản chưa ổn định chỉ vì một vài chỉ số đẹp.
- Khi retrieval đã tốt nhưng điểm vẫn thấp, cần tập trung vào lớp answer policy/fallback và grounding thay vì chỉ tối ưu retriever.
- Với hệ thống nhiều thành phần, checklist kiểm tra đầu ra (format + logic) giúp giảm rủi ro ở phút cuối.

## 8. Kế hoạch cải thiện vòng tiếp theo

- Giảm mạnh false `"Out of scope"` bằng rule ràng buộc theo retrieval signal.
- Bổ sung kiểm tra factual grounding trước khi trả lời để giảm hallucination mềm.
- Theo dõi thêm metrics bias/consistency của judge theo nhóm câu hỏi.
- Tối ưu chi phí đánh giá bằng cấu hình model judge theo tier (default rẻ, escalate khi conflict).

