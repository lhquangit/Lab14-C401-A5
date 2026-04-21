# Task Breakdown Theo Từng File


| Phase | File                           | Owner đề xuất                 | Task cụ thể                                                                                         | Estimate (phút) | Definition of Done (DoD)                                                                             |
| ----- | ------------------------------ | ----------------------------- | --------------------------------------------------------------------------------------------------- | --------------- | ---------------------------------------------------------------------------------------------------- |
| GĐ1   | `data/synthetic_gen.py`        | Data Engineer (Lam + Hiếu)    | Sinh `>=50` cases, có `question/expected_answer/expected_retrieval_ids/metadata`                    | 45              | Chạy script tạo `data/golden_set.jsonl` đủ 50+ dòng, mỗi dòng parse JSON OK                          |
| GĐ1   | `data/HARD_CASES_GUIDE.md`     | Data Engineer (Lam + Hiếu)    | Bổ sung danh sách hard cases áp dụng vào generator (adversarial, ambiguous, out-of-context)         | 20              | Tối thiểu 10 hard cases xuất hiện trong `golden_set.jsonl` với tag `metadata.type` rõ ràng           |
| GĐ2   | `engine/retrieval_eval.py`     | Backend Engineer (Hải)        | Hoàn thiện `evaluate_batch`, tính `hit_rate@k`, `mrr` theo case + trung bình                        | 35              | Metric trả về từ dữ liệu thật, không hardcode; kiểm tra tay 3 case ra đúng                           |
| GĐ2   | `engine/llm_judge.py`          | AI Engineer (Linh)            | Triển khai multi-judge (2 model), rubric, agreement, xử lý conflict                                 | 60              | Output có `individual_scores`, `final_score`, `agreement_rate`, `reasoning`; lệch lớn có nhánh xử lý |
| GĐ2   | `engine/runner.py`             | Backend Engineer (Hải)        | Nâng async runner: batch concurrency + isolate lỗi từng case                                        | 35              | 1 case lỗi không làm chết cả batch; có trạng thái `pass/fail/error`                                  |
| GĐ2   | `agent/main_agent.py`          | AI Engineer (Linh)            | Thay agent giả lập bằng pipeline thật, trả `retrieved_ids`, token usage                             | 40              | Response có `answer`, `retrieved_ids`, `metadata.tokens_used`, `sources` thực tế                     |
| GĐ3   | `main.py`                      | Tech Lead/Integrator (Quân)   | Tích hợp evaluator/judge/runner thật, chạy V1 vs V2, release gate theo ngưỡng                       | 35              | Sinh `reports/summary.json` + `reports/benchmark_results.json`; quyết định gate dựa ngưỡng rõ        |
| GĐ3   | `analysis/failure_analysis.md` | Analyst (Quân)                | Điền số liệu benchmark thật, failure clustering, 5 Whys cho 3 case tệ nhất                          | 30              | Không còn placeholder `X/XX`; có 3 root causes + action plan cụ thể                                  |
| GĐ4   | `check_lab.py`                 | QA Engineer (Quân)            | Siết validator: min 50 cases, check schema và metric bắt buộc (`hit_rate`, `mrr`, `agreement_rate`) | 25              | Script fail đúng khi thiếu trường/sai số lượng, pass khi artifact hợp lệ                             |
| GĐ4   | `README.md`                    | Tech Writer/Integrator (Quân) | Cập nhật hướng dẫn chạy thực tế, mô tả ngưỡng gate và cấu trúc reports                              | 15              | Người mới clone repo chạy tuần tự không vướng bước mơ hồ                                             |




