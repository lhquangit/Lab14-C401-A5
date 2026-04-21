# Báo cáo Phân tích Thất bại (Failure Analysis Report)

Nguồn dữ liệu: `reports/summary.json` và `reports/benchmark_results.json` (run lúc `2026-04-21 16:33:04`).

## 1. Tổng quan Benchmark
- Tổng số cases: **61**
- Pass/Fail: **33 pass / 28 fail** (Pass rate: **54.1%**, Fail rate: **45.9%**)
- Điểm trung bình (LLM Judge): **3.3687 / 5.0**
- Retrieval:
  - Hit Rate: **0.9667**
  - MRR: **0.9083**
- Judge quality:
  - Agreement Rate: **0.8641**
  - Conflict Rate: **0.3443**
  - Error Rate: **0.0000**
- Regression gate: **BLOCK RELEASE** vì `delta_score = -0.0615 < -0.05`.

Nhận định nhanh: retrieval khá tốt, nhưng chất lượng câu trả lời cuối vẫn thấp ở nhiều case, dẫn đến fail nhiều và làm V2 giảm điểm so với V1.

## 2. Phân nhóm lỗi (Failure Clustering)

### 2.1 Cụm lỗi chính (exclusive trên 28 fail cases)
| Nhóm lỗi | Số lượng | Tỉ lệ trong fail | Dấu hiệu |
|---|---:|---:|---|
| False negative "Out of scope" | 25 | 89.3% | Agent trả về `"Out of scope."` dù câu hỏi thuộc phạm vi tài liệu |
| Trả lời có nội dung nhưng sai fact | 3 | 10.7% | Sai actor/sai bước/sai điều kiện so với golden answer |

### 2.2 Yếu tố đóng góp
| Yếu tố | Số lượng | Ghi chú |
|---|---:|---|
| Fail nhưng retrieval **vẫn hit** ground-truth | 26/28 | Chủ yếu lỗi answering/routing, không phải retrieval |
| Fail do retrieval miss thật sự | 2/28 | Tập trung ở câu hỏi SLA (`doc_sla_p1_2026`) |
| Fail có câu trả lời rất ngắn (<=4 từ) | 17/28 | Hầu hết là `"Out of scope."` |

### 2.3 Phân bố fail theo tài liệu kỳ vọng
| Tài liệu kỳ vọng | Số fail |
|---|---:|
| `doc_it_helpdesk_faq` | 8 |
| `doc_access_control_sop` | 6 |
| `doc_hr_leave_policy` | 5 |
| `doc_policy_refund_v4` | 5 |
| `doc_sla_p1_2026` | 4 |

## 3. Phân tích 5 Whys (3 case tiêu biểu)

### Case A - idx=30 (score=1.0)
- Câu hỏi: "Tôi cần cài phần mềm mới, có cần sự phê duyệt không?"
- Kỳ vọng: cần gửi ticket IT-SOFTWARE và Line Manager phê duyệt.
- Thực tế: Agent trả `"Out of scope."`.
- Retrieval: đã lấy đúng `doc_it_helpdesk_faq` ở top results.

1. Symptom: Trả lời từ chối phạm vi dù có tài liệu liên quan.
2. Why 1: Lớp quyết định trả lời (answer routing) chọn nhánh từ chối.
3. Why 2: Logic fallback/ràng buộc "đủ tự tin mới trả lời" đang quá chặt.
4. Why 3: Không có bước ép extract fact khi đã có source liên quan trong top-k.
5. Why 4: Chưa có guardrail "nếu hit_rate > 0 thì phải trả lời theo context".
6. Root cause: Chính sách fallback "Out of scope" lấn át đường trả lời theo context.

### Case B - idx=38 (score=2.11)
- Câu hỏi: "Khách hàng cần làm gì để yêu cầu hoàn tiền?"
- Kỳ vọng: gửi yêu cầu qua hệ thống ticket nội bộ category `Refund Request`.
- Thực tế: Agent thêm điều kiện "7 ngày, sản phẩm chưa mở seal..." không có trong ground truth.

1. Symptom: Câu trả lời dài nhưng sai policy trọng yếu.
2. Why 1: Model sinh thêm điều kiện ngoài ngữ cảnh tài liệu.
3. Why 2: Prompt chưa ép trích dẫn/neo chặt vào câu chữ trong source.
4. Why 3: Chưa có bước kiểm tra consistency giữa answer và evidence.
5. Why 4: Hệ thống chưa hậu kiểm hallucination theo từng claim.
6. Root cause: Thiếu cơ chế answer-constrained generation (grounded answering).

### Case C - idx=55 (score=2.5)
- Câu hỏi: "Sẽ mất bao lâu để xử lý ticket P4?"
- Kỳ vọng: theo sprint cycle (2-4 tuần).
- Thực tế: Agent trả `"Out of scope."`.
- Retrieval: miss ground-truth doc (`doc_sla_p1_2026`) hoàn toàn (`hit_rate=0`).

1. Symptom: Không trả lời được vì không lấy đúng tài liệu.
2. Why 1: Retriever ưu tiên nhầm sang doc FAQ/access/refund.
3. Why 2: Query lexical match cho "P4" chưa đủ mạnh trong ranking.
4. Why 3: Không có reranker theo intent SLA/priority.
5. Why 4: Bộ chỉ mục/chunk chưa tối ưu cho pattern `P1/P2/P3/P4`.
6. Root cause: Retrieval cho nhóm câu hỏi SLA chưa đủ chính xác.

## 4. Kế hoạch cải tiến (Action Plan)

### Ưu tiên P0 (chặn lỗi lớn ngay)
- [ ] Sửa logic fallback: chỉ cho phép `"Out of scope"` khi `has_ground_truth=False` hoặc retrieval score dưới ngưỡng thấp rõ ràng.
- [ ] Thêm guardrail: nếu top-k có doc kỳ vọng/hit cao thì bắt buộc trả lời theo context, không được từ chối trống.
- [ ] Thêm test regression cho 25 câu fail kiểu `"Out of scope"` để ngăn tái diễn.

### Ưu tiên P1 (nâng chất lượng factual)
- [ ] Cập nhật prompt: yêu cầu "chỉ trả lời bằng thông tin xuất hiện trong source", cấm thêm điều kiện ngoài tài liệu.
- [ ] Bổ sung bước evidence-check đơn giản trước khi trả lời (match các fact chính với snippet truy xuất được).
- [ ] Chuẩn hóa format answer ngắn gọn theo policy (actor, action, timeline, channel).

### Ưu tiên P2 (cải thiện retrieval cho SLA)
- [ ] Tăng trọng số lexical cho token ưu tiên `P1/P2/P3/P4`, `SLA`, `incident`.
- [ ] Thêm reranking theo intent (SLA/priority).
- [ ] Bổ sung hard cases SLA vào bộ kiểm thử để theo dõi riêng hit-rate nhóm này.

## 5. Mục tiêu vòng benchmark kế tiếp
- Mục tiêu 1: giảm fail từ **28 xuống <= 18**.
- Mục tiêu 2: giảm số câu `"Out of scope"` sai ngữ cảnh từ **25 xuống <= 5**.
- Mục tiêu 3: đưa `avg_score` lên **>= 3.45** và đạt gate `delta_score >= -0.05`.
