# Báo cáo Cá nhân (Reflection Report) - Lab Day 14

**Họ và tên:** Dương Trung Hiếu
**Mã HV:** 2A202600051
**Vai trò:** Data Engineer
**Nhiệm vụ đảm nhiệm:** Thiết kế và hoàn thiện bộ dữ liệu Hard Cases (`data/HARD_CASES_GUIDE.md`).
**Đồng sự phối hợp:** Lam (phụ trách logic code `data/synthetic_gen.py`).

---

## 1. Chi tiết công việc đã thực hiện
Dựa theo Definition of Done (DoD) yêu cầu tạo tối thiểu 10 hard cases có tag `metadata.type` rõ ràng, tôi đã chịu trách nhiệm chính trong việc xây dựng tài liệu `data/HARD_CASES_GUIDE.md`. 

Cụ thể, tôi đã thiết kế thành công 12 trường hợp khó (Hard Cases) đóng vai trò làm bộ dữ liệu "Red Teaming" để thử thách Agent. Các cases này được chia làm 3 nhóm chính:
- **Adversarial Prompts:** Các câu hỏi tấn công prompt injection và yêu cầu bỏ qua context (bỏ qua quy định hệ thống).
- **Edge Cases:** Các trường hợp hỏi ngoài lề (Out of context), câu hỏi mập mờ thiếu thông tin (Ambiguous) và các thông tin mâu thuẫn giữa nhiều phiên bản tài liệu (Conflicting Info).
- **Multi-Turn / Multi-Hop Complexity:** Thử thách Agent tổng hợp thông tin từ nhiều nguồn tài liệu khác nhau và sửa lỗi đính chính ở lượt hội thoại thứ 2.

Sau khi thiết kế xong, tôi đã phối hợp chặt chẽ với Lam (người code script `synthetic_gen.py`) để đảm bảo 12 cases này được parse thành công vào file `data/golden_set.jsonl` đúng chuẩn format hệ thống yêu cầu.

## 2. Giải quyết vấn đề (Problem Solving)
Trong quá trình làm việc, một thách thức lớn về kỹ thuật (Git) đã xảy ra. Khi tôi đang làm việc trên nhánh cá nhân (nhánh `Hieu`) và cần cập nhật code mới nhất từ nhánh `main` của team, tôi đã sử dụng lệnh `git rebase main` thay vì `merge` để giữ lịch sử log gọn gàng.

Quá trình rebase đã gây ra **Merge Conflict** trực tiếp tại file `data/HARD_CASES_GUIDE.md` mà tôi đang soạn thảo. 

**Cách tôi xử lý:**
1. Bình tĩnh đọc log báo lỗi của Git để xác định đúng file bị conflict.
2. Xử lý conflict thủ công, giữ lại các thay đổi mới nhất của cả team và nội dung hard cases của mình.
3. Chạy `git rebase --continue` và thao tác trong trình soạn thảo Terminal (Vim) để lưu lại commit message.
4. Cuối cùng, do ID của các commit bị thay đổi sau khi rebase, tôi đã nhận ra tình trạng "lệch pha" (diverged) giữa máy local và server. Tôi đã chủ động sử dụng `git push --force-with-lease` thay vì `git pull` để đẩy code an toàn lên server mà không làm hỏng lịch sử của đồng nghiệp.

## 3. Bài học rút ra (Lessons Learned)
- **Về AI Engineering:** Hiểu được tầm quan trọng của việc đánh giá RAG hệ thống không chỉ qua các câu hỏi đơn giản (fact-check) mà còn phải thử thách nó bằng các câu gài bẫy (Adversarial). Một hệ thống AI giỏi phải biết cách "từ chối trả lời" khi không có dữ liệu.
- **Về Teamwork & Git:** Nắm vững bản chất của lệnh `git rebase`, sự khác biệt so với `merge` và cách xử lý conflict chuyên nghiệp trong một dự án có nhiều người cùng thay đổi file.