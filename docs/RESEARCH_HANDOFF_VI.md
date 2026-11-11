# BR-CP | Bàn giao kiểm định bổ sung SAOCP / PID–AR(1)
**Ngày:** 11/11/2026 | **Tình trạng:** PRE-SUBMISSION, cần tác giả duyệt bản cuối.

## Mới so với gói Verified Science Gates trước đó
1. **Manuscript** đã bổ sung mục *Exploratory strongly adaptive and scorecasting comparators* và bảng SAOCP / PID–AR(1) dựa trên 108 cấu hình Chronos-Bolt Small đã lưu. PDF 25 trang, LaTeX nguồn đính kèm.
2. **Point-by-Point Response** đã bổ sung phản hồi về strongly adaptive baseline và scorecaster baseline, Word 15 trang và PDF để đọc nhanh.
3. **EVIDENCE/SAOCP** gồm mã port từng bước cập nhật, 108 checkpoint, 108 kết quả, 564 cửa sổ sự kiện, kết quả paired và 4.000 bootstrap theo 6 dataset.
4. **EVIDENCE/PID_AR1** gồm mã AR(1) scorecaster causal, grid development, bảng 108 trường hợp và phiên bản có KI>0, kèm kiểm thử lookahead.
5. **verify_new_baselines.py** kiểm toán các con số trong bảng mới. Chạy từ thư mục ZIP đã giải nén bằng `python verify_new_baselines.py`.

## Lưu ý về tính khoa học
- Chỉ dùng dự báo Chronos-Bolt Small **đã cache**. Không thực hiện inference GPU mới trong nhánh này.
- Đây là thí nghiệm bổ sung **post-hoc/thăm dò**, không phải confirmatory/pre-registered.
- SAOCP là **scores-only source-derived port**, không phải chạy trực tiếp thư viện của Salesforce trên Merlion.
- PID–AR(1) là cấu hình có scorecaster riêng, **không** là cấu hình ThetaModel chuẩn của tác giả gốc; grid phát triển không hạn chế chọn KI=0, sau đó mới làm thí nghiệm KI>0 thăm dò.
- So sánh 36 kênh mỗi scenario chỉ có 6 dataset gốc; bootstrap theo cụm có rất ít cụm nên phải diễn giải dè dặt.
- SAOCP có coverage trung bình và Winkler tốt hơn BR-CP trên các pulse variance trong thiết kế này, nhưng BR-CP có tỷ lệ lỗi thấp hơn ở 100 bước đầu sau variance break. Trên level break, SAOCP tốt hơn về lỗi sớm. Không có lợi thế tuyệt đối.
- AgACI–BOA gốc và PID–ThetaModel nguyên bản **chưa được chạy nguyên dạng**; tính tách biệt tuyệt đối của dữ liệu train của Chronos và nguồn electricity pretrain ridge lịch sử **chưa chứng minh được**. Không nói đã khắc phục hết mọi phê bình.

## Cấu trúc
- `MANUSCRIPT/` PDF, main.tex, Fig1–Fig7
- `REVIEWER_RESPONSE/` Word + PDF
- `EVIDENCE/SAOCP/` mã và kết quả bổ sung
- `EVIDENCE/PID_AR1/` mã và kết quả bổ sung
- `INPUT_ARCHIVES/` các ZIP nguồn để đối chiếu có SHA-256
- `SCIENTIFIC_GATE_STATUS.md` kết luận và checklist còn mở
- `SHA256SUMS.txt` kiểm tra toàn vẹn.

**Không dùng bộ này thay lời xác nhận khoa học của tác giả.** Nộp bài do tác giả thực hiện. Hạn Major Revision theo email: 14/11/2026.
