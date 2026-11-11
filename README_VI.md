# BR-CP — Repository dành cho kiểm chứng khoa học

Đây là **gói GitHub chuẩn bị công bố**, không phải bằng chứng đã được chấp nhận đăng. Gồm mã chương trình và **kết quả thật đã lưu**, cả những kết quả không có lợi cho BR-CP.

- Chạy thử CPU: `python -m pip install -r requirements.txt`, `python code/chronos_quantile_fixed/test_quantile.py`, `python experiments/matched_detector/test_detector_equivalence.py`, `python scripts/verify_new_baselines.py`.
- Thí nghiệm Chronos-Bolt Small đã chạy bằng L4, được lưu dự báo để kiểm toán không cần GPU.
- Không dùng bảng trong `results/chronos_original_DO_NOT_CITE`: bảng ban đầu bị lỗi công thức quantile.
- Không kết luận vượt trội toàn bộ; lợi thế variance break không chuyển sang mọi level break hay mọi interval score.
- Các baseline SAOCP và PID–AR(1) là bản chuyển thể, **không** là phép chạy chính xác phần mềm tác giả gốc.
- Không đưa CSV benchmark thô, email phản biện hoặc mã bí mật lên repository công khai; tải tập dữ liệu từ nguồn chính thức và kiểm tra SHA-256.
- Repository được chuẩn bị công khai để phục vụ tái lập; việc upload chỉ được xác nhận sau khi commit trên GitHub được kiểm tra. Các mã chuyển thể giữ ghi công riêng; nhóm tác giả chưa chỉ định giấy phép chung cho mã gốc.
