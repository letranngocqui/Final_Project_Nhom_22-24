# Từ Điển Dữ Liệu (Data Dictionary)

**Dự án:** Dự đoán giá rao bán trên m² của nhà đất tại TP.HCM 
(nguồn: nhadat.cafeland.vn/nha-dat-ban-tai-tp-ho-chi-minh/)
**Đơn vị phân tích:** 1 dòng = 1 tin đăng bán nhà/đất
**Phiên bản dữ liệu:** `data/processed/cafeland_cleaned.csv` — 6.500 dòng × 9 biến
**Thời gian thu thập (snapshot):** 21/02/2026 – 18/09/2026
**Cập nhật lần cuối:** 21/09/2026

---

## 1. Tổng quan các phiên bản dữ liệu

| Giai đoạn | File | Số dòng | Số cột | Mô tả |
|---|---|---|---|---|
| Thô (Raw) | `data/raw/cafeland_raw.csv` | ~biến động theo lần cào | 8 | Dữ liệu cào trực tiếp từ HTML, chưa xử lý |
| Đã làm sạch (Processed) | `data/processed/cafeland_cleaned.csv` (+ `.json`, `.parquet`) | 6.500 | 9 | Đã chuẩn hóa kiểu dữ liệu, tách biến `so_phong_ngu` |

Cột `so_phong_ngu` **không có** trong file raw — được sinh ra ở bước làm sạch (`src/parser.py`) bằng cách bóc tách từ `tieu_de` + `mo_ta` bằng Regex.

---

## 2. Bảng mô tả biến (Data Dictionary chính)

| # | Tên biến | Kiểu dữ liệu | Đơn vị | Ý nghĩa | Nguồn gốc | % Thiếu (trên 6.500 dòng) |
|---|---|---|---|---|---|---|
| 1 | `tieu_de` | Chuỗi (string) | — | Tiêu đề tin đăng bán bất động sản | Cào trực tiếp từ thẻ `<a class="realTitle">` trên trang chi tiết tin | 0% |
| 2 | `gia` | Số thực (float64) | VNĐ (đồng) | Giá rao bán của bất động sản, đã quy đổi về đơn vị đồng | Cào từ thẻ `<span class="reales-price">` (dạng text: "2 tỷ 990 triệu", "120 triệu/m2"...), sau đó được `parser.py` chuẩn hóa qua hàm `parse_price()`. Nếu giá niêm yết theo m² (chứa "/m2"), giá trị được nhân với `dien_tich` để ra tổng giá | 7,6% (492 dòng) — chủ yếu là tin ghi "giá thương lượng" không có số cụ thể |
| 3 | `dien_tich` | Số thực (float64) | m² (mét vuông) | Diện tích bất động sản | Cào từ thẻ `<span class="reales-area">` (dạng text: "72m2"), chuẩn hóa qua hàm `parse_area()` | 27,4% (1.782 dòng) — tin không công khai diện tích |
| 4 | `vi_tri` | Chuỗi (string) | — | Vị trí/khu vực của bất động sản, định dạng `Phường/Xã, Thành phố Hồ Chí Minh` | Cào từ thẻ `<div class="info-location">`, chuẩn hóa viết tắt (P., Q., TP.) qua hàm `normalize_location()` | 0% |
| 5 | `ngay_dang` | Ngày giờ (datetime, ISO 8601) | — | Thời điểm tin được đăng/cập nhật | Cào từ thuộc tính `data-time` của thẻ `<div class="reals-update-time">`; được ép kiểu bằng `pandas.to_datetime()` | 0% |
| 6 | `link_nguon` | Chuỗi (URL) | — | Đường dẫn tới tin gốc trên nhadat.cafeland.vn, dùng để tra cứu/kiểm chứng lại tin | Cào từ thuộc tính `href` của thẻ `<a class="realTitle">` | 0% |
| 7 | `mo_ta` | Chuỗi (string, văn bản dài) | — | Đoạn mô tả/giới thiệu ngắn về bất động sản do người bán viết | Cào từ thẻ `<div class="reales-preview">` | 0% |
| 8 | `nguoi_ban` | Chuỗi (string) | — | Tên người đăng tin / môi giới | Cào từ thẻ `<a class="member-name">` | 0% |
| 9 | `so_phong_ngu` | Số thực (float64) | phòng | Số lượng phòng ngủ, được bóc tách từ văn bản tự do trong `tieu_de` và `mo_ta` | **Biến dẫn xuất (derived)** — không có trong dữ liệu thô. Sinh ra bởi hàm `extract_bedrooms()` trong `parser.py`, dùng Regex bắt các mẫu "2PN", "2 phòng ngủ", "2p"... | 57,6% (3.747 dòng) — nhiều tin không đề cập số phòng ngủ trong văn bản |

---

## 3. Ghi chú chất lượng dữ liệu (Data Quality Notes)

- **Giá trị ngoại lai (outliers):** Cột `gia` và `dien_tich` có khoảng giá trị rất rộng (ví dụ `dien_tich` tối đa quan sát được lên tới hàng triệu m², nhiều khả năng do lỗi nhập liệu hoặc lô đất dự án lớn) — cần áp dụng IQR/Z-score để lọc outlier trước khi mô hình hóa, như đã nêu trong Project Charter.
- **Nhãn địa lý gộp:** Sau sáp nhập địa giới, các khu vực trước đây thuộc Bình Dương, Đồng Nai... nay đều mang nhãn "Thành phố Hồ Chí Minh" trong `vi_tri`. Cần phân biệt "vùng lõi cũ" và "vùng mới sáp nhập" khi phân tích, tránh gộp chung sai lệch.
- **Giá trị thiếu không ngẫu nhiên:** Diện tích và giá bị thiếu có chủ đích (người bán giấu thông tin để câu khách liên hệ) — không nên xử lý như thiếu ngẫu nhiên (MCAR), cần cân nhắc khi impute hoặc loại bỏ.
- **Biến mục tiêu dự kiến cho mô hình hồi quy:** `price_per_m2 = gia / dien_tich` (triệu VNĐ/m²) — sẽ được tính toán ở bước sau, chưa có sẵn trong file cleaned hiện tại.

---

## 4. Quy trình sinh dữ liệu (Data Lineage)

```
[1] src/crawler.py   → cào HTML từ nhadat.cafeland.vn → data/raw/cafeland_raw.csv (8 biến)
[2] src/parser.py    → làm sạch, chuẩn hóa, bóc tách so_phong_ngu (Regex)
[3] src/io_manager.py→ xuất 3 định dạng → data/processed/cafeland_cleaned.{csv,json,parquet} (9 biến)
```
