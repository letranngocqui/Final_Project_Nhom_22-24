# Dự đoán đơn giá rao bán bất động sản (CafeLand) — Final Project
**Nhóm 22 & 24**

Dự án thu thập tin rao bán nhà đất tại TP.HCM (và các khu vực sáp nhập: Bình Dương, Bà Rịa - Vũng Tàu) từ `nhadat.cafeland.vn`, làm sạch dữ liệu, phân tích, rồi xây dựng mô hình dự đoán **đơn giá rao bán theo m²** (`price_per_m2 = gia_goc / dien_tich_goc`, đơn vị VNĐ/m²).

> **Phạm vi:** kết quả là dự báo trên *giá rao*, không phải giá giao dịch thực và không thay thế thẩm định chuyên gia.

## 1. Kết quả chính (Bài tập 4)
Dữ liệu modeling: 3.480 tin (train 2.784 / test 696, split 80/20, `random_state=42`). Chọn mô hình bằng 5-fold CV trên train theo quy tắc one-standard-error, sau đó mới đánh giá trên test.

| Mô hình | CV MAE (tr/m²) | Test MAE (tr/m²) | Test RMSE (tr/m²) | Test R² |
|---|---:|---:|---:|---:|
| LinearRegression | 56,6 | 56,1 | 81,3 | 0,241 |
| Ridge | 56,6 | 56,1 | 81,4 | 0,240 |
| Ridge (log1p target) | 54,0 | 55,4 | 85,0 | 0,171 |
| **RandomForest (được chọn)** | **51,7** | **53,8** | **81,0** | **0,247** |

- Với 4 feature hiện tại (`dien_tich_goc`, `so_phong_ngu`, `khu_vuc_nhom`, `loai_bds`), mô hình chỉ giải thích khoảng 25% phương sai → **chỉ nên dùng làm khoảng tham chiếu**.
- Thử nghiệm thêm `phuong_xa` (cấp phường/xã) đưa R² test lên khoảng 0,57 (MAE 38,2) nhưng **chưa nằm trong mô hình cuối**; xem `reports/model_findings.md`.
- Chi tiết diễn giải: `reports/model_findings.md`, `reports/business_interpretation.md`, `reports/modeling_report_section.md`.

## 2. Demo
- Ứng dụng Streamlit: **<DÁN LINK DEMO ĐÃ DEPLOY VÀO ĐÂY>**
- Chạy cục bộ: `streamlit run app.py`
- Ảnh demo dự phòng: `reports/figures/01_demo_streamlit_app.png`, `02_demo_streamlit_app.png`

## 3. Cấu trúc thư mục
```text
├── app.py                      # Demo Streamlit định giá nhanh
├── run_pipeline.py             # Bài 2: thu thập/làm sạch/xuất dữ liệu
├── run_model.py                # Bài 4: chạy lại toàn bộ luồng modeling
├── requirements.txt
├── AI_USAGE.md                 # Khai báo sử dụng AI
├── data/
│   ├── raw/                    # cafeland_raw.csv (dữ liệu thô)
│   └── processed/              # dữ liệu sạch, đặc trưng, tập train/test
├── docs/                       # project_charter.docx, data_dictionary.md, model_preprocessing_notes.md
├── models/best_model.joblib    # Pipeline tốt nhất (preprocess + RandomForest)
├── notebooks/                  # 01-06: dữ liệu & EDA | 07-09: modeling
├── reports/                    # bảng so sánh, importance, báo cáo; figures/ chứa biểu đồ
└── src/
    ├── crawler.py  parser.py  io_manager.py
    └── modeling/preprocess.py  # tiền xử lý chống leakage
```

## 4. Cài đặt
Yêu cầu **Python ≥ 3.10** (đã kiểm thử trên Python 3.12).
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
`requirements.txt` ghim `scikit-learn==1.7.2` vì `best_model.joblib` được lưu bằng phiên bản này; dùng phiên bản khác sẽ làm app báo lỗi khi dự đoán.

## 5. Cách chạy
**Bài 2 — làm sạch & xuất dữ liệu** (dùng dữ liệu thô có sẵn, bỏ qua bước cào web):
```bash
python run_pipeline.py
# Kiểm tra crawler thật (nhanh, 3 trang):
python run_pipeline.py --crawl --start-page 1 --end-page 3
```
**Bài 4 — modeling** (từ `cafeland_missing_outliers_cleaned.csv` đến `best_model.joblib`, ~1 phút):
```bash
python run_model.py --plots
# Ghi ra thư mục khác để không ghi đè file nộp:
python run_model.py --plots --output-root out_test
```
**Demo:**
```bash
streamlit run app.py
```
**Notebook:** chạy theo thứ tự `01` → `09` trong `notebooks/`. `model_visualization.ipynb` là phần biểu đồ của Quí.

## 6. Nguyên tắc chống leakage
- Target chỉ tính từ giá/diện tích **gốc quan sát được**, không dùng giá đã impute.
- Khử trùng theo `link_nguon` trước khi chia train/test.
- Chia train/test **trước** khi impute, scale, one-hot; mọi bước học từ dữ liệu nằm trong `Pipeline` nên được fit lại trong từng fold CV.
- Cấm đưa `gia`, `gia_goc`, `price_per_m2` vào X (xem `reports/feature_contract.csv`).

## 7. Hạn chế
- Chỉ 4 feature, R² test ≈ 0,25; sai số lớn nhất ở đất nền (ít thông tin, 96% tin không ghi số phòng ngủ).
- Dữ liệu là giá rao, có thể lệch so với giá giao dịch.
- Nhãn vùng "mới sáp nhập" là nhãn nội bộ phục vụ phân tích.

## 8. Thành viên
| Thành viên | Vai trò |
|---|---|
| Chu Minh Tuấn | Chuẩn bị dữ liệu mô hình, chống leakage |
| Bùi Đức Mạnh | Huấn luyện & so sánh mô hình |
| Lê Trần Ngọc Quí | Demo Streamlit, trực quan hoá |
| Nguyễn Hùng Phúc | Diễn giải mô hình, độ bền |
| Lê Nhật Việt | Diễn giải nghiệp vụ, báo cáo khoa học |
| Huỳnh Bá Long | Tích hợp, README, `run_model.py`, hoàn thiện repo |
