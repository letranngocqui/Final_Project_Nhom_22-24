# Bài tập 4 — Ghi chú chuẩn bị dữ liệu & chống leakage

**Người phụ trách:** Chu Minh Tuấn — Model Data Lead  
**Phạm vi:** Phần D.1 — Chuẩn bị dữ liệu cho mô hình.

## 1. Quyết định dữ liệu đầu vào

Pipeline mô hình **không dùng trực tiếp `cleaned_for_model.csv` hoặc các cột one-hot đã tạo trong `cafeland_features.csv`**. Hai file này đã được tạo trước train/test split, vì vậy dùng chúng để huấn luyện sẽ làm quy trình khó chứng minh hoàn toàn không leakage.

Điểm xuất phát được khóa tại:

`data/processed/cafeland_missing_outliers_cleaned.csv`

File này còn `link_nguon`, `gia_goc`, `dien_tich_goc` và các cờ thiếu, nên có thể truy vết target và khử trùng đúng cách.

## 2. Data eligibility / audit

| Bước | Trước | Sau | Loại | Lý do |
|---|---:|---:|---:|---|
| Khử duplicate `link_nguon` | 5.435 | 5.388 | 47 | Không để cùng một tin xuất hiện ở train và test |
| Chỉ target quan sát thực | 5.388 | 3.564 | 1.824 | Không tạo target từ giá/diện tích từng được impute |
| Scope diện tích 10–2.000 m² | 3.564 | 3.558 | 6 | Theo Project Charter |
| Loại tin cho thuê rõ ràng | 3.558 | 3.480 | 78 | Project Charter đặt tin cho thuê ngoài scope |
| Target dương/hữu hạn | 3.480 | 3.480 | 0 | Kiểm tra cuối |

**Modeling base cuối:** 3.480 tin rao bán.

## 3. Target contract

Target:

`price_per_m2 = gia_goc / dien_tich_goc` (VNĐ/m²)

Chỉ những dòng có `gia_thieu == False`, `dien_tich_thieu == False`, `gia_goc > 0`, `dien_tich_goc > 0` mới được phép tạo target.

**Cấm tuyệt đối trong X:** `gia`, `gia_goc`, `price_per_m2`. Đặc biệt, `gia_goc` là tử số trực tiếp của target nên nếu đưa vào X sẽ tạo target leakage gần như trực tiếp.

## 4. Feature contract

Feature hiện được phép:

- `dien_tich_goc` — numeric.
- `so_phong_ngu` — numeric, còn thiếu; được median-impute trong Pipeline và tự thêm missing indicator.
- `khu_vuc_nhom` — categorical, kế thừa logic nhóm khu vực của Bài tập 3.
- `loai_bds` — categorical, suy ra từ tiêu đề theo logic Bài tập 3.

Không one-hot `vi_tri` trực tiếp vì cardinality cao. `tieu_de` hiện chỉ được dùng để tạo `loai_bds`; chưa đưa văn bản thô vào mô hình vì chưa có NLP pipeline được kiểm chứng.

Bảng chi tiết: `reports/feature_contract.csv`.

## 5. Split và preprocessing

- Hold-out: 80/20, `random_state=42`.
- Train: 2.784 dòng.
- Test: 696 dòng.
- Split được thực hiện **trước** median imputation, scaling và one-hot encoding.

`src/modeling/preprocess.py` cung cấp `build_preprocessor()`:

- Numeric: `SimpleImputer(strategy="median", add_indicator=True)` → `StandardScaler()`.
- Categorical: `SimpleImputer(strategy="most_frequent")` → `OneHotEncoder(handle_unknown="ignore")`.
- Reference groups: `TP.HCM (lõi cũ)` và `Nhà ở/Nhà phố`.

Modeling Lead phải dùng `build_model_pipeline(estimator)` để preprocessor được refit trong từng cross-validation fold. Không dùng ma trận đã transform sẵn từ notebook để tuning.

## 6. Đa cộng tuyến

VIF được tính trên **train** sau khi preprocessor fit trên train. VIF lớn nhất hiện tại ≈ **1,219**, thấp hơn ngưỡng theo dõi 5. Việc chọn hai nhóm phổ biến làm reference category giúp tránh VIF cao do vô tình chọn một category hiếm làm baseline.

Chi tiết: `reports/vif_train.csv` và `reports/correlation_train.csv`.

## 7. Target cực trị và quy tắc P1–P99

Project Charter có nêu hướng xử lý giá/m² theo phân vị 1%–99%. Để không dùng phân phối target của test khi đặt ngưỡng, notebook chỉ tính ngưỡng trên `y_train`:

- P1 train ≈ **4,39 triệu VNĐ/m²**.
- P99 train ≈ **463,08 triệu VNĐ/m²**.
- 56/2.784 dòng train (≈2,01%) được **gắn cờ chẩn đoán**.

Các dòng này **không bị tự động xóa** trong bước của Tuấn. Lý do: nếu tính hoặc điều chỉnh ngưỡng sau khi nhìn test/CV validation sẽ làm đánh giá lạc quan. Modeling Lead có thể thử `log1p(target)` bằng `build_model_pipeline(..., log_target=True)` và phải báo metric trên thang VNĐ/m² gốc.

## 8. File bàn giao

- `notebooks/07_model_preprocessing.ipynb`
- `src/modeling/preprocess.py`
- `src/modeling/__init__.py`
- `data/processed/modeling_base.csv`
- `data/processed/model_train_raw.csv`
- `data/processed/model_test_raw.csv`
- `reports/preprocessing_scope_audit.csv`
- `reports/feature_contract.csv`
- `reports/correlation_train.csv`
- `reports/vif_train.csv`
- `reports/target_quantile_train_diagnostic.csv`

## 9. Checklist trước khi bàn giao cho Modeling Lead

- [x] Target chỉ từ giá/diện tích gốc quan sát thực.
- [x] Duplicate theo link đã loại trước split.
- [x] Tin cho thuê rõ ràng bị loại khỏi scope.
- [x] Train/test không trùng `link_nguon`.
- [x] Giá/target không nằm trong X.
- [x] Imputation/scaling/encoding nằm trong `scikit-learn Pipeline`.
- [x] VIF/correlation chỉ tính từ train.
- [x] Hold-out test không được dùng để fit preprocessing hoặc đặt ngưỡng percentile.
