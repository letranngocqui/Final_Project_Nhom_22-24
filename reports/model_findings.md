# Kết luận diễn giải mô hình & độ bền — Bài tập 4

**Người phụ trách:** Nguyễn Hùng Phúc — Interpretability & Robustness Lead
**Nguồn số liệu:** `notebooks/09_model_interpretability.ipynb` (seed 42; scikit-learn 1.8.0).
**Mô hình được diễn giải:** RandomForest do Mạnh chọn (`models/best_model.joblib`; test MAE 53,82 triệu VNĐ/m², khớp `model_comparison.csv`) và Ridge (`alpha = 10`) cho phần hệ số.
**Đơn vị:** triệu VNĐ/m². Residual = thực tế − dự đoán (dương = mô hình dự đoán thấp). Tất cả là quan hệ **dự báo trên giá rao**, không phải nhân quả.

---

## 1. Năm kết luận nghiệp vụ

| # | Kết luận | Bằng chứng chính |
|---|---|---|
| 1 | **Loại tài sản và vùng quyết định mặt bằng giá/m² hơn quy mô.** Đất nền, căn hộ và hai vùng ngoài TP.HCM thấp hơn rõ rệt; biệt thự chưa có bằng chứng khác nhà phố. | Ridge (CI 95% bootstrap): đất nền −60,5 (−68,7…−51,7); Bình Dương −60,3 (−67,4…−51,8); BR–VT −54,9 (−62,4…−45,8); căn hộ −53,2 (−59,9…−46,1); biệt thự +6,2 (−4,3…+17,0, cắt 0). Trung vị đất nền: 8,6 (BR–VT) so với 58,1 (TP.HCM). |
| 2 | **Vị trí cấp phường/xã là thông tin còn thiếu lớn nhất.** Nhóm khu vực 3 mức quá thô. | Thêm `phuong_xa`: CV MAE RF 51,7 → 37,4 (−27,6%); GroupKFold 52,5 → 38,1 (−27,4%). Test (xác nhận một lần): MAE 53,8 → 38,2; R² 0,25 → 0,57; tin không có bản sao: 53,95 → 38,14. |
| 3 | **Đất nền là loại tin dự đoán kém nhất** (không phải nhà phố, dù MAE nhà phố lớn nhất). | WAPE đất nền 58,7% (CI 49,3–69,7) so với 43,8% chung — nhóm duy nhất CI nằm hoàn toàn trên mức chung. Hệ số biến thiên giá 1,02 (cao nhất); 96% tin không ghi số phòng ngủ. Thêm `phuong_xa`: WAPE đất nền → 34,5%. |
| 4 | **Căn nhỏ có đơn giá/m² cao hơn; số phòng ngủ và việc không ghi phòng ngủ đều mang thông tin.** | Mỗi +10 m²: −3,16 (−3,67…−2,66). Trung vị train ≤40 m² ≈ 157 so với >200 m² ≈ 76. Bias +12 đến +16 ở nhóm ≤60 m² (dự đoán thấp). +1 phòng ngủ: +2,79 (2,21…3,58); tin không ghi phòng ngủ: +33,7 (26,8…40,5) sau khi kiểm soát loại BĐS. |
| 5 | **Mô hình hiện tại chỉ nên là khoảng tham chiếu, chưa phải định giá.** | R² test 0,25; WAPE gộp 43,8%; 5% tin sai nhiều nhất = 21,3% tổng sai số, 10% = 33,5%. Chỉ 1/12 phân khúc đủ cỡ mẫu (n≥30) tách được khỏi mức chung. |

---

## 2. Importance

**Permutation importance (RandomForest, test, 30 lần lặp, tăng MAE khi xáo trộn):**

| Feature | RF test | ± std | RF train | Ridge test | MDI (%) | Hạng perm / MDI |
|---|---:|---:|---:|---:|---:|:--:|
| `dien_tich_goc` | 8,42 | 1,00 | 16,81 | 2,89 | 52,5 | 1 / 1 |
| `loai_bds` | 7,24 | 0,83 | 11,88 | 7,89 | 19,8 | 2 / 2 |
| `so_phong_ngu` | 6,69 | 0,85 | 11,22 | 3,67 | 13,5 | 3 / 4 |
| `khu_vuc_nhom` | 6,05 | 0,75 | 7,22 | 5,31 | 14,1 | 4 / 3 |

Đọc bảng:
- **Bốn feature đóng góp gần ngang nhau** (21–30% mỗi feature); khoảng ±1 SD của các cặp liền kề chồng lấn (chỉ diện tích và khu vực là tách nhẹ) nên thứ hạng **chỉ mang tính tương đối**. Mô hình không dựa vào một biến duy nhất; điểm nghẽn là *thiếu thông tin*, không phải thiếu độ phức tạp.
- **MDI thổi phồng `dien_tich_goc`** (52,5% so với 29,6% theo permutation) vì biến liên tục có nhiều ngưỡng chia → không dùng MDI để kết luận.
- **Diện tích: RF dùng nhiều (8,42) nhưng Ridge gần như không (2,89)** → quan hệ phi tuyến/tương tác; đúng với Pearson thấp (−0,19) nhưng RF vẫn khai thác.
- **Train gấp ~1,5–2 lần test** ở diện tích và số phòng ngủ (16,8 vs 8,4; 11,2 vs 6,7) → RF phần nào ghi nhớ train, phù hợp khoảng cách train–CV 14% Mạnh đã báo.
- Đối chiếu Tuấn: VIF lớn nhất 1,22 (≪5) → hệ số/importance không bị méo bởi đa cộng tuyến, không cần gộp feature. Spearman của `so_phong_ngu` (0,455) ≫ Pearson (0,204) và tỷ lệ thiếu phòng ngủ thay đổi mạnh theo loại BĐS (đất nền 96%, căn hộ 36%) → thiếu **không ngẫu nhiên**.

Bảng đầy đủ: `reports/importance_table.csv`, `reports/linear_coefficients.csv`.

---

## 3. Lỗi theo phân khúc (RandomForest, test; chi tiết `reports/error_by_segment.csv`)

WAPE chung 43,8%. Phân loại: CI 95% bootstrap của WAPE so với mức chung; n < 30 → không kết luận.

| Phân khúc | n | MAE | WAPE % (CI 95%) | Bias | Đánh giá |
|---|---:|---:|---|---:|---|
| BR–VT (mới sáp nhập) | 29 | 28,8 | 65,7 (45,8–94,0) | −16,8 | không đủ dữ liệu |
| Bình Dương (mới sáp nhập) | 49 | 21,0 | 46,3 (34,1–61,0) | −2,9 | tương đương |
| TP.HCM (lõi cũ) | 618 | 57,6 | 43,4 (40,4–46,6) | +4,7 | tương đương |
| Biệt thự | 35 | 43,8 | 38,3 (28,9–49,7) | −0,6 | tương đương |
| Căn hộ/Chung cư | 97 | 32,5 | 49,3 (41,2–58,0) | −1,3 | tương đương |
| Khác | 97 | 46,9 | 40,3 (34,1–47,9) | −3,0 | tương đương |
| Nhà ở/Nhà phố | 394 | 64,1 | 43,0 (39,2–46,5) | +6,3 | tương đương |
| **Đất nền** | 73 | 40,9 | **58,7 (49,3–69,7)** | +3,3 | **kém** |
| ≤40 m² | 76 | 67,5 | 37,9 (30,0–46,4) | +12,0 | tương đương |
| 40–60 m² | 137 | 56,2 | 43,7 (37,1–50,8) | +15,6 | tương đương |
| 60–80 m² | 178 | 51,9 | 45,2 (39,7–51,5) | −7,9 | tương đương |
| 80–120 m² | 179 | 53,5 | 43,2 (38,8–47,9) | +6,4 | tương đương |
| 120–200 m² | 98 | 49,5 | 51,3 (43,5–60,5) | −5,2 | tương đương |
| >200 m² | 28 | 34,1 | 40,7 (25,8–60,3) | +0,3 | không đủ dữ liệu |

- **Không có phân khúc nào “tốt” có ý nghĩa thống kê**; “biệt thự” và “≤40 m²” có WAPE thấp nhất nhưng CI vẫn chạm mức chung.
- **MAE không so sánh được giữa phân khúc** (phụ thuộc mặt bằng giá): nhà phố MAE cao nhất (64,1) nhưng WAPE bình thường; đất nền MAE vừa phải nhưng WAPE cao nhất. Dùng WAPE/MdAPE.
- BR–VT có WAPE điểm cao nhất (65,7%) nhưng chỉ 29 tin, CI rất rộng; hướng *dự đoán cao* (bias −16,8) nhất quán với việc nhóm này nhỏ (113 tin train) và có hệ số biến thiên 0,97.
- **Giải thích bằng dữ liệu (train):** nhóm kém có giá phân tán cao và thông tin mô tả thấp (đất nền: CV 1,02; thiếu phòng ngủ 96%); đất nền chênh >6 lần giữa BR–VT (8,6) và TP.HCM (58,1) trong khi mô hình chỉ biết 3 nhóm vùng.

---

## 4. Đề xuất feature mới (chỉ giữ khi CV chứng minh)

Quy tắc: giảm MAE theo cặp fold ≥ 2 SE (đặt trước) **và** giảm ≥ 1% (bổ sung sau khi xem kết quả — đã ghi rõ trong notebook) **và** còn cải thiện dưới GroupKFold. Test không dùng để chọn. Bảng đầy đủ: `reports/feature_candidates_cv.csv`.

| Ứng viên | CV MAE RF | Giảm | GroupKFold giảm | Quyết định |
|---|---:|---:|---:|---|
| baseline | 51,67 | — | — | — |
| `phuong_xa` (TargetEncoder trong Pipeline) | 37,41 | **27,6%** | 27,4% | **Đề xuất giữ** |
| `phuong_xa` + 3 cờ tiêu đề | 36,18 | 30,0% | 30,6% | Đề xuất giữ (cờ là phần phụ) |
| `mat_tien` | 50,56 | 2,2% | 3,1% | Giữ, mức nhỏ |
| `so_hong` | 51,07 | 1,2% | 0,8% | Biên — qua quy tắc CV nhưng GroupKFold chỉ 0,8%, không ưu tiên |
| `hem` | 51,50 | 0,3% | 0,3% | Bỏ |
| `dt_per_pn` | 51,44 | 0,5% | — | Bỏ (<2 SE) |
| `log_dien_tich` | 51,58 | 0,2% | — | Bỏ (RF bất biến với phép đơn điệu) |

**Kiểm tra “vì sao cải thiện”:** nếu `phuong_xa` thật sự thêm thông tin địa lý thì phân khúc thô nhất phải cải thiện nhiều nhất — đúng như vậy: đất nền WAPE 58,7 → 34,5; BR–VT 65,7 → 46,9; biệt thự 38,3 → 22,4; căn hộ cải thiện ít nhất (49,3 → 42,5, có thể vì giá căn hộ phụ thuộc dự án hơn là phường — giả thuyết, chưa kiểm chứng).

**Khi biết phường/xã, `khu_vuc_nhom` gần như hết vai trò:** permutation importance trên test của `vi_tri → phuong_xa` ≈ 30,8 (gấp ~5 lần feature lớn nhất còn lại), còn `khu_vuc_nhom` chỉ ≈ 1,4 — nhóm vùng 3 mức đã bị phường/xã bao trùm (`reports/figures/09_importance_with_phuong_xa.png`).

**Lưu ý chống leakage:** `TargetEncoder` nằm trong Pipeline (cross-fitting nội bộ, fit lại ở từng fold); 99,6% tin test có phường xuất hiện trong train (phường lạ rơi về trung bình toàn cục). Cờ tiêu đề là phép biến đổi từng dòng, không học từ target.

**Đây là đề xuất, chưa phải mô hình chính thức:** `vi_tri`/`tieu_de` hiện là `FORBIDDEN_CURRENT` trong contract của Tuấn. Muốn dùng cần Tuấn cập nhật contract + `preprocess.py`, Mạnh tune/chọn lại (test mở một lần cuối). Các số test ở trên (`reports/feature_candidates_test_confirmation.csv`) chỉ là xác nhận.

---

## 5. Hạn chế và độ bền

- **Giá rao ≠ giá giao dịch**; dữ liệu là snapshot một thời điểm; mọi kết luận là quan hệ dự báo, không nhân quả.
- **Tin trùng nội dung:** train còn 147 bản sao theo (`gia_goc`, `dien_tich_goc`, `vi_tri`); 43/696 tin test có bản sao trong train. Cải thiện của `phuong_xa` giữ nguyên với GroupKFold và với tin test không có bản sao → không do trùng lặp. Bản sao *gần giống* (khác giá nhẹ) vẫn có thể còn sót.
- **Phân khúc nhỏ:** BR–VT (29) và >200 m² (28) trong test không đủ để kết luận; CI rộng ở hầu hết nhóm.
- **Thiếu không ngẫu nhiên:** `so_phong_ngu` thiếu 96% ở đất nền; hệ số “không ghi phòng ngủ” (+33,7) chỉ xuất hiện sau khi kiểm soát loại BĐS (trung vị thô gần như bằng nhau: 97,2 vs 101,4) và **chưa có giải thích đã kiểm chứng** — chỉ nên coi là tín hiệu dự báo.
- **Nhóm `loai_bds = Khác` (97 tin test)** không có nghĩa nghiệp vụ rõ; cải thiện phân loại từ tiêu đề là hướng mở.
- Số liệu sinh ra trên scikit-learn 1.8.0; file joblib của Mạnh lưu bằng 1.7.2 nên notebook tự dựng lại mô hình từ siêu tham số đã khoá nếu không nạp được, và kiểm tra khớp MAE test.

## 6. Tệp bàn giao

`notebooks/09_model_interpretability.ipynb` · `reports/importance_table.csv` · `reports/linear_coefficients.csv` · `reports/error_by_segment.csv` · `reports/error_by_segment_with_phuong_xa.csv` · `reports/feature_candidates_cv.csv` · `reports/feature_candidates_test_confirmation.csv` · `reports/model_findings.md` · `reports/figures/09_*.png` (4 hình)
