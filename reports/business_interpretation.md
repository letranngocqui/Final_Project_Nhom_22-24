# BÀI TẬP 4 — DIỄN GIẢI KẾT QUẢ MÔ HÌNH THEO GÓC NHÌN NGHIỆP VỤ

**Đề tài:** Dự đoán đơn giá rao bán bất động sản từ dữ liệu Nhà Đất CafeLand  
**Nhóm:** 22 & 24  
**Phụ trách nội dung:** Lê Nhật Việt — Business Insight & Scientific Report Lead  
**Phạm vi:** Dự báo **đơn giá rao bán theo m²**, không phải giá giao dịch thực tế hoặc lời khuyên mua bán.  
**Quy ước:** Các giá trị MAE/RMSE và mức giá trong tài liệu tính bằng **triệu VNĐ/m²**, trừ khi ghi rõ khác; số liệu được làm tròn khi trình bày.

## 1. Kết luận tổng quan cho người ra quyết định

Nhóm đã so sánh bốn cấu hình hồi quy gồm LinearRegression, Ridge, Ridge với biến mục tiêu `log1p` và RandomForest. **RandomForest có kết quả tốt nhất trong các cấu hình đã thử**, với MAE bình quân qua 5-fold cross-validation (CV) **51,67 triệu VNĐ/m²** và MAE trên tập kiểm tra (test) **53,82 triệu VNĐ/m²**. Tuy nhiên, R² test mới đạt **0,247**, nên mô hình phù hợp nhất để **minh họa một công cụ ước lượng giá rao và hỗ trợ sàng lọc ban đầu**, chưa thích hợp để thay thế nghiệp vụ thẩm định giá.

Ba thông điệp cần chuyển đến người sử dụng và nhóm triển khai là:

1. **So sánh các tin rao trong cùng phân khúc** (khu vực và loại bất động sản) thay vì so với giá trung bình của toàn thị trường.
2. **Ưu tiên chuẩn hóa địa chỉ chi tiết**; thử nghiệm thêm phường/xã có cải thiện CV đáng kể nhưng **chưa được triển khai trong mô hình chính**.
3. **Công khai độ bất định theo phân khúc**, đặc biệt với đất nền và nhóm ít quan sát, thay vì hiển thị một dự đoán như giá trị chắc chắn.

**Căn cứ:** [So sánh mô hình](model_comparison.csv), [kết luận phân tích của Phúc](model_findings.md).

## 2. Vì sao chọn RandomForest và mức độ tin cậy hiện tại

| Cấu hình | MAE CV (5-fold) | MAE test | RMSE test | R² test |
|---|---:|---:|---:|---:|
| LinearRegression | 56,64 | 56,06 | 81,34 | 0,241 |
| Ridge | 56,62 | 56,11 | 81,42 | 0,240 |
| Ridge + `log1p` target | 54,04 | 55,41 | 85,02 | 0,171 |
| **RandomForest (mô hình đã chọn)** | **51,67** | **53,82** | **81,02** | **0,247** |

*Nguồn: [model_comparison.csv](model_comparison.csv). Chỉ số MAE/RMSE trong CSV gốc tính bằng VNĐ/m² và được đổi sang triệu VNĐ/m² trong bảng này.*

So với LinearRegression, RandomForest giảm MAE CV khoảng **8,77%** nhưng trên test chỉ giảm khoảng **4,00%**. Vì vậy, phát biểu chính xác là RandomForest **tốt nhất trong phạm vi bốn cấu hình được kiểm tra**, không phải thuật toán tốt nhất cho mọi bộ dữ liệu bất động sản.

MAE train của RandomForest là **44,41**, thấp hơn MAE CV **51,67** (khoảng cách bằng **14,07% MAE CV**). Kết quả này cùng learning curve cho thấy còn chênh lệch hiệu quả giữa dữ liệu đã học và dữ liệu chưa thấy; cần đánh giá thêm overfitting và khả năng tổng quát hóa. Bản demo hiện dùng mô hình RandomForest đã chọn, **không tự động sử dụng các đặc trưng ứng viên đang nghiên cứu**.

Hình tham khảo: [So sánh chỉ số](figures/model_comparison_metrics.png) · [Learning curve](figures/learning_curve_best_model.png) · [Residual analysis](figures/residual_analysis_best_model.png).

## 3. Ba insight nghiệp vụ trọng tâm

### Insight 1 — Một mức giá tham chiếu gộp dễ dẫn tới so sánh sai phân khúc

Trên **3.480 tin đủ điều kiện modeling**, trung vị đơn giá rao bán của các nhóm địa lý trong dữ liệu như sau:

| Nhãn nhóm khu vực trong bộ dữ liệu | Số tin | Trung vị đơn giá rao (triệu VNĐ/m²) |
|---|---:|---:|
| TP.HCM (lõi cũ) | 3.069 | 110,94 |
| Bình Dương (nhãn nhóm trong dữ liệu) | 269 | 47,62 |
| Bà Rịa – Vũng Tàu (nhãn nhóm trong dữ liệu) | 142 | 40,00 |

*Nguồn: `data/processed/modeling_base.csv`. Các nhãn địa lý là nhóm dùng trong phân tích, không dùng để kết luận pháp lý về địa giới hành chính.*

Khi kiểm soát những đặc trưng khác trong Ridge, hệ số của nhãn Bình Dương và Bà Rịa – Vũng Tàu lần lượt khoảng **−60,32** và **−54,86 triệu VNĐ/m²** so với nhãn tham chiếu TP.HCM lõi cũ. Hệ số cho **đất nền** khoảng **−60,47** so với nhà ở/nhà phố. Đây là **liên hệ có điều kiện trong mô hình**, không chứng minh địa điểm hoặc loại BĐS tự gây ra chênh lệch giá.

**Giá trị sử dụng:** Trước khi đánh giá tin đăng là “cao” hay “thấp”, cần lọc ít nhất theo **khu vực và loại hình**, sau đó kiểm tra các thuộc tính còn thiếu như vị trí chi tiết và tính pháp lý. Các trung vị nêu trên không đại diện cho toàn bộ giao dịch của thị trường.

### Insight 2 — Thông tin phường/xã là ứng viên cải thiện quan trọng

Mô hình chính đang sử dụng bốn đặc trưng: `dien_tich_goc`, `so_phong_ngu`, `khu_vuc_nhom`, `loai_bds`. Kiểm tra permutation importance của RandomForest cho thấy cả bốn đóng góp cho dự báo (tăng MAE test khi xáo trộn từng biến tương ứng **8,42; 6,69; 6,05; 7,24 triệu VNĐ/m²**), nhưng không đo lường tác động nhân quả.

Trong thử nghiệm mở rộng đặc trưng của Nguyễn Hùng Phúc, đưa `phuong_xa` vào pipeline bằng TargetEncoder cho các kết quả:

| RandomForest | MAE CV | MAE GroupKFold |
|---|---:|---:|
| Baseline, bốn đặc trưng | 51,67 | 52,45 |
| Thêm `phuong_xa` | 37,41 | 38,10 |
| Mức giảm so với baseline | **27,6%** | **27,4%** |

*Nguồn: [feature_candidates_cv.csv](feature_candidates_cv.csv). GroupKFold là kiểm tra độ bền bổ sung đối với nhóm tin có đặc điểm giống nhau.*

Thử nghiệm xác nhận ứng viên trên tập test cũng cho MAE **38,21** so với baseline **53,82**, nhưng kết quả này **không đồng nghĩa nhóm đã thay thế mô hình cuối cùng**. Hơn nữa, vì đã có phân tích ứng viên trên test, cải tiến cần được đánh giá lại trên dữ liệu độc lập trước khi tuyên bố hiệu quả triển khai.

**Giá trị sử dụng:** Ưu tiên làm sạch và chuẩn hóa thông tin địa chỉ cấp phường/xã. Chỉ khi nhóm cập nhật `feature_contract`, tiền xử lý, tái huấn luyện và kiểm định độc lập mới nên đưa biến này vào Streamlit demo hoặc báo cáo hiệu năng chính thức.

### Insight 3 — Cần cảnh báo sai số khác nhau giữa các loại bất động sản

Trên **696 tin test**, WAPE toàn bộ khoảng **43,8%**. Khi phân nhóm theo loại bất động sản:

| Loại BĐS | Số tin test | MAE | WAPE |
|---|---:|---:|---:|
| Đất nền | 73 | 40,86 | **58,74%** |
| Nhà ở/Nhà phố | 394 | 64,07 | 42,95% |
| Căn hộ/Chung cư | 97 | 32,48 | 49,29% |

*Nguồn: [error_by_segment.csv](error_by_segment.csv). WAPE = tổng sai số tuyệt đối / tổng giá trị thực, phù hợp để so sánh sai số tương đối ở các nhóm có mặt bằng giá khác nhau. WAPE không phải xác suất một dự báo bị sai.*

Mặc dù MAE của nhà phố lớn hơn, **đất nền có WAPE cao hơn đáng kể** trong bộ test. Khoảng bootstrap 95% WAPE đất nền khoảng **49,33–69,71%**. Báo cáo phân tích còn ghi nhận trong tập train khoảng **96% tin đất nền thiếu số phòng ngủ**, minh họa rằng thiết kế một tập đặc trưng chung có thể không thích hợp như nhau cho mọi loại tài sản.

**Giá trị sử dụng:** Trong demo và thuyết trình, hiển thị cảnh báo rằng độ tin cậy của dự báo thay đổi theo loại hình; với đất nền cần đối chiếu thêm hồ sơ và thông tin vị trí trước khi sử dụng kết quả. Không đưa ra kết luận mạnh với các nhóm chỉ có **29 tin** (Bà Rịa – Vũng Tàu) hoặc **28 tin** (diện tích trên 200 m²) trong test.

Hình tham khảo: [Sai số theo phân khúc](figures/09_error_by_segment.png) · [Permutation importance](figures/09_rf_importance.png).

## 4. Ứng dụng thực tiễn và hành động ưu tiên

| Ưu tiên | Đối tượng | Hành động | Lý do |
|---|---|---|---|
| Ngay | Người xem tin / môi giới | Lọc và so sánh theo vùng × loại BĐS, dùng kết quả như tham chiếu | Mặt bằng giá và sai số khác nhau giữa các nhóm |
| Ngay | Nhóm thiết kế demo | Ghi rõ “Ước lượng đơn giá rao tham khảo”; có cảnh báo sai số và phạm vi dữ liệu | MAE test 53,82; R² test 0,247 |
| Tiếp theo | Nhóm dữ liệu | Đối chiếu giá/diện tích với nguồn gốc, rà soát ngoại lai và tin gần trùng | Có bản ghi giá quy đổi bất thường và khả năng một tài sản xuất hiện nhiều tin |
| Tiếp theo | Nhóm mô hình | Chuẩn hóa `phuong_xa`, kiểm định trên dữ liệu mới rồi mới cân nhắc triển khai | MAE CV ứng viên giảm 27,6%, chưa có xác nhận triển khai độc lập |
| Dài hạn | Nhóm dự án | Bổ sung dữ liệu pháp lý, tiện ích và giao dịch nếu tiếp cận hợp lệ | Hiện chưa có đủ biến giải thích để dùng như công cụ thẩm định |

## 5. Hạn chế cần nêu rõ khi báo cáo và vấn đáp

- **Khác biệt giữa giá rao và giá giao dịch:** mô hình không ước lượng giá chuyển nhượng đã xác minh, thanh khoản hay lợi nhuận đầu tư.
- **Độ đại diện:** mẫu modeling tập trung tại TP.HCM lõi cũ (**3.069/3.480 tin, khoảng 88,2%**); các nhóm ngoài lõi cũ có ít quan sát hơn.
- **Chất lượng dữ liệu:** khảo sát `modeling_base.csv` ghi nhận **7 tin có `gia_goc` dưới 10 triệu VNĐ**, **10 tin có đơn giá dưới 1 triệu VNĐ/m²** và **25 tin trên 500 triệu VNĐ/m²**. Đây là tín hiệu cần kiểm tra, **không phải bằng chứng mọi dòng đều sai**.
- **Rò rỉ và trùng tin:** tiền xử lý không đưa giá gốc vào X khi dự báo giá/m² và fit trên train/fold; tuy vậy, khử trùng theo URL chưa chắc loại được mọi tin mô tả cùng một tài sản.
- **Diễn giải không đồng nghĩa nhân quả:** hệ số Ridge và permutation importance là kết quả thống kê/dự báo có điều kiện; không nên dùng để khẳng định “vị trí làm tăng giá X” theo nghĩa nhân quả.
- **Độ tin cậy khi triển khai:** app hiện trả kết quả dự đoán điểm, chưa có khoảng dự báo hiệu chuẩn; việc chuyển sang mô hình bổ sung phường/xã vẫn là phương án đề xuất.

## 6. Nội dung cô đọng để chuyển cho slide chung

**Slide thông điệp 1 — So sánh đúng phân khúc:** Trên tập modeling, trung vị giá rao TP.HCM lõi cũ **110,94**, Bình Dương **47,62**, Bà Rịa – Vũng Tàu **40,00 triệu VNĐ/m²**. *Hàm ý:* không dùng một mốc giá gộp cho mọi tin.

**Slide thông điệp 2 — Đầu tư vào chất lượng địa chỉ:** Thêm `phuong_xa` làm CV MAE RandomForest giảm **27,6%** trong thử nghiệm. *Hàm ý:* chuẩn hóa vị trí chi tiết có triển vọng, **chưa thuộc mô hình demo chính**.

**Slide thông điệp 3 — Minh bạch mức độ sai số:** RandomForest chính có MAE test **53,82 triệu VNĐ/m²**, R² **0,247**; đất nền WAPE **58,74%**. *Hàm ý:* chỉ hỗ trợ tham khảo và sàng lọc ban đầu, cần kiểm tra thực tế.

Ba thông điệp này đã nằm trong tài liệu chính; **không cần tạo thêm `key_business_insights.md`** chỉ để nộp phần nhiệm vụ của Việt.

## 7. Nguồn dữ liệu nội bộ và trách nhiệm sử dụng AI

- `notebooks/08_model_training_comparison.ipynb` và [model_comparison.csv](model_comparison.csv): mô hình, tuning, CV và test.
- `notebooks/09_model_interpretability.ipynb`, [model_findings.md](model_findings.md), [linear_coefficients.csv](linear_coefficients.csv), [importance_table.csv](importance_table.csv): hệ số, tầm quan trọng và diễn giải.
- [error_by_segment.csv](error_by_segment.csv), [feature_candidates_cv.csv](feature_candidates_cv.csv), [feature_candidates_test_confirmation.csv](feature_candidates_test_confirmation.csv): sai số theo nhóm và thử nghiệm cải tiến.
- `data/processed/modeling_base.csv` và [feature_contract.csv](feature_contract.csv): phạm vi dữ liệu và đặc trưng sử dụng.

**Minh bạch AI:** Nội dung tài liệu được hỗ trợ bởi công cụ AI trong việc tổng hợp, đối chiếu số liệu và diễn đạt từ dữ liệu có sẵn trong repository. Đây không phải kết quả mô hình do AI tự huấn luyện. Thành viên phụ trách cần đọc, xác nhận từng số liệu và tự giải thích kết luận trước khi ghép vào báo cáo chính thức; nhóm ghi nhận việc sử dụng AI trong phần khai báo chung.
