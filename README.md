# Bài tập 2: Pipeline Thu thập & Tổ chức dữ liệu (CafeLand)
**Nhóm 22 & 24**

Dự án thu thập và làm sạch dữ liệu tin rao bán nhà đất tại TP.HCM trên `nhadat.cafeland.vn`. Phiên bản hiện tại đáp ứng yêu cầu Bài tập 2: Đạt quy mô >5.000 dòng, tổ chức đa định dạng và tự động hóa toàn bộ luồng chạy.

## 1. Cấu trúc thư mục (Tính đến bài tập 2)
```text
project/
├── data/
│   ├── raw/                     # Chứa cafeland_raw.csv (Dữ liệu thô)
│   └── processed/               # Chứa 3 file sạch (CSV, JSON, Parquet)
├── docs/
│   └── data_dictionary.md       # Từ điển mô tả 9 biến dữ liệu
|   └── project_charter.pdf      # Project Charter của dự án
├── src/
│   ├── __init__.py              # Package module
│   ├── crawler.py               # Module cào dữ liệu web
│   ├── parser.py                # Module làm sạch (Pandas/NumPy)
│   └── io_manager.py            # Module xuất file 
├── notebooks/
│   └── 01_data_quality.ipynb    # Đánh giá chất lượng dữ liệu thô
├── requirements.txt             # Thư viện cần cài đặt
├── README.md                    # File hướng dẫn này
└── run_pipeline.py              # File thực thi luồng chạy tự động
```

## 2. Hướng dẫn cài đặt & Chạy tự động 
Cài đặt môi trường:
```bash
pip install -r requirements.txt
```
### Cách 1: Chạy tự động luồng Làm sạch & Xuất file (Khuyên dùng)
Nhóm đã cào sẵn dữ liệu thô tại data/raw/cafeland_raw.csv. Để tiết kiệm thời gian chấm thi, lệnh dưới đây sẽ bỏ qua bước cào web, lập tức nạp dữ liệu thô để làm sạch và xuất ra 3 định dạng tại thư mục processed/ :
```bash
python run_pipeline.py
```

### Cách 2: Chạy kiểm chứng chế độ Cào dữ liệu (Crawler)
Nếu Giảng viên muốn kiểm tra code cào web thực tế, vui lòng sử dụng cờ --crawl kèm giới hạn số trang để test nhanh (tránh mất >20 phút do cơ chế sleep chống block):
```bash
python run_pipeline.py --crawl --start-page 1 --end-page 3
```
