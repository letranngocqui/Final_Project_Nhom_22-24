"""Module quản lý việc xuất dữ liệu ra nhiều định dạng khác nhau."""

import logging
from pathlib import Path

import pandas as pd

# Cấu hình logging để báo cáo quá trình chạy
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def save_processed_data(df: pd.DataFrame):
    """
    Nhận DataFrame đã được làm sạch từ parser.py và xuất ra 3 định dạng.
    Đáp ứng Tiêu chí 3: Đọc/ghi nhiều định dạng theo cấu trúc thư mục rõ ràng.
    """
    if df.empty:
        logging.error("LỖI: DataFrame trống, không có dữ liệu để xuất.")
        return

    # Tự động dò tìm thư mục gốc của project (project)
    root_dir = Path(__file__).resolve().parents[1]
    output_dir = root_dir / "data" / "processed"

    # Tự động tạo thư mục processed nếu chưa tồn tại
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. XUẤT ĐỊNH DẠNG CSV
    csv_path = output_dir / "cafeland_cleaned.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logging.info(f"Đã xuất file CSV thành công: {csv_path}")

    # 2. XUẤT ĐỊNH DẠNG JSON
    json_path = output_dir / "cafeland_cleaned.json"
    df_json = df.copy()
    # Chuyển đổi cột datetime sang chuỗi chuẩn ISO để file JSON không bị lỗi
    if "ngay_dang" in df_json.columns and pd.api.types.is_datetime64_any_dtype(
        df_json["ngay_dang"]
    ):
        df_json["ngay_dang"] = df_json["ngay_dang"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    df_json.to_json(json_path, orient="records", force_ascii=False, indent=4)
    logging.info(f"Đã xuất file JSON thành công: {json_path}")

    # 3. XUẤT ĐỊNH DẠNG PARQUET (Tối ưu dung lượng, siêu tốc độ)
    parquet_path = output_dir / "cafeland_cleaned.parquet"
    try:
        df.to_parquet(parquet_path, index=False)
        logging.info(f"Đã xuất file Parquet thành công: {parquet_path}")
    except Exception as e:
        logging.warning(
            f"Không thể xuất file Parquet. Nhóm cần chạy lệnh 'pip install pyarrow' hoặc 'fastparquet'. Chi tiết lỗi: {e}"
        )


# Đoạn code dưới đây giúp Việt TỰ CHẠY ĐỘC LẬP để lấy 3 file nộp bài
# Khi Long import file này, đoạn code dưới đây sẽ bị bỏ qua (không gây lỗi đè luồng)
if __name__ == "__main__":
    logging.info("--- CHẾ ĐỘ TEST ĐỘC LẬP ---")

    # 1. Khởi tạo đường dẫn tìm file raw
    root_dir = Path(__file__).resolve().parents[1]
    raw_path = root_dir / "data" / "raw" / "cafeland_raw.csv"

    if raw_path.exists():
        logging.info("1. Đã tìm thấy dữ liệu thô. Đang nạp...")
        df_tho = pd.read_csv(raw_path, encoding="utf-8-sig")

        # 2. MƯỢN hàm làm sạch (Đảm bảo logic đồng nhất 100%)
        # Cần import trực tiếp từ parser.py trong cùng thư mục src
        try:
            from parser import clean_dataframe
            logging.info("2. Đang mượn parser.py để làm sạch dữ liệu...")
            df_sach = clean_dataframe(df_tho)

            # 3. Xuất 3 file!
            logging.info("3. Bắt đầu xuất 3 định dạng file...")
            save_processed_data(df_sach)

            logging.info("HOÀN THÀNH! Đã có đủ 3 file trong thư mục processed.")
        except ImportError:
            logging.error("Lỗi: Không tìm thấy file parser.py trong cùng thư mục. Hãy chắc chắn parser.py nằm cùng chỗ với io_manager.py.")
    else:
        logging.error(f"Lỗi: Không tìm thấy dữ liệu thô tại {raw_path}. Cần chạy crawler trước.")
