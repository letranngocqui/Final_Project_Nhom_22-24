"""Pipeline dự đoán giá nhà đất TP.HCM (CafeLand).
Thực thi quy trình: Crawl (Tùy chọn) -> Clean -> Export
"""
import argparse
import logging
import sys
import time
from pathlib import Path
import pandas as pd

# Tận dụng sức mạnh của file __init__.py để import gọn gàng
from src import scrape_cafeland_to_csv, clean_dataframe, save_processed_data

BASE_DIR = Path(__file__).resolve().parent
RAW_PATH = BASE_DIR / "data" / "raw" / "cafeland_raw.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

BASE_URL = "https://nhadat.cafeland.vn/nha-dat-ban-tai-tp-ho-chi-minh"
MIN_ROWS = 5000
MIN_COLS = 8

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)
log = logging.getLogger("pipeline")


def step_crawl(start_page: int, end_page: int) -> None:
    """Bước 1: Cào dữ liệu (Chỉ chạy khi có cờ --crawl)."""
    log.info(f"Bắt đầu crawl dữ liệu từ trang {start_page} đến {end_page}...")

    # Không xóa/đổi tên file cũ để tránh rủi ro mất dữ liệu, chỉ báo log
    if RAW_PATH.exists():
        log.warning("Đang ghi đè/Bổ sung vào file raw hiện tại.")

    scrape_cafeland_to_csv(
        base_url=BASE_URL,
        start_page=start_page,
        end_page=end_page,
        output_filepath=str(RAW_PATH),
    )


def step_load_raw() -> pd.DataFrame:
    """Đọc và kiểm định quy mô dữ liệu thô."""
    if not RAW_PATH.exists() or RAW_PATH.stat().st_size == 0:
        raise FileNotFoundError(f"Không tìm thấy dữ liệu thô tại: {RAW_PATH}. Vui lòng chạy thêm cờ --crawl")

    df = pd.read_csv(RAW_PATH, encoding="utf-8-sig")
    log.info("Nạp dữ liệu thô thành công: %d dòng x %d cột", *df.shape)

    if len(df) < MIN_ROWS:
        log.warning("Dữ liệu chỉ có %d dòng (Yêu cầu đồ án: >= %d dòng)", len(df), MIN_ROWS)
    if df.shape[1] < MIN_COLS:
        log.warning("Dữ liệu chỉ có %d biến (Yêu cầu đồ án: >= %d biến)", df.shape[1], MIN_COLS)
    return df


def step_clean(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Bước 2: Gọi module parser để làm sạch."""
    log.info("Đang tiến hành làm sạch dữ liệu...")
    df_clean = clean_dataframe(df_raw)
    log.info("Làm sạch thành công. Kích thước mới: %d dòng x %d cột", *df_clean.shape)
    if df_clean.empty:
        raise ValueError("LỖI NGHIÊM TRỌNG: Dữ liệu sau khi làm sạch bị rỗng!")
    return df_clean


def step_export(df_clean: pd.DataFrame) -> None:
    """Bước 3: Gọi module io_manager để xuất file."""
    log.info("Đang xuất dữ liệu ra các định dạng (CSV, JSON, Parquet)...")
    # Xóa khối lệnh raise RuntimeError. Bàn giao toàn quyền cho io_manager
    save_processed_data(df_clean)


def run(do_crawl: bool, start_page: int, end_page: int) -> None:
    t0 = time.time()

    # Mặc định bỏ qua Crawl để khi chấm bài sẽ nhìn thấy kết quả nhanh hơn từ file đã crwawl
    if do_crawl:
        log.info("[1/3] BƯỚC CRAWL DỮ LIỆU ĐƯỢC KÍCH HOẠT")
        step_crawl(start_page, end_page)
    else:
        log.info("[1/3] Bỏ qua cào dữ liệu (Sử dụng dữ liệu RAW đã có sẵn).")

    # Pipeline nối tiếp
    df_raw = step_load_raw()

    log.info("[2/3] BƯỚC LÀM SẠCH")
    df_clean = step_clean(df_raw)

    log.info("[3/3] BƯỚC XUẤT FILE")
    step_export(df_clean)

    log.info("HOÀN TẤT PIPELINE XUẤT SẮC TRONG %.1f GIÂY!", time.time() - t0)


def main() -> None:
    # Cấu hình lại Argparse cho thân thiện
    ap = argparse.ArgumentParser(description="Pipeline CafeLand: Crawl (Tùy chọn) -> Clean -> Export")
    ap.add_argument("--crawl", action="store_true",
                    help="Kích hoạt chế độ cào dữ liệu web (Rất tốn thời gian). Mặc định là KHÔNG cào.")
    ap.add_argument("--start-page", type=int, default=1)
    ap.add_argument("--end-page", type=int, default=250)
    args = ap.parse_args()

    try:
        run(args.crawl, args.start_page, args.end_page)
    except Exception as e:
        log.exception("Pipeline buộc phải dừng do lỗi hệ thống.")
        sys.exit(1)


if __name__ == "__main__":
    main()
