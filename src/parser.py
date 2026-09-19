"""Làm sạch dữ liệu CafeLand bằng Pandas và NumPy."""
import re
from pathlib import Path

import numpy as np
import pandas as pd


def clean_text(value):
    """Xóa khoảng trắng thừa và đổi chuỗi rỗng thành giá trị thiếu."""
    if pd.isna(value):
        return np.nan

    value = str(value)
    value = re.sub(r"\s+", " ", value).strip()
    return value if value else np.nan


def convert_number(text):
    """Đổi chuỗi số kiểu Việt Nam thành float.

    Ví dụ: '2,5' thành 2.5 và '1.200' thành 1200.
    """
    if pd.isna(text):
        return np.nan

    text = str(text).strip().replace(" ", "")

    if "," in text and "." in text:
        # Ví dụ: 1.234,5
        text = text.replace(".", "").replace(",", ".")
    elif "," in text:
        # Ví dụ: 2,5
        text = text.replace(",", ".")
    elif "." in text and len(text.split(".")[-1]) == 3:
        # Ví dụ: 1.200
        text = text.replace(".", "")

    try:
        return float(text)
    except ValueError:
        return np.nan


def parse_price(value):
    """Đổi giá dạng '2 tỷ 990 triệu' thành số tiền Việt Nam."""
    if pd.isna(value):
        return np.nan

    text = str(value).lower().replace("đ", "")
    total = 0

    ty = re.search(r"([\d.,]+)\s*tỷ", text)
    trieu = re.search(r"([\d.,]+)\s*triệu", text)
    nghin = re.search(r"([\d.,]+)\s*(nghìn|ngàn)", text)

    if ty:
        total += convert_number(ty.group(1)) * 1_000_000_000
    if trieu:
        total += convert_number(trieu.group(1)) * 1_000_000
    if nghin:
        total += convert_number(nghin.group(1)) * 1_000

    if total > 0:
        return total

    # Trường hợp giá chỉ có một con số, ví dụ: "850 triệu" đã xử lý ở trên.
    number = re.search(r"[\d.,]+", text)
    return convert_number(number.group()) if number else np.nan


def parse_area(value):
    """Đổi diện tích dạng '72m2' hoặc '72 m²' thành float."""
    if pd.isna(value):
        return np.nan

    number = re.search(r"[\d.,]+", str(value))
    return convert_number(number.group()) if number else np.nan


def normalize_location(value):
    """Chuẩn hóa tên thành phố, quận, phường và xã."""
    value = clean_text(value)
    if pd.isna(value):
        return np.nan

    value = str(value)

    value = re.sub(r"\b(?:P\.|Phường)\s*", "Phường ", value, flags=re.IGNORECASE)
    value = re.sub(r"\b(?:X\.|Xã)\s*", "Xã ", value, flags=re.IGNORECASE)
    value = re.sub(r"\b(?:Q\.|Quận)\s*", "Quận ", value, flags=re.IGNORECASE)
    value = re.sub(
        r"\b(?:TP\.|Thành phố)\s*", "Thành phố ", value, flags=re.IGNORECASE
    )
    value = re.sub(r"\s*,\s*", ", ", value)
    return re.sub(r"\s+", " ", value).strip()

def extract_bedrooms(text):
    """Bóc tách số lượng phòng ngủ từ chuỗi văn bản cực mạnh.
    Bắt các trường hợp: '2PN', '2 Phòng ngủ', '25 phòng', '2 P', '2p'
    """
    if pd.isna(text):
        return np.nan

    text = str(text)

    match = re.search(r'(\d+)\s*(?:pn|phòng\s*ngủ|phòng|p\b)', text, flags=re.IGNORECASE)

    return float(match.group(1)) if match else np.nan

def clean_dataframe(data):
    """Làm sạch các cột chính và trả về DataFrame mới."""
    data = data.copy()

    # 1. BẢO VỆ DATAFRAME: Xóa các dòng rỗng hoàn toàn do bot vướng quảng cáo
    data = data.dropna(how='all')

    # Xóa khoảng trắng ở tên cột và trong các cột chữ.
    data.columns = data.columns.str.strip().str.lower()
    text_columns = ["tieu_de", "vi_tri", "ngay_dang", "link_nguon", "mo_ta", "nguoi_ban"]
    for column in text_columns:
        if column in data:
            data[column] = data[column].map(clean_text)

    # 2. Gọi hàm bóc tách phòng ngủ để ráp vào Bài tập 2
    if "tieu_de" in data and "mo_ta" in data:
        # Tạo một cột tạm chứa cả 2 chuỗi văn bản
        combined_text = data["tieu_de"].fillna("") + " " + data["mo_ta"].fillna("")
        data["so_phong_ngu"] = combined_text.map(extract_bedrooms)
    elif "mo_ta" in data:
        # Dự phòng nếu file không có cột tiêu đề
        data["so_phong_ngu"] = data["mo_ta"].map(extract_bedrooms)

    # 3. Đưa diện tích về số TRƯỚC để làm tham số tính Giá
    if "dien_tich" in data:
        data["dien_tich"] = data["dien_tich"].map(parse_area).astype(float)

    # 4. GHI ĐIỂM NUMPY (VECTORIZATION) & XỬ LÝ OUTLIER
    if "gia" in data and "dien_tich" in data:
        # Lấy con số thô từ hàm parse_price (Thương lượng -> NaN, 120 tr/m2 -> 120.000.000)
        parsed_price_raw = data["gia"].map(parse_price).astype(float)

        # Nhận diện nhanh các dòng niêm yết theo m2
        is_per_m2 = data["gia"].str.contains(r'/m2|/ m2|/m²|/ m²', case=False, na=False)

        # Vectorization: Nếu đúng là giá/m2 -> lấy Giá thô nhân Diện tích. Sai thì giữ nguyên Giá thô.
        data["gia"] = np.where(is_per_m2, parsed_price_raw * data["dien_tich"], parsed_price_raw)
    elif "gia" in data:
        data["gia"] = data["gia"].map(parse_price).astype(float)

    # Chuẩn hóa biến phân loại địa điểm và biến ngày tháng.
    if "vi_tri" in data:
        data["vi_tri"] = data["vi_tri"].map(normalize_location)
    if "ngay_dang" in data:
        data["ngay_dang"] = pd.to_datetime(
            data["ngay_dang"], errors="coerce", dayfirst=True
        )

    return data


def main():
    """Đọc file raw, làm sạch và lưu file processed."""
    root = Path(__file__).resolve().parents[1]
    input_file = root / "data" / "raw" / "cafeland_raw.csv"
    output_file = root / "data" / "processed" / "cafeland_cleaned.csv"

    raw_data = pd.read_csv(input_file, encoding="utf-8-sig")
    cleaned_data = clean_dataframe(raw_data)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    cleaned_data.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"Đã làm sạch {len(cleaned_data)} dòng dữ liệu.")
    print(f"File kết quả: {output_file}")


if __name__ == "__main__":
    main()
