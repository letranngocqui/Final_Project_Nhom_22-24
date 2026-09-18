import csv
import os
import random
import time

import requests
from bs4 import BeautifulSoup


# 1. HÀM XỬ LÝ KẾT NỐI (Chống Block)
def get_html_soup(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup
    except requests.exceptions.RequestException as e:
        print(f"Lỗi mạng khi tải trang {url}: {e}")
        return None


# 2. HÀM BÓC TÁCH DỮ LIỆU (Đảm bảo 8 biến từ HTML thực tế)
def parse_property_item(item_soup):
    data = {}

    # 1. Tiêu đề (Thẻ a, class realTitle)
    try:
        data['tieu_de'] = item_soup.find('a', class_='realTitle').text.strip()
    except AttributeError:
        data['tieu_de'] = None

    # 2. Giá (Thẻ span, class reales-price)
    try:
        data['gia'] = item_soup.find('span', class_='reales-price').text.strip()
    except AttributeError:
        data['gia'] = None

    # 3. Diện tích (Thẻ span, class reales-area)
    try:
        data['dien_tich'] = item_soup.find('span', class_='reales-area').text.strip()
    except AttributeError:
        data['dien_tich'] = None

    # 4. Vị trí (Thẻ div, class info-location)
    try:
        data['vi_tri'] = item_soup.find('div', class_='info-location').text.strip()
    except AttributeError:
        data['vi_tri'] = None

    # 5. Ngày đăng (Lấy thuộc tính data-time từ thẻ div.reals-update-time để có giờ phút chính xác)
    try:
        time_tag = item_soup.find('div', class_='reals-update-time')
        data['ngay_dang'] = time_tag['data-time'] if time_tag else None
    except (AttributeError, KeyError):
        data['ngay_dang'] = None

    # 6. Link nguồn (Lấy thuộc tính href từ thẻ a.realTitle)
    try:
        link_tag = item_soup.find('a', class_='realTitle')
        data['link_nguon'] = link_tag['href'] if link_tag else None
    except (AttributeError, KeyError):
        data['link_nguon'] = None

    # 7. Mô tả ngắn (Thẻ div, class reales-preview - Bổ sung thay cho Loại hình)
    try:
        data['mo_ta'] = item_soup.find('div', class_='reales-preview').text.strip()
    except AttributeError:
        data['mo_ta'] = None

    # 8. Người bán (Thẻ a, class member-name - Bổ sung thay cho Số phòng)
    try:
        data['nguoi_ban'] = item_soup.find('a', class_='member-name').text.strip()
    except AttributeError:
        data['nguoi_ban'] = None

    return data


# 3. HÀM ĐIỀU KHIỂN & CHECKPOINT (Lưu trực tiếp vào thư mục raw/)
def scrape_cafeland_to_csv(base_url, start_page, end_page, output_filepath):
    print(f"Bắt đầu thu thập từ trang {start_page} đến {end_page}...")

    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    file_exists = os.path.isfile(output_filepath)

    with open(output_filepath, 'a', encoding='utf-8', newline='') as csvfile:
        fieldnames = [
            'tieu_de', 'gia', 'dien_tich', 'vi_tri',
            'ngay_dang', 'link_nguon', 'mo_ta', 'nguoi_ban'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        total_scraped = 0

        for page in range(start_page, end_page + 1):
            url = f"{base_url}/page-{page}/"
            print(f"Đang cào trang {page}...")

            soup = get_html_soup(url)
            if not soup:
                time.sleep(5)
                continue

            property_list_container = soup.find('div', class_='property-list')
            if property_list_container:
                property_items = property_list_container.find_all('div', class_='row-item')
            else:
                property_items = []

            for item in property_items:
                property_data = parse_property_item(item)
                writer.writerow(property_data)
                total_scraped += 1

            print(f"  -> Đã lưu trang {page}. Tổng số tin thu thập được: {total_scraped}")
            time.sleep(random.uniform(2.0, 4.0))

    print(f"\nHoàn thành! Đã thu thập {total_scraped} dòng vào {output_filepath}")


if __name__ == "__main__":
    OUTPUT_FILE = "../data/raw/cafeland_raw.csv"
    BASE_URL = "https://nhadat.cafeland.vn/nha-dat-ban-tai-tp-ho-chi-minh"

    # Cào thử 250 trang
    scrape_cafeland_to_csv(BASE_URL, start_page=1, end_page=250, output_filepath=OUTPUT_FILE)
