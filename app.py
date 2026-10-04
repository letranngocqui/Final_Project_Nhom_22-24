import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import os

st.set_page_config(page_title="Hệ thống Định giá BĐS CafeLand", layout="wide")
st.title("🏡 Demo: Định giá nhanh Bất Động Sản")
st.markdown("Nhập các thông số của bất động sản để mô hình RandomForest dự đoán mức giá trên mỗi m².")

@st.cache_resource
def load_model():
    return joblib.load('models/best_model.joblib')

model = load_model()

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Thông tin Bất Động Sản")
    dien_tich_goc = st.number_input("Diện tích (m²)", min_value=10.0, max_value=2000.0, value=60.0, step=1.0)
    so_phong_ngu = st.number_input("Số phòng ngủ", min_value=1, max_value=20, value=2, step=1)

    khu_vuc = st.selectbox("Nhóm Khu vực",
                           ["TP.HCM (lõi cũ)", "Bình Dương (mới sáp nhập)", "Bà Rịa - Vũng Tàu (mới sáp nhập)"])

    loai_bds = st.selectbox("Loại hình BĐS",
                            ["Căn hộ/Chung cư", "Nhà ở/Nhà phố", "Đất nền", "Biệt thự", "Khác"])

    predict_btn = st.button("Dự đoán Giá/m²", type="primary")

if predict_btn:
    input_df = pd.DataFrame({
        'dien_tich_goc': [dien_tich_goc],
        'so_phong_ngu': [so_phong_ngu],
        'khu_vuc_nhom': [khu_vuc],
        'loai_bds': [loai_bds]
    })

    try:
        prediction_vnd = model.predict(input_df)[0]
        prediction_trieu = prediction_vnd / 1_000_000

        with col2:
            st.success("Hệ thống đã xử lý xong!")
            st.metric(label="Đơn giá dự kiến (Triệu VNĐ/m²)", value=f"{prediction_trieu:,.2f} Triệu")

            tong_gia = prediction_trieu * dien_tich_goc
            st.info(f"💰 Ước tính Tổng giá trị BĐS: **{tong_gia:,.2f} Triệu VNĐ** (tương đương {tong_gia/1000:,.2f} Tỷ VNĐ)")
            st.caption("Lưu ý: Mức giá này được dự báo bởi mô hình RandomForest và chỉ mang tính chất tham khảo.")

    except Exception as e:
        st.error(f"Có lỗi xảy ra: {e}")

st.markdown("---")
st.header("📊 Đánh giá chất lượng Mô hình (Trên Tập Test)")

@st.cache_data
def load_and_predict_test_data():
    test_path = 'data/processed/model_test_raw.csv'
    if os.path.exists(test_path):
        df_test = pd.read_csv(test_path)

        X_test = df_test[['dien_tich_goc', 'so_phong_ngu', 'khu_vuc_nhom', 'loai_bds']]

        df_test['Giá Dự Đoán (Triệu/m²)'] = model.predict(X_test) / 1_000_000
        df_test['Giá Thực Tế (Triệu/m²)'] = df_test['price_per_m2'] / 1_000_000
        return df_test
    return None

df_test_plot = load_and_predict_test_data()

if df_test_plot is not None:
    fig = px.scatter(df_test_plot, x='Giá Thực Tế (Triệu/m²)', y='Giá Dự Đoán (Triệu/m²)',
                     opacity=0.6, color='loai_bds',
                     hover_data=['dien_tich_goc', 'so_phong_ngu', 'khu_vuc_nhom'])

    max_val = max(df_test_plot['Giá Thực Tế (Triệu/m²)'].max(), df_test_plot['Giá Dự Đoán (Triệu/m²)'].max())
    fig.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val,
                  line=dict(color="red", dash="dash"))

    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Vui lòng đảm bảo file tập test nằm tại: `data/processed/model_test_raw.csv` để hiển thị biểu đồ.")
