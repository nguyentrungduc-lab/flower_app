import streamlit as st
import requests
from PIL import Image
import os

FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")

# Cấu hình trang
st.set_page_config(
    page_title="Nhận diện loài hoa - ResNet-18",
    page_icon="🌸",
    layout="wide"
)

# Tiêu đề ứng dụng
st.title("🌸 Ứng dụng Nhận diện Loài hoa")
st.caption("Học phần: Lập trình Web nâng cao · Phenikaa Applied AI Lab")
st.markdown("---")

# Kiểm tra kết nối Backend FastAPI
try:
    health_res = requests.get(f"{FASTAPI_URL}/health", timeout=2)
    backend_online = health_res.status_code == 200
except Exception:
    backend_online = False

if not backend_online:
    st.warning("⚠️ Không thể kết nối tới server FastAPI (http://localhost:8000). Vui lòng chạy backend trước khi thao tác.")

# Bố cục 2 cột
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Tải ảnh đầu vào")
    uploaded_file = st.file_uploader(
        "Chọn một tấm ảnh hoa (Daisy, Dandelion, Roses, Sunflowers, Tulips)", 
        type=["jpg", "jpeg", "png"]
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Ảnh đã chọn", use_container_width=True)
        
        btn_classify = st.button("🔍 Phân loại ảnh", type="primary", disabled=not backend_online, use_container_width=True)
        
        if btn_classify:
            with st.spinner("Đang đưa ảnh qua ResNet-18 xử lý..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    response = requests.post(f"{FASTAPI_URL}/api/classify", files=files, timeout=10)
                    
                    if response.status_code == 200:
                        st.session_state["result"] = response.json()
                    else:
                        st.error(f"Lỗi Backend ({response.status_code}): {response.text}")
                except Exception as e:
                    st.error(f"Không thể gọi API: {e}")

with col2:
    st.subheader("2. Kết quả nhận diện")
    if "result" in st.session_state:
        res = st.session_state["result"]
        label = res.get("label", "N/A").upper()
        confidence = res.get("confidence", 0.0) * 100
        probabilities = res.get("probabilities", {})

        # Hiển thị kết quả chính
        st.success(f"### Kết quả: **{label}**")
        st.metric(label="Độ tin cậy (Confidence)", value=f"{confidence:.2f}%")

        # Hiển thị biểu đồ phân bố xác suất các lớp
        st.write("---")
        st.write("**Xác suất chi tiết cho từng loài hoa:**")
        
        # Sắp xếp theo xác suất giảm dần
        sorted_probs = dict(sorted(probabilities.items(), key=lambda item: item[1], reverse=True))
        
        for cls_name, prob in sorted_probs.items():
            st.write(f"**{cls_name.capitalize()}**: {prob * 100:.2f}%")
            st.progress(float(prob))
    else:
        st.info("👈 Hãy tải ảnh lên và nhấn nút **Phân loại ảnh** để xem kết quả tại đây.")