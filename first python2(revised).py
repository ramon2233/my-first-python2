import os
import cv2
import numpy as np
import streamlit as st

# 1. 핵심 이미지 처리 함수
def color_transfer(source_img, target_img):
    # LAB 색공간으로 변환 (색상 통계 분리가 용이함)
    src_lab = cv2.cvtColor(source_img, cv2.COLOR_BGR2LAB).astype("float32")
    tgt_lab = cv2.cvtColor(target_img, cv2.COLOR_BGR2LAB).astype("float32")

    # 평균과 표준편차 계산
    (src_mean, src_std) = cv2.meanStdDev(src_lab)
    (tgt_mean, tgt_std) = cv2.meanStdDev(tgt_lab)

    # 채널별 분리
    l, a, b = cv2.split(tgt_lab)
    
    # 색감 강도 조절 (사용자 설정 가능하게 하면 좋음)
    color_gain = 0.6 

    # 통계 기반 색상 전이 공식 적용
    # l, a, b 각각 연산 시 mean/std 값들을 스칼라로 확실히 추출
    l = (l - tgt_mean[0][0]) * (src_std[0][0] / (tgt_std[0][0] + 1e-5)) + src_mean[0][0]
    a = (a - tgt_mean[1][0]) * (src_std[1][0] / (tgt_std[1][0] + 1e-5) * color_gain) + src_mean[1][0]
    b = (b - tgt_mean[2][0]) * (src_std[2][0] / (tgt_std[2][0] + 1e-5) * color_gain) + src_mean[2][0]

    # 값 범위를 0~255로 제한 및 병합
    transfer = cv2.merge([l, a, b])
    transfer = np.clip(transfer, 0, 255).astype("uint8")
    result = cv2.cvtColor(transfer, cv2.COLOR_LAB2BGR)
    
    # 원본과 결과물을 적절히 합성하여 자연스럽게 만듦
    final_result = cv2.addWeighted(result, 0.7, target_img, 0.3, 0)
    return final_result

# 헬퍼 함수: 업로드된 파일을 OpenCV 이미지(BGR)로 변환
def to_cv2(uploaded_file):
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        uploaded_file.seek(0) 
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        return img
    return None

# 2. Streamlit 웹 UI
st.set_page_config(page_title="Color Transfer App", layout="wide")
st.title("🎨 이미지 색상 변환 프로그램")

st.sidebar.header("📁 이미지 업로드")

uploaded_ref = st.sidebar.file_uploader("1. 기준 이미지 (Style)", type=["jpg", "jpeg", "png"])
reference_img = to_cv2(uploaded_ref)
if reference_img is not None:
    st.sidebar.image(cv2.cvtColor(reference_img, cv2.COLOR_BGR2RGB), caption="기준 스타일", use_container_width=True)

st.sidebar.write("---")

uploaded_change = st.sidebar.file_uploader("2. 대상 이미지 (Target)", type=["jpg", "jpeg", "png"])
target_img = to_cv2(uploaded_change)
if target_img is not None:
    st.sidebar.image(cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB), caption="변환할 원본", use_container_width=True)

    st.sidebar.write("---")
st.sidebar.header("⚙️ 세부 조절")

# 뚝뚝 끊어지는 슬라이더 (0.1 단위)
color_gain = st.sidebar.select_slider(
    "색상 강도 (Color Gain)",
    options=[round(i * 0.1, 1) for i in range(11)],
    value=0.6
)

blend_strength = st.sidebar.slider(
    "원본 대비 합성 비율 (Blending)",
            min_value=0.0, max_value=1.0, value=0.7, step=0.1)

# 3. 메인 로직
if reference_img is not None and target_img is not None:
    st.success(f"✔️ 준비 완료: {uploaded_change.name}")
    
    try:
        # 색상 변환 실행
        result_img = color_transfer(reference_img, target_img)
        result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        
        st.markdown("### 🖼️ 변환 결과")
        col_img, col_info = st.columns([3, 1])
        
        with col_img:
            st.image(result_rgb, caption="결과 이미지", use_container_width=True)
            
        with col_info:
            st.write("✨ **작업 완료!**")
            # 다운로드 버튼
            is_success, buffer = cv2.imencode(".jpg", result_img)
            if is_success:
                st.download_button(
                    label="📥 이미지 저장하기",
                    data=buffer.tobytes(),
                    file_name=f"transferred_{uploaded_change.name}",
                    mime="image/jpeg"
                )
        st.balloons()

    except Exception as e:
        st.error(f"❌ 오류 발생: {e}")
else:
    st.info("👈 왼쪽 사이드바에서 두 장의 이미지를 업로드해 주세요.")