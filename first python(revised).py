import os
import cv2
import numpy as np
import streamlit as st

# 1. 핵심 이미지 처리 함수 (기존 로직 유지)
def color_transfer(source_img, target_img):
    src = source_img
    tgt = target_img

    src_lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype("float32")
    tgt_lab = cv2.cvtColor(tgt, cv2.COLOR_BGR2LAB).astype("float32")

    (src_mean, src_std) = cv2.meanStdDev(src_lab)
    (tgt_mean, tgt_std) = cv2.meanStdDev(tgt_lab)

    src_mean, src_std = src_mean.flatten(), src_std.flatten()
    tgt_mean, tgt_std = tgt_mean.flatten(), tgt_std.flatten()

    l, a, b = cv2.split(tgt_lab)
    color_gain = 0.6 

    l = (l - tgt_mean[0]) * (src_std[0] / tgt_std[0]) + src_mean[0]
    a = (a - tgt_mean[1]) * (src_std[1] / tgt_std[1] * color_gain) + src_mean[1]
    b = (b - tgt_mean[2]) * (src_std[2] / tgt_std[2] * color_gain) + src_mean[2]

    transfer = cv2.merge([l, a, b])
    transfer = np.clip(transfer, 0, 255).astype("uint8")
    result = cv2.cvtColor(transfer, cv2.COLOR_LAB2BGR)
    
    final_result = cv2.addWeighted(result, 0.7, tgt, 0.3, 0)
    return final_result


# 2. Streamlit 웹 UI
st.set_page_config(page_title="Color Transfer App", layout="wide")
st.title("🎨 이미지 색상 변환(Color Transfer) 프로그램")
st.write("기준 이미지의 색감 스타일에 맞춰 대상 이미지의 색상을 변경합니다.")

st.sidebar.header("📁 이미지 업로드")

uploaded_ref = st.sidebar.file_uploader(
    "1. 색감의 기준이 될 이미지를 선택하세요", 
    type=["jpg", "jpeg", "png"], 
    key="ref_loader"
)


uploaded_change = st.sidebar.file_uploader(
    "2. 변환할 대상 이미지(Change)를 선택하세요", 
    type=["jpg", "jpeg", "png"], 
    key="change_loader"
)

# --- 메인 화면: 이미지 처리 및 결과 출력 ---

if uploaded_ref and uploaded_change:
    
    def to_cv2(uploaded_file):
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # 이미지 로드
    reference_img = to_cv2(uploaded_ref)
    target_img = to_cv2(uploaded_change) 
    
    st.success(f"✔️ 기준 사진 선정 완료: {uploaded_ref.name}")
    st.info("🚀 변환 작업을 시작합니다...")
    
    st.markdown("### 🖼️ 변환 결과 및 다운로드")
    
    try:
        # 색상 변환 실행
        result_img = color_transfer(reference_img, target_img)
        
        # 디자인용 구분선
        st.markdown("---")
        
        # 화면 출력을 위해 RGB 변환
        result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        
        # 레이아웃 설정
        col_img, col_btn = st.columns([2, 1])
        
        with col_img:
            
            st.image(result_rgb, caption=f"결과물 (원본: {uploaded_change.name})", width=600)
            
        with col_btn:
            st.write(f"**변환 성공!**")
            
            # 다운로드 기능
            is_success, buffer = cv2.imencode(".jpg", result_img)
            if is_success:
                st.download_button(
                    label="📥 내 컴퓨터에 저장하기",
                    data=buffer.tobytes(),
                    file_name="result_image.jpg", 
                    mime="image/jpeg"
                )

        st.balloons()
        st.success("🎉 이미지 변환이 완료되었습니다!")

    except Exception as e:
        st.error(f"❌ 변환 실패 - 에러 내용: {e}")

else:
    st.warning("👈 왼쪽 사이드바에서 이미지를 업로드해 주세요.")