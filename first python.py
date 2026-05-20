

import os
import cv2
import numpy as np
import streamlit as st
import io

# =================================================================
# 1. 핵심 이미지 처리 함수 (기존 코드에서 입력 방식만 수정)
# =================================================================
# 💡 수정 포인트: 경로(_path) 대신 이미지 데이터(_img)를 바로 받습니다.
def color_transfer(source_img, target_img):
    # [수정 완료] 기존의 cv2.imread(source_path) 단계를 생략하고 변수를 그대로 사용합니다.
    src = source_img
    tgt = target_img

    # 2. BGR을 LAB 색 공간으로 변환
    src_lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype("float32")
    tgt_lab = cv2.cvtColor(tgt, cv2.COLOR_BGR2LAB).astype("float32")

    # 3. 各 채널별 평균(mean)과 표준편차(std) 계산
    (src_mean, src_std) = cv2.meanStdDev(src_lab)
    (tgt_mean, tgt_std) = cv2.meanStdDev(tgt_lab)

    # 계산을 위해 차원을 평평하게(1차원) 만듭니다.
    src_mean, src_std = src_mean.flatten(), src_std.flatten()
    tgt_mean, tgt_std = tgt_mean.flatten(), tgt_std.flatten()

    # 4. 범용성을 위한 안전장치 추가
    l, a, b = cv2.split(tgt_lab)
    
    color_gain = 0.6 

    # L(밝기)은 대비가 너무 깨지지 않게 비율을 살짝 조정
    l = (l - tgt_mean[0]) * (src_std[0] / tgt_std[0]) + src_mean[0]
    
    # a, b(색상) 채널에 color_gain을 곱해서 시뻘게지는 현상을 차단
    a = (a - tgt_mean[1]) * (src_std[1] / tgt_std[1] * color_gain) + src_mean[1]
    b = (b - tgt_mean[2]) * (src_std[2] / tgt_std[2] * color_gain) + src_mean[2]

    # 5. 결과 합치기 및 제한
    transfer = cv2.merge([l, a, b])
    transfer = np.clip(transfer, 0, 255).astype("uint8")
    result = cv2.cvtColor(transfer, cv2.COLOR_LAB2BGR)
    
    # 6. 원본과 자연스럽게 합성 (Blending)
    final_result = cv2.addWeighted(result, 0.7, tgt, 0.3, 0)
    
    return final_result


# =================================================================
# 2. Streamlit 웹 화면 UI 및 실행부 (기존 로컬 경로설정/반복문 대체)
# =================================================================
st.set_page_config(page_title="Color Transfer App", layout="wide")
st.title("🎨 이미지 색상 변환(Color Transfer) 프로그램")
st.write("기준 이미지의 색감 스타일에 맞춰 대상 이미지들의 색상을 자동으로 변경합니다.")

# --- 사이드바: 파일 업로드 공간 ---
st.sidebar.header("📁 이미지 업로드")

# 기준 이미지 1장 업로드 (기존 ref_folder 대체)
uploaded_ref = st.sidebar.file_uploader(
    "1. 색감의 기준이 될 이미지를 선택하세요(한장만 가능합니다)", 
    type=["jpg", "jpeg", "png"], 
    key="ref_loader"
)

# 변경할 이미지들 여러 장 업로드 (기존 change_folder 대체)
uploaded_changes = st.sidebar.file_uploader(
    "2. 변환할 대상 이미지들을 모두 선택하세요", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=True,
    key="change_loader"
)


# --- 메인 화면: 이미지 처리 및 결과 출력 ---
if uploaded_ref and uploaded_changes:
    
    # 웹 업로드 파일(Bytes)을 OpenCV 이미지(Numpy array)로 바꾸는 헬퍼 함수
    def to_cv2(uploaded_file):
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # 기준 이미지 로드
    reference_img = to_cv2(uploaded_ref)
    st.success(f"✔️ 기준 사진 선정 완료: {uploaded_ref.name}")
    
    total_files = len(uploaded_changes)
    st.info(f"🚀 총 {total_files}장의 변환 작업을 시작합니다...")
    
    # 레이아웃 격자 설정 (한 줄에 결과물이 깔끔하게 보이도록 정렬)
    st.markdown("### 🖼️ 변환 결과 및 다운로드")
    
    # 기존 코드의 반복문 구간 구현
    for i, change_file in enumerate(uploaded_changes, 1):
        target_img = to_cv2(change_file)
        
        try:
            # 색상 변환 함수 실행 (이제 경로 대신 이미지 변수를 직접 넘깁니다)
            result_img = color_transfer(reference_img, target_img)
            
            # 파일명 자동 생성 규칙
            save_name = f"result_image_{i}.jpg"
            
            # 디자인용 구분선
            st.markdown("---")
            
            # 웹 출력을 위해 OpenCV의 BGR 서식을 RGB 서식으로 변환
            result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            
            # 좌우 화면 분할 (왼쪽: 결과 이미지, 오른쪽: 다운로드 버튼 및 정보)
            col_img, col_btn = st.columns([2, 1])
            
            with col_img:
                st.image(result_rgb, caption=f"결과물: {save_name} (원본: {change_file.name})", use_column_width=False, width=450)
                
            with col_btn:
                st.write(f"**[{i} / {total_files}] 변환 성공**")
                st.text(f"파일명: {save_name}")
                
                # --- 기존 cv2.imwrite를 대체하는 다운로드 기능 ---
                is_success, buffer = cv2.imencode(".jpg", result_img)
                if is_success:
                    st.download_button(
                        label=f"📥 내 컴퓨터에 저장하기",
                        data=buffer.tobytes(),
                        file_name=save_name,
                        mime="image/jpeg",
                        key=f"btn_{i}" # 고유 키값 지정 필수
                    )
                    
        except Exception as e:
            st.error(f"❌ 실패 [{i}/{total_files}]: {change_file.name} - 에러 내용: {e}")

    # 모든 작업 완료 시 효과
    st.balloons()
    st.success("🎉 모든 이미지의 변환 및 다운로드 준비가 완료되었습니다!")

else:
    st.warning("👈 왼쪽 사이드바에서 기준 이미지와 변환할 이미지들을 먼저 업로드해 주세요.")

