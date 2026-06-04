"""
고혈압 위험 스크리닝 웹앱 (Streamlit)
- 모델: RidgeClassifier (Reduced) — KNHANES 9기 학습
- 입력: 비침습 설문 변수 (혈압계 없이)
- 출력: HTN 예측 확률 + 위험 등급 + 사용자 피드백 수집
"""
import os
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

TARGET_COL = "hypertension"


st.set_page_config(
    page_title="고혈압 위험 스크리닝",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_htn_model():
    """joblib으로 직접 로드 (PyCaret 의존성 제거)"""
    import joblib
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "htn_final_model.pkl")
    return joblib.load(model_path)


try:
    model = load_htn_model()
    model_loaded = True
    model_error = None
except Exception as e:
    model = None
    model_loaded = False
    model_error = str(e)


def detect_required_features(m):
    if m is None:
        return None
    cols = None
    try:
        cols = list(m.feature_names_in_)
    except AttributeError:
        try:
            first_step_name = next(iter(m.named_steps))
            cols = list(m.named_steps[first_step_name].feature_names_in_)
        except Exception:
            return None
    if cols and TARGET_COL in cols:
        cols = [c for c in cols if c != TARGET_COL]
    return cols


required_features = detect_required_features(model)


st.title("🩺 고혈압 위험 스크리닝 도구")
st.markdown(
    "**KNHANES 9기 (2022~2024) 데이터 기반 머신러닝 모델** — "
    "혈압계 없이 설문 응답만으로 본인의 고혈압 위험 수준을 1차 점검합니다."
)
st.info(
    "ℹ️ 본 도구는 **1차 스크리닝**용 학술 연구 목적이며, 의학적 진단을 대체할 수 없습니다. "
    "정확한 진단은 반드시 의료기관에서 받으시기 바랍니다."
)

if not model_loaded:
    st.error(f"❌ 모델 로드 실패: {model_error}")
    st.info("`htn_final_model.pkl` 파일이 같은 폴더에 있는지 확인하세요.")
    st.stop()


st.sidebar.header("📋 설문 응답")
st.sidebar.caption(
    f"모델 요구 변수: {len(required_features) if required_features else '자동 감지'}개"
)

input_data = {}

with st.sidebar.expander("👤 기본 정보", expanded=True):
    input_data["age"] = st.number_input("나이 (만)", 19, 100, 50)
    input_data["sex"] = st.selectbox(
        "성별",
        options=[1, 2],
        format_func=lambda x: "남자" if x == 1 else "여자",
    )
    input_data["bmi"] = st.number_input(
        "BMI (kg/m²)", 10.0, 50.0, 23.0, step=0.1, help="체중(kg) / 키(m)²"
    )
    input_data["waist"] = st.number_input(
        "허리둘레 (cm)", 50.0, 150.0, 85.0, step=0.5
    )

with st.sidebar.expander("🚬 생활습관"):
    input_data["smoking_status"] = st.selectbox(
        "흡연 상태",
        options=[0, 1, 2],
        format_func=lambda x: {0: "비흡연", 1: "과거 흡연", 2: "현재 흡연"}[x],
    )
    input_data["drink_freq"] = st.slider("음주 빈도 (0=안 마심, 1~7 등급)", 0, 7, 1)
    input_data["drink_amount"] = st.slider("음주량 (0=안 마심, 1~7 등급)", 0, 7, 1)
    input_data["walk_min_week"] = st.number_input(
        "주간 총 걷기 시간 (분)", 0, 5000, 150, step=10
    )
    input_data["sedentary_min_day"] = st.number_input(
        "일일 좌식 시간 (분)", 0, 1440, 480, step=30
    )
    input_data["sleep_hours"] = st.number_input(
        "평균 수면 시간 (시간)", 0.0, 16.0, 7.0, step=0.5
    )

with st.sidebar.expander("🧠 정신건강"):
    input_data["stress"] = st.selectbox(
        "스트레스 수준",
        options=[1, 2, 3, 4],
        format_func=lambda x: {1: "거의 없음", 2: "조금", 3: "많이", 4: "대단히 많음"}[x],
    )
    input_data["phq9"] = st.number_input("PHQ-9 우울 점수 (0~27)", 0, 27, 5)

with st.sidebar.expander("🩹 주관적 건강 (EQ-5D)"):
    input_data["self_health"] = st.selectbox(
        "주관적 건강 상태",
        options=[1, 2, 3, 4, 5],
        format_func=lambda x: {1: "매우 좋음", 2: "좋음", 3: "보통", 4: "나쁨", 5: "매우 나쁨"}[x],
    )
    eq5d_dims = ["운동 능력", "자기 관리", "일상 활동", "통증/불편", "불안/우울"]
    for i, lbl in enumerate(eq5d_dims, 1):
        input_data[f"eq5d_{i}"] = st.selectbox(
            lbl,
            options=[1, 2, 3],
            format_func=lambda x: {1: "문제 없음", 2: "다소 문제", 3: "심각한 문제"}[x],
            key=f"eq5d_{i}",
        )

with st.sidebar.expander("🏠 사회인구학"):
    input_data["income"] = st.selectbox(
        "소득 5분위",
        options=[1, 2, 3, 4, 5],
        format_func=lambda x: f"{x}분위  (1=하위 ~ 5=상위)",
    )
    input_data["edu"] = st.selectbox(
        "최종 학력",
        options=[1, 2, 3, 4],
        format_func=lambda x: {1: "초졸 이하", 2: "중졸", 3: "고졸", 4: "대졸 이상"}[x],
    )
    input_data["occupation"] = st.selectbox(
        "직업군",
        options=[1, 2, 3, 4, 5, 6, 7],
        format_func=lambda x: f"{x}번 직업군",
    )
    input_data["married"] = st.selectbox(
        "혼인 상태",
        options=[1, 2],
        format_func=lambda x: "기혼 (배우자 있음)" if x == 1 else "미혼/이혼/사별",
    )
    input_data["household_size"] = st.number_input("가구원 수", 1, 10, 2)
    input_data["urban"] = st.selectbox(
        "거주지",
        options=[1, 2],
        format_func=lambda x: "도시(동)" if x == 1 else "농촌(읍·면)",
    )

predict_btn = st.sidebar.button("🔍 위험도 예측", type="primary", use_container_width=True)


def predict_htn(model, input_dict, required_cols):
    df = pd.DataFrame([input_dict])

    if required_cols:
        for col in required_cols:
            if col not in df.columns:
                df[col] = np.nan
        df = df[required_cols]

    if TARGET_COL in df.columns:
        df = df.drop(columns=[TARGET_COL])

    y_pred = model.predict(df)
    label = int(y_pred[0])

    proba = None
    if hasattr(model, "predict_proba"):
        try:
            proba_arr = model.predict_proba(df)
            cls_list = list(model.classes_) if hasattr(model, "classes_") else None
            if cls_list and 1 in cls_list:
                proba = float(proba_arr[0, cls_list.index(1)])
            else:
                proba = float(proba_arr[0, -1])
        except Exception:
            proba = None

    if proba is None and hasattr(model, "decision_function"):
        try:
            from scipy.special import expit
            score = model.decision_function(df)
            s = float(score[0]) if hasattr(score, "__len__") else float(score)
            proba = float(expit(s))
        except Exception:
            proba = None

    if proba is None:
        proba = float(label)

    return proba, label


if predict_btn:
    with st.spinner("예측 중..."):
        try:
            proba, label = predict_htn(model, input_data, required_features)
        except Exception as e:
            st.error(f"예측 중 오류: {e}")
            st.exception(e)
            st.stop()

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.metric("🎯 예측 확률 (HTN)", f"{proba*100:.1f}%" if proba is not None else "N/A")
    with col2:
        if proba is None:
            risk = "—"
            risk_color = "⚪"
        elif proba < 0.3:
            risk = "낮음"
            risk_color = "🟢"
        elif proba < 0.6:
            risk = "중간"
            risk_color = "🟡"
        else:
            risk = "높음"
            risk_color = "🔴"
        st.metric("📊 위험 등급", f"{risk_color} {risk}")
    with col3:
        st.metric("📌 권고", "정밀 검진" if (proba or 0) >= 0.5 else "주기 점검")

    st.divider()
    if proba is None:
        st.warning("예측 확률을 가져올 수 없습니다.")
    elif proba >= 0.5:
        st.error(
            f"⚠ 본 모델은 귀하의 고혈압 위험을 **높음 ({proba*100:.1f}%)** 으로 평가했습니다. "
            "병원·보건소에서 정확한 혈압 측정을 받으시기를 권합니다."
        )
    elif proba >= 0.3:
        st.warning(
            f"⚠ 본 모델은 귀하의 고혈압 위험을 **중간 ({proba*100:.1f}%)** 으로 평가했습니다. "
            "정기적인 자가 혈압 측정과 생활습관 관리를 권합니다."
        )
    else:
        st.success(
            f"✓ 본 모델은 귀하의 고혈압 위험을 **낮음 ({proba*100:.1f}%)** 으로 평가했습니다. "
            "건강한 생활습관을 유지하시고 연 1회 정기 검진을 받으시기 바랍니다."
        )

    st.session_state["last_prediction"] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "input": input_data.copy(),
        "probability": proba,
        "label": label,
    }


st.divider()
st.subheader("📝 사용자 피드백")
st.caption("예측 결과에 대한 의견은 향후 모델 개선에 활용됩니다.")

with st.form("feedback_form", clear_on_submit=True):
    col1, col2 = st.columns(2)
    with col1:
        actual_dx = st.radio(
            "본인의 실제 고혈압 진단 결과와 일치하나요?",
            options=["예 — 일치", "아니오 — 불일치", "모름"],
        )
    with col2:
        agree = st.slider(
            "예측된 위험 등급이 본인의 체감과 일치하나요?",
            min_value=1, max_value=5, value=3,
        )

    comment = st.text_area(
        "자유 의견 (선택)",
        placeholder="예: '혈압 측정 결과는 정상이었어요'",
    )

    submitted = st.form_submit_button("📤 피드백 제출")

    if submitted:
        if "last_prediction" not in st.session_state:
            st.warning("먼저 위험도 예측을 실행해주세요.")
        else:
            lp = st.session_state["last_prediction"]
            row = {
                "timestamp": lp["timestamp"],
                "probability": lp["probability"],
                "predicted_label": lp["label"],
                "actual_match": actual_dx,
                "perception_agreement": agree,
                "comment": comment,
            }
            for k, v in lp["input"].items():
                row[f"input_{k}"] = v

            script_dir = os.path.dirname(os.path.abspath(__file__))
            csv_path = os.path.join(script_dir, "user_feedback.csv")
            df_fb = pd.DataFrame([row])
            mode = "a" if os.path.exists(csv_path) else "w"
            header = not os.path.exists(csv_path)
            df_fb.to_csv(csv_path, mode=mode, header=header, index=False, encoding="utf-8-sig")

            st.success("✓ 피드백이 저장되었습니다. 감사합니다!")
            st.balloons()


st.divider()
st.caption(
    "**Data**: 국민건강영양조사(KNHANES) 제9기 (질병관리청, 2022~2024)  •  "
    "**Model**: RidgeClassifier (Reduced)  •  "
    "**Sensitivity**: 79.1%  •  **NPV**: 86.5%"
)
st.caption(
    "© 2026 백승엽 (응용정보공학)  •  고급데이터분석 기말 프로젝트"
)
