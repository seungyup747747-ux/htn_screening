"""
고혈압 위험 스크리닝 웹앱 (Streamlit)
- 모델: GradientBoostingClassifier (Top12) — KNHANES 9기 (2022~2024) 학습
- 임계값: OOF 민감도 85% 목표 기준 데이터 기반 도출 (0.3072)
- 입력: 비침습 자가보고 변수 12개 (혈압계 없이)
- 출력: 스크리닝 위험점수 + 데이터 기반 임계값 기준 분류
"""
import json
import os
import pickle
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
    """노트북 final_exp.save_model 로 생성한 PyCaret 파이프라인 로드.

    PyCaret 의 save_model 은 내부적으로 joblib.dump 를 사용하므로
    동일하게 joblib.load 로 읽는다. 만일을 대비해 pickle 도 시도한다.
    저장된 객체가 (pipeline, name) 형태의 튜플/리스트인 경우
    predict 가 가능한 요소를 추출한다.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "htn_final_model.pkl")

    obj = None
    try:
        import joblib
        obj = joblib.load(model_path)
    except Exception:
        with open(model_path, "rb") as f:
            obj = pickle.load(f)

    if isinstance(obj, (tuple, list)):
        for elem in obj:
            if hasattr(elem, "predict"):
                obj = elem
                break

    if not hasattr(obj, "predict"):
        raise RuntimeError(
            f"로드된 객체에 predict 메서드가 없습니다. type={type(obj).__name__}"
        )

    return obj


@st.cache_resource
def load_screening_config():
    """노트북에서 저장한 임계값·메타데이터 로드."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "screening_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


try:
    model = load_htn_model()
    model_loaded = True
    model_error = None
except Exception as e:
    model = None
    model_loaded = False
    model_error = str(e)

try:
    screening_config = load_screening_config()
except Exception:
    screening_config = {
        "screening_threshold": 0.3072,
        "target_sensitivity": 0.85,
        "model": "GradientBoostingClassifier",
        "feature_version": "Top12",
        "score_method": "predict_proba",
        "input_variables": [
            "age", "waist", "dyslipidemia_dx", "bmi",
            "fh_htn_mother", "fh_htn_sibling", "fh_htn_father",
            "married", "self_health", "edu",
            "drink_amount", "drink_freq",
        ],
    }

SCREENING_THRESHOLD = float(screening_config.get("screening_threshold", 0.3072))
TARGET_SENS = float(screening_config.get("target_sensitivity", 0.85))


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
    "혈압계 없이 12개 설문 응답만으로 고혈압 위험 수준을 1차 점검합니다."
)
st.info(
    "ℹ️ 본 도구는 **1차 스크리닝**용 학술 연구 목적이며, 의학적 진단을 대체할 수 없습니다. "
    "임계값을 초과한 경우 의료기관에서 정확한 혈압 측정을 받으시기 바랍니다."
)

if not model_loaded:
    st.error(f"❌ 모델 로드 실패: {model_error}")
    st.info("`htn_final_model.pkl` 파일이 같은 폴더에 있는지 확인하세요.")
    st.stop()

st.sidebar.header("📋 설문 응답 (12문항)")
st.sidebar.caption(
    f"모델: {screening_config.get('model', 'GradientBoostingClassifier')} • "
    f"변수: {screening_config.get('feature_version', 'Top12')} • "
    f"임계값: {SCREENING_THRESHOLD:.4f}"
)

input_data = {}


def yes_no_unknown(label, key, help_text=None):
    """0=아니오, 1=예, np.nan=모름"""
    choice = st.selectbox(
        label,
        options=["아니오", "예", "모름"],
        index=0,
        key=key,
        help=help_text,
    )
    return {"아니오": 0, "예": 1, "모름": np.nan}[choice]


with st.sidebar.expander("👤 기본 정보", expanded=True):
    input_data["age"] = st.number_input("나이 (만)", 19, 100, 50)
    input_data["bmi"] = st.number_input(
        "BMI (kg/m²)", 10.0, 50.0, 23.0, step=0.1,
        help="체중(kg) / 키(m)²"
    )
    input_data["waist"] = st.number_input(
        "허리둘레 (cm)", 50.0, 150.0, 85.0, step=0.5,
        help="배꼽 높이에서 줄자로 측정"
    )

with st.sidebar.expander("🍷 음주", expanded=True):
    input_data["drink_freq"] = st.selectbox(
        "최근 1년간 음주 빈도",
        options=[0, 1, 2, 3, 4, 5, 6],
        format_func=lambda x: {
            0: "0. 안 마심",
            1: "1. 월 1회 미만",
            2: "2. 월 1회 정도",
            3: "3. 월 2~4회",
            4: "4. 주 2~3회",
            5: "5. 주 4회 이상",
            6: "6. 매일",
        }[x],
    )
    input_data["drink_amount"] = st.selectbox(
        "한 번에 마시는 음주량 (술잔 기준)",
        options=[0, 1, 2, 3, 4, 5],
        format_func=lambda x: {
            0: "0. 안 마심",
            1: "1. 1~2잔",
            2: "2. 3~4잔",
            3: "3. 5~6잔",
            4: "4. 7~9잔",
            5: "5. 10잔 이상",
        }[x],
    )

with st.sidebar.expander("🏠 사회인구학", expanded=True):
    input_data["married"] = st.selectbox(
        "혼인 상태",
        options=[1, 2],
        format_func=lambda x: "1. 기혼 (배우자 있음)" if x == 1
                              else "2. 미혼 / 이혼 / 사별",
    )
    input_data["edu"] = st.selectbox(
        "최종 학력",
        options=[1, 2, 3, 4],
        format_func=lambda x: {
            1: "1. 초졸 이하",
            2: "2. 중졸",
            3: "3. 고졸",
            4: "4. 대졸 이상",
        }[x],
    )
    input_data["self_health"] = st.selectbox(
        "주관적 건강 상태",
        options=[1, 2, 3, 4, 5],
        format_func=lambda x: {
            1: "1. 매우 좋음",
            2: "2. 좋음",
            3: "3. 보통",
            4: "4. 나쁨",
            5: "5. 매우 나쁨",
        }[x],
    )

with st.sidebar.expander("🩺 기저질환", expanded=True):
    input_data["dyslipidemia_dx"] = yes_no_unknown(
        "이상지질혈증 의사진단 받은 적 있음",
        key="dyslipidemia_dx",
        help_text="콜레스테롤·중성지방 이상 등으로 진단받은 경우",
    )

with st.sidebar.expander("👨‍👩‍👧 가족력 (직계가족 고혈압)", expanded=True):
    input_data["fh_htn_father"] = yes_no_unknown(
        "아버지 — 고혈압 진단 이력",
        key="fh_htn_father",
    )
    input_data["fh_htn_mother"] = yes_no_unknown(
        "어머니 — 고혈압 진단 이력",
        key="fh_htn_mother",
    )
    input_data["fh_htn_sibling"] = yes_no_unknown(
        "형제자매 — 고혈압 진단 이력",
        key="fh_htn_sibling",
    )

predict_btn = st.sidebar.button(
    "🔍 위험도 평가", type="primary", use_container_width=True
)


def predict_htn(model, input_dict, required_cols):
    df = pd.DataFrame([input_dict])
    if required_cols:
        for col in required_cols:
            if col not in df.columns:
                df[col] = np.nan
        df = df[required_cols]
    if TARGET_COL in df.columns:
        df = df.drop(columns=[TARGET_COL])

    risk_score = None
    if hasattr(model, "predict_proba"):
        try:
            proba_arr = model.predict_proba(df)
            cls_list = list(model.classes_) if hasattr(model, "classes_") else None
            if cls_list and 1 in cls_list:
                risk_score = float(proba_arr[0, cls_list.index(1)])
            else:
                risk_score = float(proba_arr[0, -1])
        except Exception:
            risk_score = None

    if risk_score is None and hasattr(model, "decision_function"):
        try:
            from scipy.special import expit
            score = model.decision_function(df)
            s = float(score[0]) if hasattr(score, "__len__") else float(score)
            risk_score = float(expit(s))
        except Exception:
            risk_score = None

    if risk_score is None:
        y_pred = model.predict(df)
        return float(int(y_pred[0])), int(y_pred[0])

    label = int(risk_score >= SCREENING_THRESHOLD)
    return risk_score, label


if predict_btn:
    with st.spinner("평가 중..."):
        try:
            risk_score, label = predict_htn(model, input_data, required_features)
        except Exception as e:
            st.error(f"평가 중 오류: {e}")
            st.exception(e)
            st.stop()

    is_positive = (risk_score >= SCREENING_THRESHOLD)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.metric(
            "🎯 위험 점수",
            f"{risk_score:.4f}",
            help="모델이 산출한 스크리닝 점수 (보정된 확률 아님)",
        )
    with col2:
        st.metric(
            "📏 임계값",
            f"{SCREENING_THRESHOLD:.4f}",
            help=f"OOF 민감도 {TARGET_SENS*100:.0f}% 목표 기준",
        )
    with col3:
        st.metric(
            "📌 판정",
            "🔴 양성 (정밀 검진 권고)" if is_positive else "🟢 음성 (주기 점검)",
        )

    st.divider()

    if is_positive:
        st.error(
            f"⚠ **양성** — 위험 점수 {risk_score:.4f} ≥ 임계값 {SCREENING_THRESHOLD:.4f}\n\n"
            "본 스크리닝 모델은 귀하의 고혈압 위험을 **임계값 이상**으로 평가했습니다. "
            "병원·보건소에서 정확한 혈압 측정을 권장합니다.\n\n"
            "※ 본 도구의 양성 판정 정확도(특이도)는 약 69%이므로, "
            "실제 고혈압이 아닌 경우에도 양성으로 분류될 수 있습니다. "
            "최종 진단은 의료기관의 혈압 측정으로 확정됩니다."
        )
    else:
        st.success(
            f"✓ **음성** — 위험 점수 {risk_score:.4f} < 임계값 {SCREENING_THRESHOLD:.4f}\n\n"
            "본 스크리닝 모델은 귀하의 고혈압 위험을 **임계값 미만**으로 평가했습니다. "
            "건강한 생활습관 유지와 연 1회 정기 검진을 권장합니다.\n\n"
            "※ 본 도구의 음성 판정 정확도(NPV)는 약 89%로, "
            "약 11%의 음성 판정은 실제 고혈압을 놓칠 수 있습니다."
        )

    st.session_state["last_prediction"] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "input": input_data.copy(),
        "risk_score": risk_score,
        "threshold": SCREENING_THRESHOLD,
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
            "예측된 판정이 본인의 체감과 일치하나요?",
            min_value=1, max_value=5, value=3,
        )
    comment = st.text_area(
        "자유 의견 (선택)",
        placeholder="예: '혈압 측정 결과는 정상이었어요'",
    )
    submitted = st.form_submit_button("📤 피드백 제출")
    if submitted:
        if "last_prediction" not in st.session_state:
            st.warning("먼저 위험도 평가를 실행해주세요.")
        else:
            lp = st.session_state["last_prediction"]
            row = {
                "timestamp": lp["timestamp"],
                "risk_score": lp["risk_score"],
                "threshold": lp["threshold"],
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
            df_fb.to_csv(
                csv_path, mode=mode, header=header,
                index=False, encoding="utf-8-sig",
            )
            st.success("✓ 피드백이 저장되었습니다. 감사합니다!")
            st.balloons()

st.divider()
st.caption(
    "**Data**: 국민건강영양조사(KNHANES) 제9기 (질병관리청, 2022~2024)  •  "
    f"**Model**: {screening_config.get('model', 'GradientBoostingClassifier')} "
    f"({screening_config.get('feature_version', 'Top12')})  •  "
    "**Holdout Sensitivity**: 84.8%  •  **Specificity**: 68.9%  •  "
    "**NPV**: 89.3%  •  **ROC-AUC**: 0.856"
)
st.caption(
    "© 2026 백승엽 (응용정보공학)  •  고급데이터분석 기말 프로젝트"
)
