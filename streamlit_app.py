"""
고혈압 위험 스크리닝 웹앱 (Streamlit) — v3 (메인+보조 분석)
- 메인 분석: 2022-2024년 KNHANES 9기 전체 (17,017명), PHQ-9/EQ-5D 제외
- 보조 분석: 2022·2024년 (11,193명), PHQ-9/EQ-5D 포함 — 변수 가치 검토용
- 최종 채택: 메인 분석 (3개년 전체 표본 활용)
- 입력 UI: screening_config.json의 input_variables를 읽어 동적 생성
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


# =========================================================================
# 모든 KNHANES 9기 변수 입력 UI 사전 (Feature Registry)
# screening_config.json의 input_variables에 포함된 변수만 화면에 표시됨.
# =========================================================================
def yes_no_unknown_widget(label, key, help_text=None):
    choice = st.selectbox(
        label, options=["아니오", "예", "모름"],
        index=0, key=key, help=help_text,
    )
    return {"아니오": 0, "예": 1, "모름": np.nan}[choice]


FEATURE_REGISTRY = {
    # ── 인구·신체 ───────────────────────────────────────
    "age": {
        "group": "demo", "label": "나이 (만)", "type": "number",
        "min": 19, "max": 100, "step": 1, "default": 50,
    },
    "sex": {
        "group": "demo", "label": "성별", "type": "select",
        "options": {"1. 남성": 1, "2. 여성": 2}, "default": "1. 남성",
    },
    "bmi": {
        "group": "demo", "label": "BMI (kg/m²)", "type": "number",
        "min": 10.0, "max": 50.0, "step": 0.1, "default": 23.0,
        "help": "체중(kg) / 키(m)²",
    },
    "waist": {
        "group": "demo", "label": "허리둘레 (cm)", "type": "number",
        "min": 50.0, "max": 150.0, "step": 0.5, "default": 85.0,
        "help": "배꼽 높이에서 줄자로 측정",
    },
    "household_size": {
        "group": "demo", "label": "가구원 수", "type": "number",
        "min": 1, "max": 10, "step": 1, "default": 3,
    },

    # ── 사회경제 ───────────────────────────────────────
    "edu": {
        "group": "socio", "label": "최종 학력", "type": "select",
        "options": {"1. 초졸 이하": 1, "2. 중졸": 2, "3. 고졸": 3, "4. 대졸 이상": 4},
        "default": "3. 고졸",
    },
    "income": {
        "group": "socio", "label": "가구 소득 5분위", "type": "select",
        "options": {"1. 하 (1분위)": 1, "2. 중하 (2분위)": 2, "3. 중 (3분위)": 3,
                    "4. 중상 (4분위)": 4, "5. 상 (5분위)": 5},
        "default": "3. 중 (3분위)",
    },
    "occupation": {
        "group": "socio", "label": "직업군", "type": "select",
        "options": {"1. 관리자/전문가": 1, "2. 사무종사자": 2, "3. 서비스/판매": 3,
                    "4. 농림어업": 4, "5. 기능/장치/조립": 5, "6. 단순노무": 6,
                    "7. 무직/주부/학생": 7},
        "default": "2. 사무종사자",
    },
    "married": {
        "group": "socio", "label": "혼인 상태", "type": "select",
        "options": {"1. 기혼 (배우자 있음)": 1, "2. 미혼/이혼/사별": 2},
        "default": "1. 기혼 (배우자 있음)",
    },
    "urban": {
        "group": "socio", "label": "거주 지역", "type": "select",
        "options": {"1. 동 (도시)": 1, "2. 읍/면 (농촌)": 2},
        "default": "1. 동 (도시)",
    },

    # ── 음주·흡연 ───────────────────────────────────────
    "smoking_status": {
        "group": "lifestyle", "label": "흡연 상태", "type": "select",
        "options": {"0. 비흡연": 0, "1. 과거 흡연": 1, "2. 현재 흡연": 2},
        "default": "0. 비흡연",
    },
    "drink_freq": {
        "group": "lifestyle", "label": "최근 1년간 음주 빈도", "type": "select",
        "options": {"0. 안 마심": 0, "1. 월 1회 미만": 1, "2. 월 1회 정도": 2,
                    "3. 월 2~4회": 3, "4. 주 2~3회": 4, "5. 주 4회 이상": 5,
                    "6. 매일": 6},
        "default": "1. 월 1회 미만",
    },
    "drink_amount": {
        "group": "lifestyle", "label": "한 번에 마시는 음주량 (잔)", "type": "select",
        "options": {"0. 안 마심": 0, "1. 1~2잔": 1, "2. 3~4잔": 2,
                    "3. 5~6잔": 3, "4. 7~9잔": 4, "5. 10잔 이상": 5},
        "default": "0. 안 마심",
    },

    # ── 신체활동·수면·스트레스 ─────────────────────────
    "walk_min_week": {
        "group": "lifestyle", "label": "주간 걷기 시간 (분/주)", "type": "number",
        "min": 0.0, "max": 10000.0, "step": 10.0, "default": 150.0,
        "help": "1주일 동안 누적 걷기 시간",
    },
    "sedentary_min_day": {
        "group": "lifestyle", "label": "하루 앉아있는 시간 (분/일)", "type": "number",
        "min": 0.0, "max": 1440.0, "step": 30.0, "default": 480.0,
    },
    "sleep_hours": {
        "group": "lifestyle", "label": "평균 수면시간 (시간/일)", "type": "number",
        "min": 2.0, "max": 16.0, "step": 0.5, "default": 7.0,
    },
    "stress": {
        "group": "lifestyle", "label": "스트레스 인지", "type": "select",
        "options": {"1. 대단히 많이 느낌": 1, "2. 많이 느낌": 2,
                    "3. 조금 느낌": 3, "4. 거의 안 느낌": 4},
        "default": "3. 조금 느낌",
    },

    # ── 건강 인식·체형 ─────────────────────────────────
    "self_health": {
        "group": "perception", "label": "주관적 건강 상태", "type": "select",
        "options": {"1. 매우 좋음": 1, "2. 좋음": 2, "3. 보통": 3,
                    "4. 나쁨": 4, "5. 매우 나쁨": 5},
        "default": "3. 보통",
    },
    "body_shape": {
        "group": "perception", "label": "본인이 인지하는 체형", "type": "select",
        "options": {"1. 매우 마름": 1, "2. 약간 마름": 2, "3. 보통": 3,
                    "4. 약간 비만": 4, "5. 매우 비만": 5},
        "default": "3. 보통",
    },
    "weight_change": {
        "group": "perception", "label": "최근 1년 체중 변화", "type": "select",
        "options": {"1. 변화 없음": 1, "2. 3kg 이상 감소": 2, "3. 3kg 이상 증가": 3},
        "default": "1. 변화 없음",
    },

    # ── 식이 (영양소) ───────────────────────────────────
    "energy_kcal": {
        "group": "nutrition", "label": "1일 에너지 섭취량 (kcal)", "type": "number",
        "min": 0.0, "max": 6000.0, "step": 10.0, "default": 2000.0,
    },
    "food_intake_g": {
        "group": "nutrition", "label": "1일 음식 섭취량 (g)", "type": "number",
        "min": 0.0, "max": 10000.0, "step": 50.0, "default": 1500.0,
    },
    "water_g": {
        "group": "nutrition", "label": "1일 수분 섭취량 (g)", "type": "number",
        "min": 0.0, "max": 10000.0, "step": 50.0, "default": 1000.0,
    },
    "carbohydrate_g": {
        "group": "nutrition", "label": "1일 탄수화물 섭취량 (g)", "type": "number",
        "min": 0.0, "max": 1000.0, "step": 5.0, "default": 280.0,
    },
    "protein_g": {
        "group": "nutrition", "label": "1일 단백질 섭취량 (g)", "type": "number",
        "min": 0.0, "max": 500.0, "step": 1.0, "default": 70.0,
    },
    "fat_g": {
        "group": "nutrition", "label": "1일 지방 섭취량 (g)", "type": "number",
        "min": 0.0, "max": 500.0, "step": 1.0, "default": 50.0,
    },
    "saturated_fat_g": {
        "group": "nutrition", "label": "1일 포화지방 섭취량 (g)", "type": "number",
        "min": 0.0, "max": 200.0, "step": 0.5, "default": 15.0,
    },
    "sugar_g": {
        "group": "nutrition", "label": "1일 당류 섭취량 (g)", "type": "number",
        "min": 0.0, "max": 500.0, "step": 1.0, "default": 60.0,
    },
    "fiber_g": {
        "group": "nutrition", "label": "1일 식이섬유 섭취량 (g)", "type": "number",
        "min": 0.0, "max": 100.0, "step": 0.5, "default": 20.0,
    },
    "sodium_mg": {
        "group": "nutrition", "label": "1일 나트륨 섭취량 (mg)", "type": "number",
        "min": 0.0, "max": 20000.0, "step": 100.0, "default": 3500.0,
    },
    "potassium_mg": {
        "group": "nutrition", "label": "1일 칼륨 섭취량 (mg)", "type": "number",
        "min": 0.0, "max": 20000.0, "step": 100.0, "default": 2500.0,
    },
    "diet_therapy": {
        "group": "nutrition", "label": "식사요법 여부", "type": "select",
        "options": {"1. 함": 1, "2. 안 함": 2}, "default": "2. 안 함",
    },
    "usual_meal_amount": {
        "group": "nutrition", "label": "평소 식사량", "type": "select",
        "options": {"1. 매우 많이": 1, "2. 약간 많이": 2, "3. 보통": 3,
                    "4. 약간 적게": 4, "5. 매우 적게": 5},
        "default": "3. 보통",
    },

    # ── 기저질환 (의사진단) ────────────────────────────
    "dyslipidemia_dx": {
        "group": "disease", "label": "이상지질혈증 의사진단", "type": "yes_no",
        "help": "콜레스테롤·중성지방 이상으로 진단받은 경우",
    },
    "diabetes_dx": {
        "group": "disease", "label": "당뇨병 의사진단", "type": "yes_no",
    },
    "asthma_dx": {
        "group": "disease", "label": "천식 의사진단", "type": "yes_no",
    },
    "atopic_dermatitis_dx": {
        "group": "disease", "label": "아토피 피부염 의사진단", "type": "yes_no",
    },
    "allergic_rhinitis_dx": {
        "group": "disease", "label": "알레르기 비염 의사진단", "type": "yes_no",
    },
    "kidney_disease_dx": {
        "group": "disease", "label": "신장 질환 의사진단", "type": "yes_no",
    },

    # ── 가족력: 고혈압 ─────────────────────────────────
    "fh_htn_father": {
        "group": "fh_htn", "label": "아버지 — 고혈압 진단 이력", "type": "yes_no",
    },
    "fh_htn_mother": {
        "group": "fh_htn", "label": "어머니 — 고혈압 진단 이력", "type": "yes_no",
    },
    "fh_htn_sibling": {
        "group": "fh_htn", "label": "형제자매 — 고혈압 진단 이력", "type": "yes_no",
    },

    # ── 가족력: 당뇨 ───────────────────────────────────
    "fh_dm_father": {
        "group": "fh_dm", "label": "아버지 — 당뇨 진단 이력", "type": "yes_no",
    },
    "fh_dm_mother": {
        "group": "fh_dm", "label": "어머니 — 당뇨 진단 이력", "type": "yes_no",
    },
    "fh_dm_sibling": {
        "group": "fh_dm", "label": "형제자매 — 당뇨 진단 이력", "type": "yes_no",
    },

    # ── 가족력: 이상지질혈증 ───────────────────────────
    "fh_dyslip_father": {
        "group": "fh_dyslip", "label": "아버지 — 이상지질혈증 진단 이력", "type": "yes_no",
    },
    "fh_dyslip_mother": {
        "group": "fh_dyslip", "label": "어머니 — 이상지질혈증 진단 이력", "type": "yes_no",
    },
    "fh_dyslip_sibling": {
        "group": "fh_dyslip", "label": "형제자매 — 이상지질혈증 진단 이력", "type": "yes_no",
    },

    # ── 정신건강·삶의 질 (보조 분석용; 메인 분석에는 없을 가능성 큼) ─
    "phq9_depression_risk": {
        "group": "mental", "label": "PHQ-9 우울 위험 (≥10점)", "type": "yes_no",
        "help": "PHQ-9 총점이 10점 이상이면 '예'",
    },
    "eq5d_any_problem": {
        "group": "mental", "label": "EQ-5D 어느 한 영역이라도 문제 있음", "type": "yes_no",
        "help": "EQ-5D 5개 영역 중 어느 하나라도 '약간 문제 있음' 이상",
    },
}

GROUP_LABELS = {
    "demo":        ("👤 기본 정보", True),
    "socio":       ("💰 사회경제", True),
    "lifestyle":   ("🍷 생활습관", True),
    "perception":  ("🪞 건강 인식·체형", True),
    "nutrition":   ("🍱 식이·영양", False),
    "disease":     ("🩺 기저질환", True),
    "fh_htn":      ("👨‍👩‍👧 가족력 — 고혈압", True),
    "fh_dm":       ("👨‍👩‍👧 가족력 — 당뇨", False),
    "fh_dyslip":   ("👨‍👩‍👧 가족력 — 이상지질혈증", False),
    "mental":      ("🧠 정신건강·삶의 질", True),
}


# =========================================================================
# 모델·설정 로딩
# =========================================================================
@st.cache_resource
def load_htn_model():
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
    # v3 메인 분석 기준 fallback
    screening_config = {
        "screening_threshold": 0.40,
        "target_sensitivity": 0.85,
        "model": "LogisticRegression",
        "strategy": "SMOTE",
        "feature_version": "Top?",
        "score_method": "predict_proba",
        "input_variables": list(FEATURE_REGISTRY.keys())[:12],
    }

SCREENING_THRESHOLD = float(screening_config.get("screening_threshold", 0.40))
TARGET_SENS = float(screening_config.get("target_sensitivity", 0.85))
INPUT_VARS = screening_config.get("input_variables", [])


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
# 우선순위: 모델에서 직접 추출한 변수 > screening_config의 input_variables
ACTIVE_FEATURES = required_features if required_features else INPUT_VARS


# =========================================================================
# 헤더
# =========================================================================
st.title("🩺 고혈압 위험 스크리닝 도구")
st.markdown(
    "**KNHANES 9기 (2022, 2023, 2024) 데이터 기반 머신러닝 모델 v3** — "
    f"혈압계 없이 {len(ACTIVE_FEATURES)}개 설문 응답만으로 고혈압 위험 수준을 1차 점검합니다."
)
st.info(
    "ℹ️ 본 도구는 **1차 스크리닝**용 학술 연구 목적이며, 의학적 진단을 대체할 수 없습니다. "
    "임계값을 초과한 경우 의료기관에서 정확한 혈압 측정을 받으시기 바랍니다."
)
st.caption(
    "📅 **학습 데이터**: KNHANES 제9기 전체 (2022, 2023, 2024년 차수), "
    "만 19세 이상 총 약 17,017명. "
    "PHQ-9·EQ-5D 변수의 추가 활용 가능성은 보조 분석 (2022·2024, 약 11,193명) "
    "으로 별도 검토되었으며, 성능 이득이 제한적이어서 "
    "본 모델은 3개년 전체 표본을 사용한 주 분석 결과를 채택하였습니다."
)

if not model_loaded:
    st.error(f"❌ 모델 로드 실패: {model_error}")
    st.info("`htn_final_model.pkl` 파일이 같은 폴더에 있는지 확인하세요.")
    st.stop()

st.sidebar.header(f"📋 설문 응답 ({len(ACTIVE_FEATURES)}문항)")
st.sidebar.caption(
    f"전략: {screening_config.get('strategy', '-')} • "
    f"모델: {screening_config.get('model', '-')} • "
    f"변수: {screening_config.get('feature_version', '-')} • "
    f"임계값: {SCREENING_THRESHOLD:.4f}"
)


# =========================================================================
# 동적 입력 폼 생성 — ACTIVE_FEATURES에 포함된 변수만 표시
# =========================================================================
def render_widget(feature_name: str):
    cfg = FEATURE_REGISTRY.get(feature_name)
    if cfg is None:
        # Registry에 없는 변수는 number_input으로 대체 (안전장치)
        return st.number_input(
            f"{feature_name} (자동 생성)",
            value=0.0, step=0.1, key=f"input_{feature_name}",
        )

    label = cfg["label"]
    help_text = cfg.get("help", None)

    if cfg["type"] == "number":
        return st.number_input(
            label,
            min_value=cfg["min"], max_value=cfg["max"],
            value=float(cfg["default"]) if isinstance(cfg["min"], float) else int(cfg["default"]),
            step=cfg["step"], help=help_text,
            key=f"input_{feature_name}",
        )
    elif cfg["type"] == "select":
        options = cfg["options"]
        labels = list(options.keys())
        default_label = cfg["default"] if cfg["default"] in labels else labels[0]
        choice = st.selectbox(
            label, labels, index=labels.index(default_label),
            help=help_text, key=f"input_{feature_name}",
        )
        return options[choice]
    elif cfg["type"] == "yes_no":
        return yes_no_unknown_widget(label, key=feature_name, help_text=help_text)
    else:
        return st.number_input(label, value=0.0, key=f"input_{feature_name}")


input_data = {}
features_by_group = {}
for feat in ACTIVE_FEATURES:
    grp = FEATURE_REGISTRY.get(feat, {}).get("group", "other")
    features_by_group.setdefault(grp, []).append(feat)

for grp_key, (grp_label, expanded_default) in GROUP_LABELS.items():
    grp_feats = features_by_group.get(grp_key, [])
    if not grp_feats:
        continue
    with st.sidebar.expander(grp_label, expanded=expanded_default):
        for feat in grp_feats:
            input_data[feat] = render_widget(feat)

# Registry에 group이 없는 변수 처리
other_feats = features_by_group.get("other", [])
if other_feats:
    with st.sidebar.expander("📦 기타 변수", expanded=True):
        for feat in other_feats:
            input_data[feat] = render_widget(feat)

predict_btn = st.sidebar.button(
    "🔍 위험도 평가", type="primary", use_container_width=True
)


# =========================================================================
# 예측 함수
# =========================================================================
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


# =========================================================================
# 예측 실행 + 결과 표시
# =========================================================================
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
            "🎯 위험 점수", f"{risk_score:.4f}",
            help="모델이 산출한 스크리닝 점수 (보정된 확률 아님)",
        )
    with col2:
        st.metric(
            "📏 임계값", f"{SCREENING_THRESHOLD:.4f}",
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
            "※ 본 도구의 양성 판정은 추가 검진의 필요성을 시사하는 것이며, "
            "실제 고혈압 진단은 의료기관의 임상 평가로 확정됩니다."
        )
    else:
        st.success(
            f"✓ **음성** — 위험 점수 {risk_score:.4f} < 임계값 {SCREENING_THRESHOLD:.4f}\n\n"
            "본 스크리닝 모델은 귀하의 고혈압 위험을 **임계값 미만**으로 평가했습니다. "
            "건강한 생활습관 유지와 연 1회 정기 검진을 권장합니다."
        )

    with st.expander("📋 입력값 요약 보기"):
        rows = []
        for k, v in input_data.items():
            label = FEATURE_REGISTRY.get(k, {}).get("label", k)
            rows.append({"변수": label, "입력값": v})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.session_state["last_prediction"] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "input": input_data.copy(),
        "risk_score": risk_score,
        "threshold": SCREENING_THRESHOLD,
        "label": label,
    }


# =========================================================================
# 사용자 피드백
# =========================================================================
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


# =========================================================================
# 푸터
# =========================================================================
st.divider()
st.caption(
    "**Data**: 국민건강영양조사(KNHANES) 제9기 (질병관리청, 2022·2023·2024)  •  "
    f"**Model**: {screening_config.get('strategy', '-')} "
    f"{screening_config.get('model', '-')} "
    f"({screening_config.get('feature_version', '-')})  •  "
    "**분석 구조**: 메인 분석 (3개년 전체) + 보조 분석 (2개년·PHQ-9/EQ-5D 포함)"
)
st.caption(
    "© 2026 백승엽 (응용정보공학)  •  고급데이터분석 기말 프로젝트"
)
