import os
import pandas as pd
import streamlit as st



# --------------------------------------------------
# 기본 설정
# --------------------------------------------------
st.set_page_config(
    page_title="학교안전사고 예방대책 추천 시스템",
    page_icon="🛡️",
    layout="wide"
)

DATA_FILE = "통합_사고예방_데이터.csv"
TEMPLATE_FILE = "사고형태별_예방대책.csv"


# --------------------------------------------------
# 데이터 불러오기
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE, encoding="utf-8-sig")

    search_cols = ["학교급", "사고장소", "사고시간", "사고형태"]

    for col in search_cols:
        df[col] = (
            df[col]
            .fillna("정보 없음")
            .astype(str)
            .str.strip()
        )

    # 예상보상금 결측치는 평균보상액으로 대체
    df["최종예측보상액_출력용"] = pd.to_numeric(
        df["최종예측보상액"],
        errors="coerce"
    )

    df["평균보상액"] = pd.to_numeric(
        df["평균보상액"],
        errors="coerce"
    )

    df["최종예측보상액_출력용"] = (
        df["최종예측보상액_출력용"]
        .fillna(df["평균보상액"])
    )

    return df


df = load_data()


# --------------------------------------------------
# 안전한 값 출력 함수
# --------------------------------------------------
def safe_value(row, col, default="정보 없음"):
    if col not in row.index:
        return default

    value = row[col]

    if pd.isna(value):
        return default

    return value


def format_number(value, decimals=0):
    try:
        if pd.isna(value):
            return "정보 없음"

        return f"{float(value):,.{decimals}f}"

    except (TypeError, ValueError):
        return str(value)


# --------------------------------------------------
# SHAP 주요 요인 계산
# --------------------------------------------------
def get_shap_factors(row):
    shap_columns = {
        "학교급": "학교급_평균_절대_SHAP",
        "사고장소": "사고장소_평균_절대_SHAP",
        "사고시간": "사고시간_평균_절대_SHAP",
        "사고형태": "사고형태_평균_절대_SHAP"
    }

    factors = []

    for factor_name, col_name in shap_columns.items():
        try:
            score = float(row[col_name])

            if pd.notna(score):
                factors.append((factor_name, score))

        except (KeyError, TypeError, ValueError):
            continue

    factors.sort(key=lambda x: x[1], reverse=True)

    return factors[:3]


# --------------------------------------------------
# 예방대책 템플릿 및 하이브리드 추천
# --------------------------------------------------
@st.cache_data
def load_prevention_templates():
    """사고형태별로 미리 생성한 기본 예방대책을 불러옵니다."""
    if not os.path.exists(TEMPLATE_FILE):
        return {}

    try:
        template_df = pd.read_csv(TEMPLATE_FILE, encoding="utf-8-sig")
    except Exception:
        return {}

    required_cols = {"사고형태", "기본예방대책"}
    if not required_cols.issubset(template_df.columns):
        return {}

    template_df = template_df.dropna(subset=["사고형태", "기본예방대책"]).copy()
    template_df["사고형태"] = template_df["사고형태"].astype(str).str.strip()
    template_df["기본예방대책"] = template_df["기본예방대책"].astype(str).str.strip()
    return dict(zip(template_df["사고형태"], template_df["기본예방대책"]))


prevention_templates = load_prevention_templates()


def make_case_key(row):
    return "|".join([
        str(safe_value(row, "학교급")),
        str(safe_value(row, "사고장소")),
        str(safe_value(row, "사고시간")),
        str(safe_value(row, "사고형태")),
        str(safe_value(row, "위험등급")),
        str(safe_value(row, "TOPSIS_우선순위"))
    ])


def get_school_level_actions(school_level):
    school_level = str(school_level).strip()
    if any(word in school_level for word in ["유치원", "유아"]):
        return [
            "활동 전에 교사가 위험요소를 먼저 확인하고 전 과정을 가까이에서 감독합니다.",
            "안전수칙은 그림·시범·반복 체험 방식으로 짧고 명확하게 교육합니다.",
            "위험구역은 학생이 단독으로 접근하지 못하도록 물리적으로 분리합니다."
        ]
    if "초등" in school_level:
        return [
            "활동 시작 전 담당 교사가 핵심 안전수칙을 짧게 반복 안내합니다.",
            "학생의 신체 발달 수준을 고려해 활동구역과 난이도를 구분합니다.",
            "쉬는 시간과 이동 시간에는 사고 다발 구간의 순회 지도를 강화합니다."
        ]
    if "중학교" in school_level or school_level == "중":
        return [
            "학생이 활동 전 스스로 위험요소를 확인할 수 있도록 간단한 점검표를 제공합니다.",
            "장난·과도한 경쟁·안전수칙 미준수 행동에 대한 사전 지도를 실시합니다.",
            "반장이나 안전도우미를 활용해 학생 참여형 안전관리를 운영합니다."
        ]
    if "고등" in school_level:
        return [
            "실험·실습·체육활동 전에 위험요인과 실제 사고 사례를 중심으로 사전교육을 실시합니다.",
            "학생이 보호구와 시설 상태를 직접 확인하는 자율 점검 절차를 운영합니다.",
            "안전수칙 위반 시 즉시 활동을 중단하는 기준을 명확히 안내합니다."
        ]
    if "특수" in school_level:
        return [
            "학생별 신체·인지 특성을 고려한 개별 안전지원 계획을 적용합니다.",
            "필요한 경우 보조인력을 배치하고 이동 및 활동 구역을 단순화합니다.",
            "교사가 학생의 행동 변화와 피로 상태를 지속적으로 확인합니다."
        ]
    return [
        "학생의 발달 수준에 맞는 안전교육을 실시합니다.",
        "활동 전 담당 교직원이 안전수칙과 위험요소를 안내합니다.",
        "학생 이동과 활동 상황을 교직원이 정기적으로 확인합니다."
    ]


def get_location_actions(location):
    location = str(location).strip()
    if any(word in location for word in ["운동장", "운동 시설"]):
        return [
            "활동 전 바닥의 돌·패임·물기와 시설물의 고정 상태를 점검합니다.",
            "달리기·구기활동·놀이활동 구역을 분리해 학생 동선이 겹치지 않게 합니다.",
            "준비운동을 실시하고 활동 인원과 학생 간 간격을 관리합니다."
        ]
    if any(word in location for word in ["체육관", "강당"]):
        return [
            "벽면·기둥·체육기구 주변의 충돌 방지 설비와 바닥 상태를 확인합니다.",
            "사용하지 않는 체육기구는 활동구역 밖에 고정 보관합니다.",
            "동시에 활동하는 학생 수를 제한하고 종목별 공간을 분리합니다."
        ]
    if "계단" in location:
        return [
            "계단의 물기·파손·미끄럼 방지 상태와 난간 고정 상태를 매일 확인합니다.",
            "학생이 우측통행하고 난간을 이용하도록 이동수칙을 반복 안내합니다.",
            "등하교·쉬는 시간에는 이용 인원을 분산하거나 교직원을 배치합니다."
        ]
    if "복도" in location:
        return [
            "복도에 가방·청소도구·전선 등 이동을 방해하는 물건을 두지 않습니다.",
            "교차 지점과 출입문 주변의 시야를 확보하고 뛰지 않도록 지도합니다.",
            "쉬는 시간에는 학생 동선이 집중되는 구간을 중심으로 순회합니다."
        ]
    if any(word in location for word in ["교실", "학급"]):
        return [
            "책상과 의자 사이에 충분한 이동공간을 확보하고 통로에 물건을 두지 않습니다.",
            "창문·문·가구 모서리와 전기설비의 파손 여부를 정기적으로 확인합니다.",
            "수업 전후 학생이 뛰거나 위험한 장난을 하지 않도록 지도합니다."
        ]
    if any(word in location for word in ["급식", "식당", "조리"]):
        return [
            "바닥의 물기와 음식물을 즉시 제거하고 미끄럼 주의표지를 설치합니다.",
            "뜨거운 음식과 조리도구 이동 동선을 학생 이동 동선과 분리합니다.",
            "배식 대기 줄의 간격을 유지하고 교직원이 이동 흐름을 관리합니다."
        ]
    if any(word in location for word in ["과학실", "실험실"]):
        return [
            "실험 전 보호구 착용 여부와 실험기구의 파손 상태를 확인합니다.",
            "화학물질은 라벨과 보관기준에 따라 관리하고 학생 단독 사용을 제한합니다.",
            "환기·세안·소화설비 등 비상설비의 작동 상태를 정기적으로 점검합니다."
        ]
    if any(word in location for word in ["화장실", "세면"]):
        return [
            "바닥 물기를 수시로 제거하고 미끄럼 방지 상태를 확인합니다.",
            "파손된 타일·문·수도시설은 즉시 사용을 제한하고 수리합니다.",
            "혼잡 시간대에는 뛰거나 장난하지 않도록 순회 지도를 실시합니다."
        ]
    if any(word in location for word in ["통학", "도로", "횡단보도", "주차"]):
        return [
            "차량 동선과 학생 보행 동선을 물리적으로 분리합니다.",
            "등하교 시간에는 주요 진입로와 횡단보도에 안전인력을 배치합니다.",
            "운전자와 학생에게 정차 위치·제한속도·보행수칙을 안내합니다."
        ]
    if any(word in location for word in ["놀이터", "놀이"]):
        return [
            "놀이기구의 고정 상태·날카로운 부분·바닥 충격흡수 상태를 점검합니다.",
            "연령과 신체조건에 맞지 않는 놀이기구 사용을 제한합니다.",
            "교사가 놀이기구별 이용 인원과 학생 간 간격을 관리합니다."
        ]
    return [
        f"{location}의 바닥·시설물·이동 통로에 위험요소가 없는지 활동 전에 점검합니다.",
        "학생 활동구역과 이동구역이 겹치지 않도록 동선을 정리합니다.",
        "사고가 집중되는 시간에는 담당 교직원의 순회 점검을 강화합니다."
    ]


def build_hybrid_prevention_plan(row):
    school_level = str(safe_value(row, "학교급"))
    location = str(safe_value(row, "사고장소"))
    accident_time = str(safe_value(row, "사고시간"))
    accident_type = str(safe_value(row, "사고형태"))
    risk_grade = str(safe_value(row, "위험등급"))

    base_plan = prevention_templates.get(accident_type)
    if not base_plan:
        base_plan = """### 시설·환경 관리
- 활동 전에 시설과 주변 환경의 위험요소를 점검합니다.
- 파손되거나 안전성이 확인되지 않은 시설은 즉시 사용을 제한합니다.

### 학생 행동·안전교육
- 해당 사고유형의 핵심 안전수칙을 활동 전에 교육합니다.
- 위험행동을 발견하면 즉시 활동을 중단하고 다시 지도합니다.

### 감독·운영 관리
- 위험 활동에는 담당 교직원을 배치하고 학생 수를 관리합니다.
- 사고 다발 시간과 장소를 중심으로 점검 횟수를 늘립니다.

### 사고 발생 시 초기 대응
- 사고 발생 즉시 활동을 중단하고 부상 상태를 확인합니다.
- 응급처치 후 보호자 및 학교 보고 절차를 신속히 진행합니다."""

    school_actions = get_school_level_actions(school_level)
    location_actions = get_location_actions(location)
    priority_actions = [
        location_actions[0],
        school_actions[0],
        "활동 전 해당 사고형태의 핵심 안전수칙을 안내하고 준수 여부를 확인합니다."
    ]

    school_markdown = "\n".join(f"- {action}" for action in school_actions)
    location_markdown = "\n".join(f"- {action}" for action in location_actions)
    priority_markdown = "\n".join(
        f"{idx}. {action}" for idx, action in enumerate(priority_actions, start=1)
    )
    checklist_items = [
        f"{location}의 시설 및 바닥 상태를 점검했는가?",
        "학생 활동구역과 이동 동선을 분리했는가?",
        f"{school_level} 학생 수준에 맞는 안전교육을 실시했는가?",
        "담당 교직원과 사고 발생 시 역할을 공유했는가?",
        "응급물품과 비상연락 체계를 확인했는가?"
    ]
    checklist_markdown = "\n".join(f"- [ ] {item}" for item in checklist_items)

    return f"""## 선택 조건

- **학교급:** {school_level}
- **사고장소:** {location}
- **사고시간:** {accident_time}
- **사고형태:** {accident_type}
- **위험등급:** {risk_grade}

---

## 사고형태별 핵심 예방대책

{base_plan}

---

## {school_level} 맞춤 추가대책

{school_markdown}

---

## {location} 맞춤 추가대책

{location_markdown}

---

## 우선 실행순서

{priority_markdown}

---

## 현장 점검 체크리스트

{checklist_markdown}""".strip()


# --------------------------------------------------
# 화면 구성
# --------------------------------------------------
st.title("🛡️ 학교안전사고 예방대책 추천 시스템")

st.caption(
    "학교급·사고장소·사고시간·사고형태를 선택하면 "
    "위험도, 예상보상금, 사고빈도, SHAP 요인과 "
    "맞춤형 예방대책을 제공합니다."
)

st.divider()


# --------------------------------------------------
# 조건 선택
# --------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    school_level = st.selectbox(
        "학교급",
        sorted(df["학교급"].dropna().unique())
    )

    location_options = sorted(
        df.loc[
            df["학교급"] == school_level,
            "사고장소"
        ].dropna().unique()
    )

    accident_location = st.selectbox(
        "사고장소",
        location_options
    )

with col2:
    time_options = sorted(
        df.loc[
            (df["학교급"] == school_level)
            & (df["사고장소"] == accident_location),
            "사고시간"
        ].dropna().unique()
    )

    accident_time = st.selectbox(
        "사고시간",
        time_options
    )

    type_options = sorted(
        df.loc[
            (df["학교급"] == school_level)
            & (df["사고장소"] == accident_location)
            & (df["사고시간"] == accident_time),
            "사고형태"
        ].dropna().unique()
    )

    accident_type = st.selectbox(
        "사고형태",
        type_options
    )


# --------------------------------------------------
# 데이터 검색
# --------------------------------------------------
filtered = df[
    (df["학교급"] == school_level)
    & (df["사고장소"] == accident_location)
    & (df["사고시간"] == accident_time)
    & (df["사고형태"] == accident_type)
]

st.divider()

if filtered.empty:
    st.warning("선택한 조건과 일치하는 사고 데이터가 없습니다.")

else:
    # 동일 조건이 여러 개라면 TOPSIS 점수가 가장 높은 행 사용
    filtered = filtered.copy()

    filtered["TOPSIS_점수_정렬용"] = pd.to_numeric(
        filtered["TOPSIS_점수"],
        errors="coerce"
    )

    row = filtered.sort_values(
        "TOPSIS_점수_정렬용",
        ascending=False
    ).iloc[0]

    st.subheader("📊 사고 위험 분석 결과")

    metric1, metric2, metric3, metric4 = st.columns(4)


    with metric1:

        risk_grade = str(safe_value(row, "위험등급"))

        if "1등급" in risk_grade or "초고위험" in risk_grade:
            st.markdown(
                f"""
                <div style="
                    background:#ffebee;
                    border-left:8px solid #d32f2f;
                    padding:18px;
                    border-radius:12px;
                    text-align:center;
                ">
                    <h4 style="margin:0;color:#d32f2f;">🚨 위험등급</h4>
                    <h2 style="margin:6px;color:#d32f2f;">{risk_grade}</h2>
                </div>
                """,
                unsafe_allow_html=True
            )

        elif "2등급" in risk_grade or "심각" in risk_grade:
            st.markdown(
                f"""
                <div style="
                    background:#fff3e0;
                    border-left:8px solid #fb8c00;
                    padding:18px;
                    border-radius:12px;
                    text-align:center;
                ">
                    <h4 style="margin:0;color:#ef6c00;">⚠️ 위험등급</h4>
                    <h2 style="margin:6px;color:#ef6c00;">{risk_grade}</h2>
                </div>
                """,
                unsafe_allow_html=True
            )

        elif "3등급" in risk_grade:
            st.markdown(
                f"""
                <div style="
                    background:#fffde7;
                    border-left:8px solid #fbc02d;
                    padding:18px;
                    border-radius:12px;
                    text-align:center;
                ">
                    <h4 style="margin:0;color:#f9a825;">🟡 위험등급</h4>
                    <h2 style="margin:6px;color:#f9a825;">{risk_grade}</h2>
                </div>
                """,
                unsafe_allow_html=True
            )

        else:
            st.markdown(
                f"""
                <div style="
                    background:#e8f5e9;
                    border-left:8px solid #43a047;
                    padding:18px;
                    border-radius:12px;
                    text-align:center;
                ">
                    <h4 style="margin:0;color:#2e7d32;">✅ 위험등급</h4>
                    <h2 style="margin:6px;color:#2e7d32;">{risk_grade}</h2>
                </div>
                """,
                unsafe_allow_html=True
            )

    with metric2:
        st.metric(
            "위험도 점수",
            format_number(
                safe_value(row, "위험도_점수"),
                decimals=4
            )
        )

    with metric3:
        compensation = safe_value(
            row,
            "최종예측보상액_출력용"
        )

        if compensation == "정보 없음":
            compensation_text = compensation
        else:
            compensation_text = (
                f"{format_number(compensation, decimals=0)}원"
            )

        st.metric(
            "예상보상금",
            compensation_text
        )

    with metric4:
        accident_count = safe_value(row, "사고건수")

        if accident_count == "정보 없음":
            accident_count_text = accident_count
        else:
            accident_count_text = (
                f"{format_number(accident_count, decimals=0)}건"
            )

        st.metric(
            "사고빈도",
            accident_count_text
        )

    metric5, metric6, metric7 = st.columns(3)

    with metric5:
        st.metric(
            "TOPSIS 우선순위",
            format_number(
                safe_value(row, "TOPSIS_우선순위"),
                decimals=0
            )
        )

    with metric6:
        st.metric(
            "TOPSIS 점수",
            format_number(
                safe_value(row, "TOPSIS_점수"),
                decimals=4
            )
        )

    with metric7:
        probability = safe_value(
            row,
            "예측확률_1등급_초고위험"
        )

        if probability == "정보 없음":
            probability_text = probability
        else:
            try:
                probability_float = float(probability)

                if probability_float <= 1:
                    probability_float *= 100

                probability_text = f"{probability_float:.1f}%"

            except (TypeError, ValueError):
                probability_text = str(probability)

        st.metric(
            "초고위험 예측확률",
            probability_text
        )

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("🔍 SHAP 주요 영향요인")

        shap_factors = get_shap_factors(row)

        if shap_factors:
            for rank, (factor_name, score) in enumerate(
                shap_factors,
                start=1
            ):
                st.write(
                    f"**{rank}위. {factor_name}** — "
                    f"평균 절대 SHAP 영향도 `{score:.4f}`"
                )
        else:
            st.info("SHAP 중요요인 정보가 없습니다.")

    with right:
        st.subheader("📌 기존 분석 정보")

        st.write(
            "**모델 예측등급:**",
            safe_value(row, "모델_예측등급")
        )

        st.write(
            "**주요 위험상승요인:**",
            safe_value(row, "주요_위험상승요인")
        )

        st.write(
            "**기존 통합 예방전략:**",
            safe_value(row, "통합_예방전략")
        )

    st.divider()

    st.subheader("🧭 데이터 기반 맞춤형 예방대책")

    st.info(
        "선택된 사고 조건과 위험 분석 결과를 바탕으로 "
        "현장 적용형 예방대책을 제공합니다."
    )

    case_key = make_case_key(row)

    if st.button(
        "예방대책 확인",
        type="primary",
        use_container_width=True
    ):
        st.session_state["current_plan"] = build_hybrid_prevention_plan(row)
        st.session_state["current_case_key"] = case_key

    if (
        st.session_state.get("current_case_key") == case_key
        and st.session_state.get("current_plan")
    ):
        st.success("✅ 맞춤형 예방대책을 불러왔습니다.")

        with st.container(border=True):
            st.markdown(st.session_state["current_plan"])

    # -------------------------------
    # 공식 안전 지침
    # -------------------------------
    st.divider()
    st.subheader("📚 관련 안전 지침")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.link_button(
            "📘 학교 안전사고관리 지침",
            "https://www.moe.go.kr/boardCnts/viewRenew.do?boardID=141&boardSeq=103692&lev=0&m=0404&opType=N&s=moe&statusYN=W"
        )

    with col2:
        st.link_button(
            "⚖ 학교안전법",
            "https://www.law.go.kr/LSW/lsInfoP.do?chrClsCd=010202&lsId=010380&lsiSeq=268509&urlMode=lsInfoP"
        )

    with col3:
        st.link_button(
            "📘 학교안전교육 자료(학교안전지원시스템)",
            "https://www.schoolsafe24.or.kr/front/bbs/BBSMSTR_000000006018/bbsView.do?menuSn=214&upperMenuSn=151&bbscttNo=7104535",
            use_container_width=True
        )
        st.caption(
            "※ 학교안전교육 자료는 고등학교 자료가 먼저 표시되며, "
            "중학교·초등학교 자료는 같은 페이지에서 아래로 스크롤하여 확인할 수 있습니다."
        )
