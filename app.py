# app.py
"""
개인 지출 분석 대시보드 (5탭 구조)
탭1: 개요 | 탭2: 패턴 분석(G1) | 탭3: 코호트 비교 | 탭4: 예산 추천(G3) | 탭5: 이상치 탐지(G4)
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from openai import OpenAI
from utils.sub_category_rules import assign_sub_category

# ── AI 분석 함수 3종 (F006 / F007 / F008) ──────────────────────────

def _get_openai_client() -> OpenAI:
    """API 키를 secrets에서 안전하게 로드"""
    return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


def build_monthly_kpi(df: pd.DataFrame, target_month: str) -> dict:
    """target_month 기준 월간 KPI 계산 (전월 비교 포함)"""
    this_df = df[df["year_month"] == target_month]
    if this_df.empty:
        return {}

    total = int(this_df["amount"].sum())

    prev_month = (
        pd.to_datetime(target_month + "-01") - pd.DateOffset(months=1)
    ).strftime("%Y-%m")
    prev_df = df[df["year_month"] == prev_month]
    mom_rate = (
        round((total - prev_df["amount"].sum()) / prev_df["amount"].sum() * 100, 1)
        if not prev_df.empty and prev_df["amount"].sum() > 0
        else None
    )

    cat_share = (
        this_df.groupby("category")["amount"].sum()
        .sort_values(ascending=False)
        .apply(lambda x: round(x / total * 100, 1))
        .to_dict()
    )
    sub_top5 = (
        this_df.groupby("sub_category")["amount"].sum()
        .sort_values(ascending=False)
        .head(5)
        .to_dict()
    )

    return {
        "target_month": target_month,
        "total_spend":  total,
        "mom_rate":     mom_rate,
        "cat_share":    cat_share,
        "sub_top5":     sub_top5,
        "max_single":   int(this_df["amount"].max()),
        "tx_count":     len(this_df),
    }


def resolve_band(monthly_total: int) -> str:
    """월간 총지출 기준으로 코호트 밴드 자동 추론"""
    for limit, band in [
        (3_000_000, "300~400만"),
        (5_500_000, "400~550만"),
        (7_000_000, "550~700만"),
        (float("inf"), "700만+"),
    ]:
        if monthly_total < limit:
            return band
    return "700만+"


def f006_pattern_analysis(monthly_kpi: dict) -> str:
    """F006: 월간 지출 패턴 AI 분석"""
    top3 = "\n".join(
        f"  - {cat}: {share}%"
        for cat, share in list(monthly_kpi["cat_share"].items())[:3]
    )
    sub5 = "\n".join(
        f"  - {sub}: {amt:,}원"
        for sub, amt in monthly_kpi["sub_top5"].items()
    )
    mom_str = (
        f"{monthly_kpi['mom_rate']:+.1f}%"
        if monthly_kpi["mom_rate"] is not None else "전월 데이터 없음"
    )
    prompt = f"""
당신은 데이터 분석 전문 재무 코치입니다.
{monthly_kpi['target_month']} 지출 패턴을 분석하고 인사이트를 제공하세요.

[이번 달 지표]
- 총지출: {monthly_kpi['total_spend']:,}원 (전월 대비 {mom_str})
- 거래 건수: {monthly_kpi['tx_count']}건 / 최대 단건: {monthly_kpi['max_single']:,}원

[카테고리 비중 TOP3]
{top3}

[세부 항목 TOP5]
{sub5}

[요청]
1. 이번 달 지출 패턴에서 주목할 점 2가지 (데이터 근거 포함)
2. 소비 습관 관점 한 줄 총평
간결하게 팩트 중심으로 작성하세요.
"""
    resp = _get_openai_client().chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "냉철하고 분석적인 재무 컨설턴트입니다."},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.4,
        max_tokens=500,
    )
    return resp.choices[0].message.content


def f007_budget_recommendation(df: pd.DataFrame, target_month: str) -> str:
    """F007: 최근 3개월 평균 기반 다음 달 예산 AI 추천"""
    months = sorted(df["year_month"].unique())
    idx = months.index(target_month) if target_month in months else len(months) - 1
    recent_3 = months[max(0, idx - 2): idx + 1]

    recent_df = df[df["year_month"].isin(recent_3)]
    avg_total = int(recent_df.groupby("year_month")["amount"].sum().mean())
    cat_avg = (
        recent_df.groupby(["year_month", "category"])["amount"]
        .sum().groupby("category").mean()
        .sort_values(ascending=False)
        .apply(lambda x: int(round(x, -2)))
        .to_dict()
    )
    cat_str = "\n".join(f"  - {cat}: {amt:,}원" for cat, amt in cat_avg.items())

    prompt = f"""
최근 {len(recent_3)}개월({', '.join(recent_3)}) 평균 기반으로 다음 달 예산을 추천하세요.

[월평균 총지출] {avg_total:,}원
[카테고리별 월평균]
{cat_str}

[요청]
1. 다음 달 권장 총 예산 (구체적 금액)
2. 카테고리별 권장 예산 상위 5개 (절감 이유 포함)
3. 가장 먼저 줄여야 할 항목 1개와 목표 절감액
현실적이고 실행 가능한 금액으로 작성하세요.
"""
    resp = _get_openai_client().chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "냉철하고 분석적인 재무 컨설턴트입니다."},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.4,
        max_tokens=500,
    )
    return resp.choices[0].message.content


def f008_build_report(monthly_kpi: dict, pattern: str, budget: str) -> str:
    """F008: F006 + F007 결과를 합쳐 다운로드용 마크다운 리포트 생성"""
    cat_rows = "\n".join(
        f"| {cat} | {share}% |"
        for cat, share in monthly_kpi["cat_share"].items()
    )
    sub_rows = "\n".join(
        f"| {sub} | {amt:,}원 |"
        for sub, amt in monthly_kpi["sub_top5"].items()
    )
    mom_str = (
        f"{monthly_kpi['mom_rate']:+.1f}%"
        if monthly_kpi["mom_rate"] is not None else "전월 데이터 없음"
    )
    return f"""# {monthly_kpi['target_month']} 월간 지출 리포트

## 1. 핵심 지표
| 항목 | 값 |
|---|---|
| 총지출 | {monthly_kpi['total_spend']:,}원 |
| 전월 대비 | {mom_str} |
| 거래 건수 | {monthly_kpi['tx_count']}건 |
| 최대 단건 | {monthly_kpi['max_single']:,}원 |

## 2. 카테고리별 비중
| 카테고리 | 비중 |
|---|---|
{cat_rows}

## 3. 세부 항목 TOP5
| 항목 | 금액 |
|---|---|
{sub_rows}

## 4. AI 패턴 분석 (F006)
{pattern}

## 5. 다음 달 예산 추천 (F007)
{budget}
"""
# ↓ 이렇게 교체
from utils.data_processor import (
    preprocess,
    calc_kpi,
    build_monthly_kpi,
    resolve_band,
)
from utils.ai_analyzer import (
    f006_pattern_analysis,
    f007_budget_recommendation,
    f008_build_report,
)
from gitignore.cohort_engine import (
    run_cohort_analysis,
    run_band_cohort_analysis,
    load_cohort_parquet,
    INCOME_BANDS,
)


@st.cache_data(show_spinner="코호트 데이터 로딩 중...")
def _load_cohort() -> "pd.DataFrame":
    return load_cohort_parquet("data/cohort.parquet")

st.set_page_config(page_title="개인 지출 분석 대시보드", layout="wide")
st.title("개인 지출 분석 대시보드")


# =========================================================
# 파일 업로드 및 전처리
# =========================================================

uploaded_file = st.file_uploader(
    "지출 데이터(CSV/Excel)를 업로드하세요",
    type=["csv", "xlsx"]
)


def read_file(file):
    name = file.name.lower()
    if name.endswith(".csv"):
        try:
            return pd.read_csv(file, encoding="utf-8-sig")
        except UnicodeDecodeError:
            return pd.read_csv(file, encoding="cp949")
    elif name.endswith(".xlsx"):
        return pd.read_excel(file)


if uploaded_file is None:
    st.info("파일을 업로드하면 대시보드가 생성됩니다.")
    st.stop()

df_raw = read_file(uploaded_file)
if df_raw is None or df_raw.empty:
    st.error("파일을 읽지 못했거나 데이터가 비어있습니다.")
    st.stop()


# =========================================================
# 전처리 → df_proc 생성
# =========================================================

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = ["date", "amount", "category"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"필수 컬럼 누락: {missing}")
        st.stop()

    df = df.copy()
    df["date"]   = pd.to_datetime(df["date"],  errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    # ✅ dropna 먼저 (NaN이 "nan" 문자열로 변신하기 전에)
    df = df.dropna(subset=["date", "amount", "category"])
    df["category"] = df["category"].astype(str).str.strip()
    df = df[df["category"].ne("")]  # 공백만 있던 행 추가 제거

    for col, default in [
        ("description",    ""),
        ("payment_method", ""),
        ("is_fixed",       False),
        ("memo",           ""),
    ]:
        if col not in df.columns:
            df[col] = default

    df["year_month"]   = df["date"].dt.strftime("%Y-%m")
    df["weekday"]      = df["date"].dt.weekday
    weekday_kr_map     = {0:"월",1:"화",2:"수",3:"목",4:"금",5:"토",6:"일"}
    df["weekday_kr"]   = df["weekday"].map(weekday_kr_map)
    df["day_type"]     = df["weekday"].apply(lambda x: "WEEKEND" if x >= 5 else "WEEKDAY")
    df["season"]       = df["date"].dt.month.map(
        {12:"겨울",1:"겨울",2:"겨울",
          3:"봄",  4:"봄",  5:"봄",
          6:"여름",7:"여름",8:"여름",
          9:"가을",10:"가을",11:"가을"}
    )
    df["sub_category"] = df.apply(assign_sub_category, axis=1)
    return df.sort_values("date").reset_index(drop=True)


df_proc = preprocess(df_raw)


# =========================================================
# 사이드바 필터
# =========================================================

st.sidebar.header("필터")

min_date = df_proc["date"].min().date()
max_date = df_proc["date"].max().date()

st.sidebar.subheader("날짜 범위")
start_date = st.sidebar.date_input("시작일", value=min_date, min_value=min_date, max_value=max_date)
end_date   = st.sidebar.date_input("종료일", value=max_date, min_value=min_date, max_value=max_date)

if start_date > end_date:
    st.sidebar.warning("시작일이 종료일보다 클 수 없습니다.")
    st.stop()

all_categories = sorted(df_proc["category"].astype(str).unique())

for cat in all_categories:
    st.session_state.setdefault(f"cat_{cat}", True)

st.sidebar.subheader("카테고리 선택")

col_a, col_b = st.sidebar.columns(2)
if col_a.button("전체 +"):
    for cat in all_categories:
        st.session_state[f"cat_{cat}"] = True
    st.rerun()
if col_b.button("전체 -"):
    for cat in all_categories:
        st.session_state[f"cat_{cat}"] = False
    st.rerun()

for cat in all_categories:
    key = f"cat_{cat}"
    if key not in st.session_state:
        st.session_state[key] = True
    st.sidebar.checkbox(cat, key=key)

selected_categories = [cat for cat in all_categories if st.session_state.get(f"cat_{cat}", True)]

if len(selected_categories) == 0:
    st.sidebar.warning("카테고리를 1개 이상 선택해주세요.")
    st.stop()

st.sidebar.subheader("금액 범위")
min_amt = int(df_proc["amount"].min())
max_amt = int(df_proc["amount"].max())
step = 1000 if (max_amt - min_amt) >= 1000 else max(1, (max_amt - min_amt) or 1)
amt_range = st.sidebar.slider("지출 금액 범위", min_value=min_amt, max_value=max_amt,
                               value=(min_amt, max_amt), step=step)
min_sel, max_sel = amt_range

# 필터 적용
filtered = df_proc[
    (df_proc["date"].dt.date >= start_date) &
    (df_proc["date"].dt.date <= end_date) &
    (df_proc["category"].isin(selected_categories)) &
    (df_proc["amount"] >= min_sel) &
    (df_proc["amount"] <= max_sel)
].copy()


# =========================================================
# KPI 공통 계산
# =========================================================

def calc_kpi(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    total_spend   = df["amount"].sum()
    monthly_avg   = df.groupby("year_month")["amount"].sum().mean()
    max_tx        = df["amount"].max()
    fixed_ratio   = (df[df["is_fixed"] == True]["amount"].sum() / total_spend * 100
                     if total_spend > 0 else 0)
    monthly_total = df.groupby("year_month")["amount"].sum().sort_index()
    mom_rate      = 0.0
    if len(monthly_total) >= 2:
        last, prev = monthly_total.iloc[-1], monthly_total.iloc[-2]
        mom_rate   = round((last - prev) / prev * 100, 1) if prev != 0 else 0.0

    cat_share = (df.groupby("category")["amount"].sum() / total_spend * 100).round(1).to_dict()

    return {
        "total_spend":   int(total_spend),
        "monthly_avg":   int(monthly_avg),
        "max_tx":        int(max_tx),
        "fixed_ratio":   round(fixed_ratio, 1),
        "mom_rate":      mom_rate,
        "cat_share":     cat_share,
    }


kpi = calc_kpi(filtered)

def _parse_platform(desc: str) -> str:
    """'쿠팡 | 생필품 구매' → '쿠팡'"""
    if " | " in desc:
        return desc.split(" | ")[0].strip()
    return None


def _build_personal_insights(filtered: pd.DataFrame) -> dict:
    """
    개인 데이터 특성 감지 후 섹션별 데이터 반환
    없는 섹션은 빈값으로 → 렌더링에서 조건부 표시
    """
    result = {}

    # ── 1) 카페 ───────────────────────────────────────────────
    cafe_keywords = ["스타벅스", "투썸", "이디야", "메가", "컴포즈", "더벤티",
                     "커피베이", "공차", "파스쿠찌", "매머드", "개인카페"]
    cafe_df = filtered[
        filtered["description"].str.contains("|".join(cafe_keywords), na=False)
    ]
    if len(cafe_df) >= 3:  # 최소 3건 이상일 때만 표시
        def extract_cafe(desc):
            for kw in cafe_keywords:
                if kw in desc:
                    return kw
            return None

        cafe_df = cafe_df.copy()
        cafe_df["브랜드"] = cafe_df["description"].apply(extract_cafe)
        cafe_top = (
            cafe_df.groupby("브랜드")
            .agg(횟수=("amount", "count"), 총지출=("amount", "sum"))
            .sort_values("총지출", ascending=False)
            .head(3)
            .reset_index()
        )
        result["cafe"] = {
            "total": int(cafe_df["amount"].sum()),
            "count": len(cafe_df),
            "top":   cafe_top,
        }

    # ── 2) 쇼핑 플랫폼 ────────────────────────────────────────
    shop_df = filtered[
        (filtered["category"] == "쇼핑") &
        (filtered["description"].str.contains(r"\|", na=False))
    ].copy()
    if len(shop_df) >= 3:
        shop_df["플랫폼"] = shop_df["description"].apply(_parse_platform)
        shop_top = (
            shop_df.groupby("플랫폼")
            .agg(횟수=("amount", "count"), 총지출=("amount", "sum"))
            .sort_values("총지출", ascending=False)
            .head(3)
            .reset_index()
        )
        result["shopping"] = {
            "total": int(shop_df["amount"].sum()),
            "top":   shop_top,
        }

    # ── 3) 취미 ───────────────────────────────────────────────
    hobby_df = filtered[filtered["memo"].str.contains("취미\\(", na=False)].copy()
    if len(hobby_df) >= 1:
        def extract_hobby(memo):
            # "취미(게임|light)" → "게임"
            import re
            m = re.search(r"취미\(([^|)]+)", memo)
            return m.group(1) if m else None

        hobby_df["취미종류"] = hobby_df["memo"].apply(extract_hobby)
        hobby_sum = (
            hobby_df.groupby("취미종류")
            .agg(총지출=("amount", "sum"), 횟수=("amount", "count"))
            .sort_values("총지출", ascending=False)
            .reset_index()
        )
        result["hobby"] = {
            "total": int(hobby_df["amount"].sum()),
            "detail": hobby_sum,
        }

    # ── 4) 덕질 ───────────────────────────────────────────────
    fandom_df = filtered[filtered["memo"].str.contains("덕질", na=False)].copy()
    if len(fandom_df) >= 1:
        concert_amt = int(
            fandom_df[fandom_df["memo"].str.contains("콘서트", na=False)]["amount"].sum()
        )
        goods_amt = int(
            fandom_df[~fandom_df["memo"].str.contains("콘서트", na=False)]["amount"].sum()
        )
        result["fandom"] = {
            "total":   int(fandom_df["amount"].sum()),
            "concert": concert_amt,
            "goods":   goods_amt,
            "count":   len(fandom_df),
        }

    # ── 5) 반려동물 ───────────────────────────────────────────
    pet_df = filtered[filtered["memo"].str.contains("반려동물", na=False)].copy()
    if len(pet_df) >= 1:
        pet_by_type = (
            pet_df.groupby("description")["amount"]
            .sum().sort_values(ascending=False)
            .head(3).reset_index()
        )
        pet_by_type.columns = ["항목", "지출금액"]
        result["pet"] = {
            "total":  int(pet_df["amount"].sum()),
            "detail": pet_by_type,
        }
# ── 6) 교통 패턴 ──────────────────────────────────────────
    trans_df = filtered[filtered["category"] == "교통비"].copy()
    if len(trans_df) >= 3:
        taxi_amt   = int(trans_df[trans_df["description"] == "택시"]["amount"].sum())
        public_amt = int(trans_df[trans_df["description"].isin(["지하철 교통카드","버스"])]["amount"].sum())
        car_amt    = int(trans_df[trans_df["description"].isin(["주유","주차","통행료","세차","정비/오일"])]["amount"].sum())
        total_trans = taxi_amt + public_amt + car_amt
        if total_trans > 0:
            result["transport"] = {
                "total":  int(trans_df["amount"].sum()),
                "taxi":   taxi_amt,
                "public": public_amt,
                "car":    car_amt,
            }

    # ── 7) 배달 vs 외식 ───────────────────────────────────────
    food_df = filtered[filtered["category"] == "식비"].copy()
    if len(food_df) >= 5:
        delivery_amt = int(food_df[food_df["description"].str.contains("배달", na=False)]["amount"].sum())
        dine_amt     = int(food_df[food_df["description"].isin(["점심 식사","저녁 외식","회식"])]["amount"].sum())
        grocery_amt  = int(food_df[food_df["description"].str.contains("장보기|마트", na=False)]["amount"].sum())
        if delivery_amt + dine_amt + grocery_amt > 0:
            result["food_style"] = {
                "total":    int(food_df["amount"].sum()),
                "delivery": delivery_amt,
                "dine_out": dine_amt,
                "grocery":  grocery_amt,
            }

    # ── 8) 뷰티 ───────────────────────────────────────────────
    beauty_keywords = ["미용실", "네일", "왁싱", "피부관리"]
    beauty_df = filtered[
        filtered["description"].str.contains("|".join(beauty_keywords), na=False)
    ].copy()
    if len(beauty_df) >= 1:
        beauty_df["항목"] = beauty_df["description"].apply(
            lambda d: next((k for k in beauty_keywords if k in d), "기타")
        )
        beauty_sum = (
            beauty_df.groupby("항목")
            .agg(총지출=("amount","sum"), 횟수=("amount","count"))
            .sort_values("총지출", ascending=False)
            .reset_index()
        )
        result["beauty"] = {
            "total":  int(beauty_df["amount"].sum()),
            "detail": beauty_sum,
        }

    # ── 9) 의료 ───────────────────────────────────────────────
    medical_df = filtered[filtered["category"] == "의료/건강"].copy()
    if len(medical_df) >= 1:
        ortho_amt   = int(medical_df[medical_df["memo"].str.contains("교정", na=False)]["amount"].sum())
        checkup_amt = int(medical_df[medical_df["memo"].str.contains("검진", na=False)]["amount"].sum())
        derm_amt    = int(medical_df[medical_df["memo"].str.contains("피부과|시술", na=False)]["amount"].sum())
        general_amt = int(medical_df["amount"].sum()) - ortho_amt - checkup_amt - derm_amt
        result["medical"] = {
            "total":   int(medical_df["amount"].sum()),
            "ortho":   ortho_amt,
            "checkup": checkup_amt,
            "derm":    derm_amt,
            "general": general_amt,
        }

    # ── 10) 자기계발 ──────────────────────────────────────────
    edu_df = filtered[filtered["category"] == "교육"].copy()
    if len(edu_df) >= 1:
        edu_df["항목"] = edu_df["description"].apply(
            lambda d: "등록금" if "등록금" in d
            else "학원비(자녀)" if "자녀" in d
            else "학원비(본인)" if "본인" in d
            else "도서" if any(k in d for k in ["도서","책"])
            else "자격증" if "자격증" in d
            else "강의/세미나" if any(k in d for k in ["강의","특강","세미나"])
            else "기타"
        )
        edu_sum = (
            edu_df.groupby("항목")["amount"]
            .sum().sort_values(ascending=False)
            .reset_index()
        )
        edu_sum.columns = ["항목", "지출금액"]
        result["education"] = {
            "total":  int(edu_df["amount"].sum()),
            "detail": edu_sum,
        }

    # ── 11) 여행 ──────────────────────────────────────────────
    travel_keywords = ["항공권", "호텔 숙박비", "리조트 숙박비", "에어비앤비 숙박비", "KTX/기차"]
    travel_df = filtered[
        filtered["description"].str.contains("|".join(travel_keywords), na=False)
    ].copy()
    if len(travel_df) >= 1:
        travel_df["항목"] = travel_df["description"].apply(
            lambda d: next((k for k in travel_keywords if k in d), "기타")
        )
        travel_sum = (
            travel_df.groupby("항목")["amount"]
            .sum().sort_values(ascending=False)
            .reset_index()
        )
        travel_sum.columns = ["항목", "지출금액"]
        result["travel"] = {
            "total":  int(travel_df["amount"].sum()),
            "count":  len(travel_df),
            "detail": travel_sum,
        }

    # ── 12) 선물/경조사 ───────────────────────────────────────
    gift_df = filtered[
        filtered["description"].str.contains("선물|경조사|카카오선물하기", na=False)
    ].copy()
    if len(gift_df) >= 1:
        result["gift"] = {
            "total": int(gift_df["amount"].sum()),
            "count": len(gift_df),
            "avg":   int(gift_df["amount"].mean()),
        }

    # ── 13) 중고거래 ──────────────────────────────────────────
    second_df = filtered[
        filtered["description"].str.contains("당근마켓|번개장터|중고나라", na=False)
    ].copy()
    if len(second_df) >= 1:
        second_sum = (
            second_df.groupby("description")["amount"]
            .sum().sort_values(ascending=False)
            .reset_index()
        )
        second_sum.columns = ["플랫폼", "지출금액"]
        # 플랫폼명 정리 ("당근마켓 | 중고 의류 구매" → "당근마켓")
        second_sum["플랫폼"] = second_sum["플랫폼"].apply(
            lambda d: d.split(" | ")[0].strip()
        )
        result["secondhand"] = {
            "total":  int(second_df["amount"].sum()),
            "count":  len(second_df),
            "detail": second_sum,
        }

    return result



def render_personal_insights(filtered: pd.DataFrame):
    """개인 데이터 특성 기반 섹션 조건부 렌더링"""

    data = _build_personal_insights(filtered)

    if not data:
        st.info("개인화 인사이트를 생성할 데이터가 부족합니다.")
        return

    st.markdown("#### ☕ 내 소비 스타일 분석")

    # 카페
    if "cafe" in data:
        with st.expander(
            f"☕ 카페 — 총 {data['cafe']['total']:,}원 / {data['cafe']['count']}회",
            expanded=False,
        ):
            top = data["cafe"]["top"]
            col_tbl, col_bar = st.columns(2)
            with col_tbl:
                st.dataframe(top.rename(columns={"브랜드": "카페"}), use_container_width=True)
            with col_bar:
                fig = px.bar(
                    top, x="브랜드", y="총지출",
                    text="횟수",
                    color="브랜드",
                    labels={"총지출": "지출금액"},
                )
                fig.update_traces(texttemplate="%{text}회", textposition="outside")
                fig.update_layout(
                    showlegend=False, yaxis_tickformat=",",
                    xaxis_title=None, height=250,
                )
                st.plotly_chart(fig, use_container_width=True)

    # 쇼핑 플랫폼
    if "shopping" in data:
        with st.expander(
            f"🛒 쇼핑 플랫폼 — 총 {data['shopping']['total']:,}원",
            expanded=False,
        ):
            top = data["shopping"]["top"]
            col_tbl, col_bar = st.columns(2)
            with col_tbl:
                st.dataframe(top.rename(columns={"플랫폼": "플랫폼"}), use_container_width=True)
            with col_bar:
                fig = px.bar(
                    top, x="플랫폼", y="총지출",
                    text="횟수", color="플랫폼",
                    labels={"총지출": "지출금액"},
                )
                fig.update_traces(texttemplate="%{text}회", textposition="outside")
                fig.update_layout(
                    showlegend=False, yaxis_tickformat=",",
                    xaxis_title=None, height=250,
                )
                st.plotly_chart(fig, use_container_width=True)

    # 취미
    if "hobby" in data:
        with st.expander(
            f"🎮 취미 — 총 {data['hobby']['total']:,}원",
            expanded=False,
        ):
            detail = data["hobby"]["detail"]
            col_tbl, col_bar = st.columns(2)
            with col_tbl:
                st.dataframe(detail, use_container_width=True)
            with col_bar:
                fig = px.bar(
                    detail, x="취미종류", y="총지출",
                    text="횟수", color="취미종류",
                    labels={"총지출": "지출금액"},
                )
                fig.update_traces(texttemplate="%{text}회", textposition="outside")
                fig.update_layout(
                    showlegend=False, yaxis_tickformat=",",
                    xaxis_title=None, height=250,
                )
                st.plotly_chart(fig, use_container_width=True)

    # 덕질
    if "fandom" in data:
        d = data["fandom"]
        with st.expander(
            f"🎤 덕질 — 총 {d['total']:,}원 / {d['count']}건",
            expanded=False,
        ):
            f1, f2, f3 = st.columns(3)
            f1.metric("덕질 총지출",   f"{d['total']:,}원")
            f2.metric("콘서트/팬미팅", f"{d['concert']:,}원")
            f3.metric("굿즈/소액",     f"{d['goods']:,}원")

    # 반려동물
    if "pet" in data:
        with st.expander(
            f"🐾 반려동물 — 총 {data['pet']['total']:,}원",
            expanded=False,
        ):
            col_tbl, col_bar = st.columns(2)
            with col_tbl:
                st.dataframe(data["pet"]["detail"], use_container_width=True)
            with col_bar:
                fig = px.bar(
                    data["pet"]["detail"], x="항목", y="지출금액",
                    color="항목", text="지출금액",
                )
                fig.update_traces(texttemplate="%{text:,.0f}원", textposition="outside")
                fig.update_layout(
                    showlegend=False, yaxis_tickformat=",",
                    xaxis_title=None, height=250,
                )
                st.plotly_chart(fig, use_container_width=True)
# 교통 패턴
    if "transport" in data:
        d = data["transport"]
        with st.expander(f"🚗 교통 패턴 — 총 {d['total']:,}원", expanded=False):
            t1, t2, t3 = st.columns(3)
            t1.metric("대중교통", f"{d['public']:,}원")
            t2.metric("택시",     f"{d['taxi']:,}원")
            t3.metric("차량",     f"{d['car']:,}원")
            trans_pie = pd.DataFrame({
                "항목":   ["대중교통", "택시", "차량"],
                "금액":   [d["public"], d["taxi"], d["car"]],
            })
            trans_pie = trans_pie[trans_pie["금액"] > 0]
            fig = px.pie(trans_pie, names="항목", values="금액", hole=0.35)
            fig.update_layout(height=220, margin=dict(t=10,b=10))
            st.plotly_chart(fig, use_container_width=True)

    # 배달 vs 외식
    if "food_style" in data:
        d = data["food_style"]
        with st.expander(f"🍔 식비 스타일 — 총 {d['total']:,}원", expanded=False):
            f1, f2, f3 = st.columns(3)
            f1.metric("배달",   f"{d['delivery']:,}원")
            f2.metric("외식",   f"{d['dine_out']:,}원")
            f3.metric("장보기", f"{d['grocery']:,}원")

    # 뷰티
    if "beauty" in data:
        with st.expander(f"💆 뷰티 — 총 {data['beauty']['total']:,}원", expanded=False):
            st.dataframe(data["beauty"]["detail"], use_container_width=True)

    # 의료
    if "medical" in data:
        d = data["medical"]
        with st.expander(f"🏥 의료/건강 — 총 {d['total']:,}원", expanded=False):
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("교정",     f"{d['ortho']:,}원")
            m2.metric("검진",     f"{d['checkup']:,}원")
            m3.metric("피부과",   f"{d['derm']:,}원")
            m4.metric("일반 진료",f"{d['general']:,}원")

    # 자기계발
    if "education" in data:
        with st.expander(f"📚 자기계발 — 총 {data['education']['total']:,}원", expanded=False):
            st.dataframe(data["education"]["detail"], use_container_width=True)

    # 여행
    if "travel" in data:
        with st.expander(
            f"✈️ 여행 — 총 {data['travel']['total']:,}원 / {data['travel']['count']}건",
            expanded=False,
        ):
            st.dataframe(data["travel"]["detail"], use_container_width=True)

    # 선물/경조사
    if "gift" in data:
        d = data["gift"]
        with st.expander(f"🎁 선물/경조사 — 총 {d['total']:,}원", expanded=False):
            g1, g2 = st.columns(2)
            g1.metric("건수",       f"{d['count']}건")
            g2.metric("건당 평균",  f"{d['avg']:,}원")

    # 중고거래
    if "secondhand" in data:
        d = data["secondhand"]
        with st.expander(f"♻️ 중고거래 — 총 {d['total']:,}원 / {d['count']}건", expanded=False):
            st.dataframe(d["detail"], use_container_width=True)


# =========================================================
# 탭 구성
# =========================================================

# 탭 7개로 확장
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "내 지출 현황",
    "구독/고정비",
    "패턴 분석",
    "코호트 비교",
    "예산 추천",
    "이상치 탐지",
    "월간 리포트",
])


# =========================================================
# TAB 1: 개요
# =========================================================

def render_tab1(filtered: pd.DataFrame, kpi: dict, df_raw: pd.DataFrame):
    
    # 원본 요약 + 필터 요약 분리
    st.subheader("데이터 미리보기")

    col_raw, col_filtered = st.columns(2)

    with col_raw:
        st.markdown("**📁 원본 데이터 요약**")
        r1, r2 = st.columns(2)
        r1.metric("전체 행 수",   f"{len(df_raw):,}")
        r2.metric("전체 총지출",  f"{df_raw['amount'].sum():,.0f}원")

    with col_filtered:
        st.markdown("**🔍 필터 적용 요약**")
        f1, f2, f3 = st.columns(3)
        f1.metric("필터 행 수",       f"{len(filtered):,}")
        f2.metric("기간",             f"{start_date} ~ {end_date}")
        f3.metric("필터 총지출",      f"{filtered['amount'].sum():,.0f}원")

    st.markdown("**원본 데이터 (상위 15건)**")
    derived_cols  = ["year_month", "weekday", "weekday_kr", "day_type", "sub_category", "season"]
    raw_show_cols = [c for c in df_raw.columns if c not in derived_cols]
    st.dataframe(df_raw[raw_show_cols].head(15), use_container_width=True)

    st.markdown("**필터 적용 데이터 (상위 15건)**")
    filtered_display = filtered[raw_show_cols].head(15).copy()
    filtered_display["date"] = filtered_display["date"].dt.date  # 시간 제거
    st.dataframe(filtered_display, use_container_width=True)

    
    st.subheader("지출 요약 통계")

    if not kpi:
        st.warning("선택한 조건에서 데이터가 없습니다.")
        return

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("총지출",         f"{kpi['total_spend']:,}원")
    c2.metric("월 평균 지출",   f"{kpi['monthly_avg']:,}원")
    c3.metric("최대 지출(1건)", f"{kpi['max_tx']:,}원")
    c4.metric("고정지출 비중",  f"{kpi['fixed_ratio']:.1f}%")
    c5.metric("전월 대비 변화", f"{kpi['mom_rate']:.1f}%")

    st.caption("카테고리별 비중(%) Top5")
    share = pd.Series(kpi["cat_share"]).sort_values(ascending=False).head(5)
    st.dataframe(share.rename("비중(%)").to_frame(), use_container_width=True)

    left, right = st.columns(2)

    with left:
        st.subheader("카테고리별 지출 비중")
        cat_sum = filtered.groupby("category")["amount"].sum().sort_values(ascending=False)
        if not cat_sum.empty:
            top_n = 6
            top   = cat_sum.head(top_n).copy()
            etc   = cat_sum.iloc[top_n:].sum()
            if etc > 0:
                top["기타"] = etc
            pie_df = top.reset_index()
            pie_df.columns = ["category", "amount"]
            fig = px.pie(pie_df, names="category", values="amount", hole=0.0)
            fig.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("카테고리별 지출 금액")
        cat_bar = (filtered.groupby("category", as_index=False)["amount"]
                   .sum().sort_values("amount", ascending=False))
        if not cat_bar.empty:
            fig_bar = px.bar(cat_bar, x="category", y="amount", text="amount")
            fig_bar.update_traces(texttemplate="%{text:,.0f}원", textposition="outside")
            fig_bar.update_layout(yaxis_tickformat=",", xaxis_title=None, yaxis_title="지출 금액")
            st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("월별 총지출 추이")
    monthly_df = (filtered.groupby("year_month", as_index=False)["amount"]
                  .sum().sort_values("year_month"))
    if not monthly_df.empty:
        fig2 = px.line(monthly_df, x="year_month", y="amount", markers=True)
        N = 6
        if len(monthly_df) > 1:
            x_end   = monthly_df["year_month"].iloc[-1]
            x_start = monthly_df["year_month"].iloc[max(0, len(monthly_df) - N)]
            fig2.update_xaxes(range=[x_start, x_end])
        st.plotly_chart(fig2, use_container_width=True)



with tab1:
    render_tab1(filtered, kpi, df_raw)

# =========================================================
# TAB 2: 구독/고정비 분석 (G2)
# =========================================================

def render_tab_g2(filtered: pd.DataFrame):
    st.subheader("구독/고정비 분석 (G2)")
    st.caption("is_fixed=True 항목 기준으로 고정·구독 지출을 분석합니다.")

    if filtered.empty:
        st.warning("선택한 조건에서 데이터가 없습니다.")
        return

    # ── 데이터 분리 ──────────────────────────────────────────
    fixed_df = filtered[filtered["is_fixed"] == True].copy()
    var_df   = filtered[filtered["is_fixed"] == False].copy()

    if fixed_df.empty:
        st.info("고정지출 항목이 없습니다. (is_fixed=True 데이터 없음)")
        return

    total_all   = filtered["amount"].sum()
    total_fixed = fixed_df["amount"].sum()
    fixed_ratio = round(total_fixed / total_all * 100, 1) if total_all > 0 else 0

    # ── KPI 카드 ─────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)

    # 월평균 고정지출
    monthly_fixed_avg = int(
        fixed_df.groupby("year_month")["amount"].sum().mean()
    )

    # 구독 항목 수 (category == "구독")
    sub_count = fixed_df[fixed_df["category"] == "구독"]["description"].nunique()

    # 전월 대비 고정지출 증감률
    months_sorted = sorted(fixed_df["year_month"].unique())
    if len(months_sorted) >= 2:
        last_m = months_sorted[-1]
        prev_m = months_sorted[-2]
        last_fixed = fixed_df[fixed_df["year_month"] == last_m]["amount"].sum()
        prev_fixed = fixed_df[fixed_df["year_month"] == prev_m]["amount"].sum()
        mom_fixed = round((last_fixed - prev_fixed) / prev_fixed * 100, 1) if prev_fixed > 0 else 0.0
    else:
        mom_fixed = 0.0
        last_m = months_sorted[-1] if months_sorted else "-"

    k1.metric("고정지출 비중",       f"{fixed_ratio}%")
    k2.metric("월평균 고정지출",     f"{monthly_fixed_avg:,}원")
    k3.metric("구독 항목 수",        f"{sub_count}개")
    k4.metric("전월 대비 증감",      f"{mom_fixed:+.1f}%",
              delta=f"{mom_fixed:+.1f}%",
              delta_color="inverse")  # 증가=빨강, 감소=초록

    st.markdown("---")

    # ── 차트 2열 ─────────────────────────────────────────────
    left, right = st.columns(2)

    with left:
        # 고정 vs 변동 파이차트
        st.markdown("#### 고정 vs 변동 지출 비중")
        pie_data = pd.DataFrame({
            "구분":   ["고정지출", "변동지출"],
            "금액":   [int(total_fixed), int(total_all - total_fixed)],
        })
        fig_pie = px.pie(
            pie_data, names="구분", values="금액",
            color="구분",
            color_discrete_map={"고정지출": "#EF4444", "변동지출": "#60A5FA"},
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig_pie, use_container_width=True)

    with right:
        # 카테고리별 고정지출 바차트
        st.markdown("#### 고정지출 카테고리별 금액")
        cat_fixed = (
            fixed_df.groupby("category")["amount"]
            .sum().reset_index()
            .sort_values("amount", ascending=False)
        )
        fig_cat = px.bar(
            cat_fixed, x="category", y="amount",
            text="amount",
            labels={"amount": "금액 (원)", "category": "카테고리"},
        )
        fig_cat.update_traces(texttemplate="%{text:,.0f}원", textposition="outside")
        fig_cat.update_layout(yaxis_tickformat=",", xaxis_title=None)
        st.plotly_chart(fig_cat, use_container_width=True)

    # ── 월별 고정지출 추이 ────────────────────────────────────
    st.markdown("#### 월별 고정지출 추이")
    monthly_fixed = (
        fixed_df.groupby("year_month")["amount"]
        .sum().reset_index()
        .sort_values("year_month")
    )
    fig_line = px.line(
        monthly_fixed, x="year_month", y="amount",
        markers=True,
        labels={"amount": "고정지출 (원)", "year_month": "월"},
    )
    fig_line.update_layout(yaxis_tickformat=",", xaxis_title=None)
    st.plotly_chart(fig_line, use_container_width=True)

    # ── 구독 항목 상세 테이블 ─────────────────────────────────
    st.markdown("#### 구독 항목 상세")
    sub_df = fixed_df[fixed_df["category"] == "구독"].copy()

    if sub_df.empty:
        st.info("구독 항목(category='구독')이 없습니다.")
    else:
        # 항목별 월평균 + 결제일 추출
        sub_summary = (
            sub_df.groupby("description")
            .agg(
                월평균금액=("amount", "mean"),
                결제건수=("amount", "count"),
                최근결제일=("date", "max"),
            )
            .reset_index()
        )
        sub_summary["월평균금액"] = sub_summary["월평균금액"].astype(int)
        sub_summary["결제일(일)"] = sub_df.groupby("description")["date"].apply(
            lambda x: int(x.dt.day.mode()[0])
        ).values
        sub_summary["최근결제일"] = sub_summary["최근결제일"].dt.strftime("%Y-%m-%d")
        sub_summary.columns = ["구독 서비스", "월평균금액(원)", "결제건수", "최근결제일", "결제일(일)"]

        st.dataframe(
            sub_summary.sort_values("월평균금액(원)", ascending=False)
            .reset_index(drop=True),
            use_container_width=True,
        )

        # 구독 합계 강조
        total_sub = sub_summary["월평균금액(원)"].sum()
        st.warning(f"📌 구독 서비스 {len(sub_summary)}개 · 월 합계 약 **{total_sub:,}원**")

    # ── 고정비 전체 상세 테이블 ───────────────────────────────
    st.markdown("#### 고정지출 전체 내역")
    show_cols = ["date", "category", "description", "amount", "memo"]
    show_cols = [c for c in show_cols if c in fixed_df.columns]
    st.dataframe(
        fixed_df[show_cols]
        .sort_values("date", ascending=False)
        .reset_index(drop=True)
        .head(50),
        use_container_width=True,
    )
    
with tab2:
    render_tab_g2(filtered)
    
# =========================================================
# TAB 3: 패턴 분석 (G1)
# =========================================================
def _build_sub_insights(filtered: pd.DataFrame) -> list:
    insights = []
    total = filtered["amount"].sum()
    if total == 0:
        return insights

    # 1) 카테고리 비중 30% 이상
    cat_share = filtered.groupby("category")["amount"].sum() / total * 100
    for cat, share in cat_share.sort_values(ascending=False).items():
        if share >= 30:
            insights.append(
                f"💡 **{cat}** 지출이 전체의 **{share:.1f}%**를 차지합니다."
            )

    # 2) 카테고리 내 서브카테고리 쏠림 70% 이상
    for cat in filtered["category"].unique():
        cat_df    = filtered[filtered["category"] == cat]
        cat_total = cat_df["amount"].sum()
        if cat_total == 0:
            continue
        sub_share = cat_df.groupby("sub_category")["amount"].sum() / cat_total * 100
        top_sub   = sub_share.idxmax()
        top_share = sub_share.max()
        if top_share >= 70:
            insights.append(
                f"💡 **{cat}** 지출의 **{top_share:.1f}%**가 "
                f"**{top_sub}**에 집중되어 있습니다."
            )

    return insights[:5]  # 최대 5개

def render_tab2(filtered: pd.DataFrame):
    st.subheader("소비 패턴 분석")

    if filtered.empty:
        st.warning("선택한 조건에서 데이터가 없습니다.")
        return
    
    # ── [신규] 서브카테고리 드릴다운 ──────────────────────────
    st.markdown("#### 카테고리 → 서브카테고리 구성")

    col_sel, _ = st.columns([1, 3])
    with col_sel:
        all_cats     = sorted(filtered["category"].unique())
        selected_cat = st.selectbox("대분류 선택", ["전체"] + all_cats, key="sub_drilldown")

    sub_df = filtered if selected_cat == "전체" else filtered[filtered["category"] == selected_cat]

    col_tree, col_bar = st.columns(2)

    with col_tree:
        tree_df  = sub_df.groupby(["category", "sub_category"])["amount"].sum().reset_index()
        fig_tree = px.treemap(
            tree_df,
            path=["category", "sub_category"],
            values="amount",
            color="amount",
            color_continuous_scale="Blues",
            title="카테고리 → 서브카테고리"
        )
        fig_tree.update_traces(textinfo="label+percent root")
        fig_tree.update_layout(coloraxis_showscale=False, margin=dict(t=40,l=0,r=0,b=0))
        st.plotly_chart(fig_tree, use_container_width=True)

    with col_bar:
        sel_sub = (
            sub_df.groupby("sub_category")["amount"]
            .sum().sort_values(ascending=False)
            .head(8).reset_index()
        )
        sel_sub.columns  = ["서브카테고리", "지출금액"]
        sel_sub["비중(%)"] = (sel_sub["지출금액"] / sel_sub["지출금액"].sum() * 100).round(1)

        fig_sub = px.bar(
            sel_sub, x="지출금액", y="서브카테고리",
            orientation="h", text="비중(%)",
            color="비중(%)", color_continuous_scale="Oranges",
            title=f"{selected_cat} 서브카테고리 비중"
        )
        fig_sub.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_sub.update_layout(
            yaxis={"categoryorder": "total ascending"},
            xaxis_tickformat=",", yaxis_title=None,
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_sub, use_container_width=True)

    # 서브카테고리 월별 추이 (Top5)
    st.markdown("##### 서브카테고리 월별 추이 (Top5)")
    top5_sub    = sel_sub["서브카테고리"].head(5).tolist()
    sub_monthly = (
        sub_df[sub_df["sub_category"].isin(top5_sub)]
        .groupby(["year_month", "sub_category"])["amount"]
        .sum().reset_index()
    )
    if not sub_monthly.empty:
        fig_line = px.line(
            sub_monthly, x="year_month", y="amount",
            color="sub_category", markers=True,
            labels={"amount": "지출금액", "year_month": "월", "sub_category": "서브카테고리"}
        )
        fig_line.update_layout(yaxis_tickformat=",", xaxis_title=None)
        st.plotly_chart(fig_line, use_container_width=True)

    # ── [신규] 내 소비 특징 인사이트 ──────────────────────────
    st.markdown("---")
    st.markdown("#### 📌 내 소비 특징")

    insights = _build_sub_insights(filtered)
    if not insights:
        st.info("특이한 소비 패턴이 감지되지 않았습니다.")
    else:
        for item in insights:
            st.info(item)

    # 2-1. 전월 대비 카테고리 증감률 바차트
    st.markdown("#### 전월 대비 카테고리 증감률")
    monthly_cat = (
        filtered.groupby(["year_month", "category"])["amount"]
        .sum().reset_index()
    )
    months_sorted = sorted(monthly_cat["year_month"].unique())

    if len(months_sorted) >= 2:
        last_m = months_sorted[-1]
        prev_m = months_sorted[-2]
        last_df = monthly_cat[monthly_cat["year_month"] == last_m].set_index("category")["amount"]
        prev_df = monthly_cat[monthly_cat["year_month"] == prev_m].set_index("category")["amount"]
        all_cats = last_df.index.union(prev_df.index)
        chg = []
        for cat in all_cats:
            l = last_df.get(cat, 0)
            p = prev_df.get(cat, 0)
            rate = round((l - p) / p * 100, 1) if p > 0 else 0.0
            chg.append({"category": cat, "change_rate": rate})
        chg_df = pd.DataFrame(chg).sort_values("change_rate", ascending=False)
        chg_df["color"] = chg_df["change_rate"].apply(
            lambda x: "증가" if x > 0 else ("감소" if x < 0 else "유지")
        )
        fig_chg = px.bar(
            chg_df, x="category", y="change_rate",
            color="color",
            color_discrete_map={"증가": "#EF4444", "감소": "#22C55E", "유지": "#9CA3AF"},
            labels={"change_rate": "증감률 (%)", "category": "카테고리"},
        )
        fig_chg.update_layout(showlegend=True, xaxis_title=None)
        st.caption(f"{prev_m} → {last_m} 비교")
        st.plotly_chart(fig_chg, use_container_width=True)
    else:
        st.info("전월 대비 비교를 위해 2개월 이상 데이터가 필요합니다.")

    # 2-2. 계절별 카테고리 히트맵
    st.markdown("#### 계절별 카테고리 지출 히트맵")
    season_order = ["봄", "여름", "가을", "겨울"]
    season_cat = (
        filtered.groupby(["season", "category"])["amount"]
        .mean().reset_index()
    )
    pivot_season = season_cat.pivot(index="season", columns="category", values="amount").fillna(0)
    pivot_season = pivot_season.reindex([s for s in season_order if s in pivot_season.index])

    if not pivot_season.empty:
        fig_s = px.imshow(
            pivot_season, aspect="auto",
            labels=dict(x="카테고리", y="계절", color="월평균 지출"),
            color_continuous_scale="YlOrRd",
        )
        st.plotly_chart(fig_s, use_container_width=True)

    # 2-3. 요일×월 히트맵 (최근 6개월)
    st.markdown("#### 요일 x 월 지출 패턴 (최근 6개월)")
    recent_months = sorted(filtered["year_month"].astype(str).unique())[-6:]
    heat_df = filtered[filtered["year_month"].isin(recent_months)].copy()
    pivot = heat_df.pivot_table(
        index="weekday_kr", columns="year_month",
        values="amount", aggfunc="sum", fill_value=0
    )
    weekday_order = ["월", "화", "수", "목", "금", "토", "일"]
    pivot = pivot.reindex(weekday_order).reindex(columns=recent_months)

    if not pivot.empty:
        fig_hm = px.imshow(
            pivot, aspect="auto",
            labels=dict(x="월", y="요일", color="지출금액"),
        )
        st.plotly_chart(fig_hm, use_container_width=True)

    # 2-4. 서브카테고리 월평균 Top 10 가로바
    st.markdown("#### 서브카테고리 월평균 지출 Top 10")
    subcat_monthly = (
        filtered.groupby(["year_month", "sub_category"])["amount"]
        .sum().reset_index()
        .groupby("sub_category")["amount"].mean()
        .reset_index().rename(columns={"amount": "monthly_avg"})
        .sort_values("monthly_avg", ascending=True)
        .tail(10)
    )
    if not subcat_monthly.empty:
        fig_sc = px.bar(
            subcat_monthly, x="monthly_avg", y="sub_category",
            orientation="h",
            labels={"monthly_avg": "월평균 지출 (원)", "sub_category": "서브카테고리"},
            text="monthly_avg",
        )
        fig_sc.update_traces(texttemplate="%{text:,.0f}원", textposition="outside")
        fig_sc.update_layout(xaxis_tickformat=",", yaxis_title=None)
        st.plotly_chart(fig_sc, use_container_width=True)
        
# ── G1: 전년 대비 연간 비교 ──────────────────────────────
    st.markdown("---")
    st.markdown("#### 전년 대비 연간 지출 비교 (2024 vs 2025)")

    years = sorted(filtered["date"].dt.year.unique())

    if len(years) < 2:
        st.info("전년 대비 비교를 위해 2개 연도 이상 데이터가 필요합니다.")
        return
    
    # ✅ 연도 컬럼 미리 생성
    df_yoy = filtered.copy()
    df_yoy["year"] = df_yoy["date"].dt.year
    
    # 연도별 총지출
    yearly_total = (
        df_yoy.groupby("year")["amount"]
        .sum().reset_index()
    )
    yearly_total.columns = ["연도", "총지출"]
    yearly_total["연도"] = yearly_total["연도"].astype(str)

    # 전년 대비 변화율 계산
    if len(yearly_total) >= 2:
        y_last = yearly_total["총지출"].iloc[-1]
        y_prev = yearly_total["총지출"].iloc[-2]
        yoy_rate = round((y_last - y_prev) / y_prev * 100, 1) if y_prev > 0 else 0

        col_y1, col_y2, col_y3 = st.columns(3)
        col_y1.metric(f"{yearly_total['연도'].iloc[-2]} 총지출",
                      f"{int(y_prev):,}원")
        col_y2.metric(f"{yearly_total['연도'].iloc[-1]} 총지출",
                      f"{int(y_last):,}원")
        col_y3.metric("전년 대비 변화율",
                      f"{yoy_rate:+.1f}%",
                      delta=f"{yoy_rate:+.1f}%",
                      delta_color="inverse")

    # 연도별 카테고리 비교 바차트
    st.markdown("##### 연도별 카테고리 지출 비교")
    yearly_cat = (
        df_yoy.groupby(["year", "category"])["amount"]   # ✅ "year" 컬럼 사용
        .sum().reset_index()
    )
    yearly_cat.columns = ["연도", "category", "amount"]
    yearly_cat["연도"] = yearly_cat["연도"].astype(str)

    fig_yoy = px.bar(
        yearly_cat,
        x="category", y="amount",
        color="연도",
        barmode="group",
        text="amount",
        labels={"amount": "지출 금액 (원)", "category": "카테고리"},
        color_discrete_map={
            str(years[0]): "#60A5FA",
            str(years[1]): "#F97316",
        },
    )
    fig_yoy.update_traces(texttemplate="%{text:,.0f}원", textposition="outside")
    fig_yoy.update_layout(yaxis_tickformat=",", xaxis_title=None)
    st.plotly_chart(fig_yoy, use_container_width=True)

    # 계절별 전년 대비 편차
    st.markdown("##### 계절별 전년 대비 지출 편차")
    season_year = (
        df_yoy.groupby(["year", "season"])["amount"]
        .sum().reset_index()
    )
    season_year.columns = ["연도", "season", "amount"]

    season_pivot = season_year.pivot(
        index="season", columns="연도", values="amount"
    ).fillna(0)

    if len(season_pivot.columns) >= 2:
        prev_y, last_y = season_pivot.columns[0], season_pivot.columns[-1]
        season_pivot["변화율(%)"] = (
            (season_pivot[last_y] - season_pivot[prev_y])
            / season_pivot[prev_y] * 100
        ).round(1)
        season_pivot["판정"] = season_pivot["변화율(%)"].apply(
            lambda x: "증가" if x > 0 else "감소"
        )
        season_order = ["봄", "여름", "가을", "겨울"]
        season_pivot = season_pivot.reindex(
            [s for s in season_order if s in season_pivot.index]
        ).reset_index()

        fig_season = px.bar(
            season_pivot,
            x="season", y="변화율(%)",
            color="판정",
            color_discrete_map={"증가": "#EF4444", "감소": "#22C55E"},
            text="변화율(%)",
            labels={"season": "계절"},
        )
        fig_season.update_traces(
            texttemplate="%{text:+.1f}%", textposition="outside"
        )
        fig_season.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_season.update_layout(xaxis_title=None, showlegend=True)
        st.plotly_chart(fig_season, use_container_width=True)

    # 연도별 월 추이 비교 라인차트
    df_yoy["month"] = df_yoy["date"].dt.month   # ← 반드시 groupby 전에
    st.markdown("##### 월별 지출 추이 연도 비교")
    monthly_year = (
        df_yoy.groupby(["year", "month"])["amount"]
        .sum().reset_index()
    )
    monthly_year.columns = ["연도", "월", "amount"]
    monthly_year["연도"] = monthly_year["연도"].astype(str)

    fig_monthly_yoy = px.line(
        monthly_year,
        x="월", y="amount",
        color="연도",
        markers=True,
        labels={"amount": "지출 금액 (원)", "월": "월"},
        color_discrete_map={
            str(years[0]): "#60A5FA",
            str(years[1]): "#F97316",
        },
    )
    fig_monthly_yoy.update_layout(
        yaxis_tickformat=",",
        xaxis={"tickvals": list(range(1, 13)),
               "ticktext": [f"{m}월" for m in range(1, 13)]},
    )
    st.plotly_chart(fig_monthly_yoy, use_container_width=True)
# render_tab2 맨 끝에 추가
    st.markdown("---")
    st.markdown("#### 🤖 AI 패턴 분석 (F006)")

    available_months = sorted(filtered["year_month"].unique())
    selected_month = st.selectbox(
        "분석할 월 선택",
        options=available_months[::-1],
        index=0,
        key="f006_month_select",
    )

    monthly_kpi = build_monthly_kpi(filtered, selected_month)

    if monthly_kpi and st.button("📊 패턴 분석 실행", type="primary", key="f006_btn"):
        with st.spinner("분석 중..."):
            try:
                result = f006_pattern_analysis(monthly_kpi)
                st.session_state["f006_result"] = result
                st.session_state["f006_month"] = selected_month
            except Exception as e:
                st.error(f"F006 오류: {e}")

    if "f006_result" in st.session_state:
        with st.expander(
            f"📊 분석 결과 ({st.session_state.get('f006_month', '')})",
            expanded=False,
        ):
            st.markdown(st.session_state["f006_result"])
        st.info("📌 월간 리포트 탭에서 F008 리포트로 내보낼 수 있습니다.")
# ── 개인 소비 스타일 분석 ← 여기 추가 (맨 끝)
    st.markdown("---")
    render_personal_insights(filtered)


with tab3:
    render_tab2(filtered)


# =========================================================
# TAB 4: 코호트 비교 (소득 밴드 기반)
# =========================================================

def render_tab3(filtered: pd.DataFrame):
    st.subheader("소득 밴드별 코호트 비교 분석")
    st.caption(
        "소득 밴드를 선택하면 유사 소득 집단 80명 코호트와 내 지출을 비교합니다. "
        "(SEED=2024 고정 가상 코호트 기준)"
    )

    if filtered.empty:
        st.warning("선택한 조건에서 데이터가 없습니다.")
        return

    # ── 상단 컨트롤 ──────────────────────────────────────────────────
    col_band, col_excl, col_sol = st.columns([2, 1.5, 1])

    with col_band:
        band = st.selectbox(
            "소득 밴드 선택 (월 가구소득 기준)",
            options=INCOME_BANDS,
            index=1,                      # 기본: 400~550만
            help="본인 월 가구소득에 가장 가까운 구간을 선택하세요.",
        )

    with col_excl:
        exclude_event = st.checkbox(
            "이벤트성 지출 제외",
            value=False,
            help="해외여행·출장·전자기기 구매 등 이벤트성 고액 지출을 분석에서 제외합니다.\n"
                 "이벤트 달 포함 시 월 총지출이 왜곡될 수 있어 비교 정확도가 높아집니다.",
        )

    with col_sol:
        show_insight = st.checkbox("절감 인사이트 표시", value=True)

    # ── 코호트 로드 + 분석 ──────────────────────────────────────────
    cohort_all = _load_cohort()
    result = run_band_cohort_analysis(
        filtered,
        band=band,
        cohort_all=cohort_all,
        top_n=5,
        exclude_event=exclude_event,
    )

    pct   = result["percentile"]
    lift  = result["lift_df"]
    top5  = result["top_subcat_df"]

    # ── 소득 밴드 불일치 경고 ─────────────────────────────────────────
    user_monthly = pct["user_value"]
    band_center  = {"300~400만": 3_500_000, "400~550만": 4_750_000,
                    "550~700만": 6_250_000, "700만+": 8_500_000}[band]
    ratio = user_monthly / band_center if band_center else 1.0
    if ratio > 1.5:
        st.error(
            f"내 월평균 지출({user_monthly:,.0f}원)이 선택 소득대 중심값 대비 "
            f"{ratio:.1f}배로 높습니다. 더 높은 소득 밴드를 선택하거나 "
            f"이벤트성 지출 제외를 체크해주세요."
        )
    elif ratio < 0.4:
        st.warning(
            f"내 월평균 지출({user_monthly:,.0f}원)이 선택 소득대 중심값 대비 "
            f"현저히 낮습니다. 더 낮은 소득 밴드를 선택해보세요."
        )

    # ── KPI 카드 ─────────────────────────────────────────────────────
    st.markdown("---")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("내 월평균",         f"{pct['user_value']:,.0f}원")
    k2.metric(f"코호트 평균\n({band})", f"{pct['cohort_mean']:,.0f}원")
    k3.metric("지출 순위",         f"상위 {pct['percentile_rank']:.1f}%",
              help="퍼센타일이 낮을수록 절약형입니다.")
    k4.metric("코호트 중앙값",     f"{pct['cohort_median']:,.0f}원")

    st.markdown("---")

    # ── 절감 인사이트 ─────────────────────────────────────────────────
    if show_insight:
        overspend = lift[lift["label"] == "과소비"]
        if not overspend.empty:
            lines = []
            for _, row in overspend.iterrows():
                diff = int(row["user_avg"] - row["cohort_avg"])
                lines.append(
                    f"- **{row['category']}**: 코호트 대비 {row['lift']:.2f}배 "
                    f"(월 +{diff:,}원 초과)"
                )
            st.warning("과소비 카테고리 발견\n" + "\n".join(lines))
        else:
            st.success("선택 소득대 대비 모든 카테고리 지출이 양호합니다.")

    # ── 차트 2열 ─────────────────────────────────────────────────────
    left, right = st.columns(2)

    with left:
        st.markdown("#### 카테고리별 Lift (코호트 대비 배수)")
        color_map = {"과소비": "#EF4444", "보통": "#60A5FA", "절약": "#22C55E"}
        fig_lift = px.bar(
            lift.sort_values("lift"),
            x="lift", y="category",
            orientation="h",
            color="label",
            color_discrete_map=color_map,
            labels={"lift": "Lift 배수", "category": "카테고리"},
            text="lift",
        )
        fig_lift.update_traces(texttemplate="%{text:.2f}x", textposition="outside")
        fig_lift.add_vline(x=1.0, line_dash="dash", line_color="gray")
        fig_lift.update_layout(showlegend=True, yaxis_title=None, margin=dict(l=0))
        st.plotly_chart(fig_lift, use_container_width=True)

    with right:
        st.markdown("#### 서브카테고리 Top 5 (월평균)")
        if not top5.empty:
            fig_top5 = px.bar(
                top5.sort_values("monthly_avg"),
                x="monthly_avg", y="sub_category",
                orientation="h",
                labels={"monthly_avg": "월평균 지출 (원)", "sub_category": "서브카테고리"},
                text="monthly_avg",
                color_discrete_sequence=["#818CF8"],
            )
            fig_top5.update_traces(texttemplate="%{text:,.0f}원", textposition="outside")
            fig_top5.update_layout(xaxis_tickformat=",", yaxis_title=None)
            st.plotly_chart(fig_top5, use_container_width=True)

    # ── 월 총지출 분포: 박스플롯 + 내 값 마커 ─────────────────────────
    st.markdown("#### 월 총지출 분포 (소득 밴드 내 내 위치)")

    band_df      = cohort_all[cohort_all["income_band"] == band]
    cohort_totals = band_df["total"].values if not band_df.empty else np.array([0])

    fig_box = go.Figure()
    fig_box.add_trace(go.Box(
        y=cohort_totals,
        name=f"코호트 ({band})",
        marker_color="#60A5FA",
        boxmean="sd",
    ))
    fig_box.add_trace(go.Scatter(
        x=[f"코호트 ({band})"],
        y=[user_monthly],
        mode="markers",
        marker=dict(color="#EF4444", size=14, symbol="star"),
        name="내 월평균",
    ))
    fig_box.update_layout(
        yaxis_title="월 총지출 (원)",
        yaxis_tickformat=",",
        showlegend=True,
        height=380,
    )
    st.plotly_chart(fig_box, use_container_width=True)

    # ── 카테고리별 퍼센타일 히트맵 ────────────────────────────────────
    st.markdown("#### 카테고리별 지출 퍼센타일 (소득 밴드 내 내 위치)")
    from scipy import stats as scipy_stats

    pct_rows = []
    for _, row in lift.iterrows():
        cat = row["category"]
        if cat not in band_df.columns:
            continue
        cat_vals = band_df[cat].values
        user_cat = row["user_avg"]
        pct_val  = scipy_stats.percentileofscore(cat_vals, user_cat, kind="rank")
        pct_rows.append({
            "카테고리":   cat,
            "내 월평균":  f"{int(user_cat):,}원",
            "퍼센타일":   round(100.0 - pct_val, 1),
            "판정":       row["label"],
        })

    if pct_rows:
        pct_df = pd.DataFrame(pct_rows)

        fig_heat = px.bar(
            pct_df.sort_values("퍼센타일", ascending=True),
            x="퍼센타일", y="카테고리",
            orientation="h",
            color="판정",
            color_discrete_map={"과소비": "#EF4444", "보통": "#60A5FA", "절약": "#22C55E"},
            text="퍼센타일",
            range_x=[0, 100],
        )
        fig_heat.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
        fig_heat.add_vline(x=50, line_dash="dot", line_color="gray",
                           annotation_text="중간(50%)", annotation_position="top")
        fig_heat.update_layout(xaxis_title="상위 퍼센타일 (낮을수록 절약)", yaxis_title=None)
        st.plotly_chart(fig_heat, use_container_width=True)

    # ── 상세 비교 테이블 ──────────────────────────────────────────────
    st.markdown("#### 카테고리별 상세 비교")
    display_lift = lift[["category", "user_avg", "cohort_avg", "lift", "label"]].copy()
    display_lift.columns = ["카테고리", "내 월평균(원)", f"코호트 평균(원) [{band}]", "Lift", "판정"]
    st.dataframe(display_lift.reset_index(drop=True), use_container_width=True)


with tab4:
    render_tab3(filtered)


# =========================================================
# TAB 5: 예산 추천 (G3)
# =========================================================

FIXED_CATEGORIES = ["주거/통신", "구독"]


def render_tab4(filtered: pd.DataFrame):
    st.subheader("예산 추천")
    st.caption("최근 3개월 평균을 기반으로 다음 달 권장 예산을 제안합니다.")

    if filtered.empty:
        st.warning("선택한 조건에서 데이터가 없습니다.")
        return

    # 최근 3개월 추출
    months_sorted = sorted(filtered["year_month"].unique())
    recent_3 = months_sorted[-3:]
    df_recent = filtered[filtered["year_month"].isin(recent_3)].copy()

    # 절감 목표 슬라이더
    save_pct = st.slider(
        "절감 목표 (%)",
        min_value=0, max_value=30, value=10, step=5,
        help="고정비(주거/통신, 구독)를 제외한 카테고리에 적용됩니다."
    )

    # 카테고리별 최근 3개월 평균
    cat_avg = (
        df_recent.groupby(["year_month", "category"])["amount"]
        .sum().reset_index()
        .groupby("category")["amount"].mean()
        .reset_index().rename(columns={"amount": "recent_avg"})
    )
    cat_avg["is_fixed_cat"] = cat_avg["category"].isin(FIXED_CATEGORIES)
    cat_avg["target_budget"] = cat_avg.apply(
        lambda r: round(r["recent_avg"])
        if r["is_fixed_cat"]
        else round(r["recent_avg"] * (1 - save_pct / 100) / 100) * 100,
        axis=1
    )
    cat_avg["절감 가능 금액"] = (cat_avg["recent_avg"] - cat_avg["target_budget"]).clip(lower=0).astype(int)

    # KPI: 현재 총예산 vs 권장 총예산
    total_current = int(cat_avg["recent_avg"].sum())
    total_target  = int(cat_avg["target_budget"].sum())
    total_save    = total_current - total_target

    k1, k2, k3 = st.columns(3)
    k1.metric("현재 월평균 지출",   f"{total_current:,}원")
    k2.metric("권장 월 목표 지출",  f"{total_target:,}원")
    k3.metric("예상 절감 금액",     f"{total_save:,}원")

    st.caption(f"최근 3개월 기준: {', '.join(recent_3)}")

    # 그룹드 바차트: 현재 vs 권장
    st.markdown("#### 카테고리별 현재 지출 vs 권장 예산")
    bar_data = pd.concat([
        cat_avg[["category", "recent_avg"]].rename(columns={"recent_avg": "금액"}).assign(구분="현재 지출"),
        cat_avg[["category", "target_budget"]].rename(columns={"target_budget": "금액"}).assign(구분="권장 예산"),
    ])
    fig_budget = px.bar(
        bar_data, x="category", y="금액",
        color="구분", barmode="group",
        color_discrete_map={"현재 지출": "#60A5FA", "권장 예산": "#22C55E"},
        labels={"금액": "금액 (원)", "category": "카테고리"},
        text="금액",
    )
    fig_budget.update_traces(texttemplate="%{text:,.0f}원", textposition="outside")
    fig_budget.update_layout(yaxis_tickformat=",", xaxis_title=None)
    st.plotly_chart(fig_budget, use_container_width=True)

    # 테이블
    st.markdown("#### 카테고리별 예산 상세")
    display_budget = cat_avg[["category", "recent_avg", "target_budget", "절감 가능 금액", "is_fixed_cat"]].copy()
    display_budget.columns = ["카테고리", "현재 월평균(원)", "권장 예산(원)", "절감 가능 금액(원)", "고정비 여부"]
    display_budget["현재 월평균(원)"] = display_budget["현재 월평균(원)"].astype(int)
    st.dataframe(display_budget.reset_index(drop=True), use_container_width=True)
# ── F007: AI 예산 추천 ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🤖 AI 예산 추천 (F007)")
    st.caption("최근 3개월 데이터를 AI가 분석하여 다음 달 맞춤 예산을 제안합니다.")

    months_for_f7 = sorted(filtered["year_month"].unique())
    if months_for_f7:
        f7_month = st.selectbox(
            "기준 월 선택",
            options=months_for_f7[::-1],
            index=0,
            key="f007_month_select",
        )

        if st.button("💰 AI 예산 추천 받기 (F007)", key="f007_btn"):
            with st.spinner("예산 계산 중..."):
                try:
                    result = f007_budget_recommendation(filtered, f7_month)
                    st.session_state["f007_result"] = result
                    st.session_state["f007_month"] = f7_month
                except Exception as e:
                    st.error(f"F007 오류: {e}")

        if "f007_result" in st.session_state:
            with st.expander(
                f"💰 AI 예산 추천 결과 ({st.session_state.get('f007_month', '')})",
                expanded=False
            ):
                st.markdown(st.session_state["f007_result"])
            st.info("📌 개요 탭에서 F006 실행 후 F008 리포트에 자동 포함됩니다.")
# ── G3: 목표 달성 여부 ────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 이번 달 목표 달성 여부")

    # 분석 기준 월 선택
    months_sorted_g3 = sorted(filtered["year_month"].unique())
    target_month_g3 = st.selectbox(
        "달성 여부 확인할 월",
        options=months_sorted_g3[::-1],
        index=0,
        key="g3_month_select",
    )

    this_month_df = filtered[filtered["year_month"] == target_month_g3]
    this_month_total = int(this_month_df["amount"].sum())

    # 카테고리별 실제 지출
    actual_by_cat = (
        this_month_df.groupby("category")["amount"]
        .sum().reset_index()
        .rename(columns={"amount": "actual"})
    )

    # cat_avg(권장예산)와 합치기
    goal_df = pd.merge(
        cat_avg[["category", "target_budget"]],
        actual_by_cat,
        on="category",
        how="left",
    ).fillna({"actual": 0})

    goal_df["actual"]        = goal_df["actual"].astype(int)
    goal_df["target_budget"] = goal_df["target_budget"].astype(int)
    goal_df["달성률(%)"]     = (
        goal_df["actual"] / goal_df["target_budget"] * 100
    ).round(1).clip(upper=200)  # 200% 상한
    goal_df["초과여부"] = goal_df["달성률(%)"].apply(
        lambda x: "초과" if x > 100 else "달성"
    )

    # KPI
    total_target  = int(goal_df["target_budget"].sum())
    achieve_rate  = round(this_month_total / total_target * 100, 1) if total_target > 0 else 0
    over_cat_count = int((goal_df["달성률(%)"] > 100).sum())

    g1, g2, g3 = st.columns(3)
    g1.metric("이번 달 실제 지출",  f"{this_month_total:,}원")
    g2.metric("권장 목표 예산",     f"{total_target:,}원")
    g3.metric(
        "목표 달성률",
        f"{achieve_rate:.1f}%",
        delta=f"초과 카테고리 {over_cat_count}개",
        delta_color="inverse",
    )

    # 게이지 차트 (목표 대비 현재)
    st.markdown("##### 전체 목표 대비 실제 지출 게이지")
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=achieve_rate,
        delta={"reference": 100, "increasing": {"color": "#EF4444"},
               "decreasing": {"color": "#22C55E"}},
        gauge={
            "axis": {"range": [0, 150], "tickwidth": 1},
            "bar":  {"color": "#EF4444" if achieve_rate > 100 else "#22C55E"},
            "steps": [
                {"range": [0,   80],  "color": "#DCFCE7"},
                {"range": [80,  100], "color": "#FEF9C3"},
                {"range": [100, 150], "color": "#FEE2E2"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 3},
                "thickness": 0.75,
                "value": 100,
            },
        },
        title={"text": "목표 달성률 (%) / 100% 초과 = 예산 초과"},
        number={"suffix": "%"},
    ))
    fig_gauge.update_layout(height=280)
    st.plotly_chart(fig_gauge, use_container_width=True)

    # 카테고리별 달성률 바차트
    st.markdown("##### 카테고리별 목표 달성률")
    fig_goal = px.bar(
        goal_df.sort_values("달성률(%)"),
        x="달성률(%)", y="category",
        orientation="h",
        color="초과여부",
        color_discrete_map={"초과": "#EF4444", "달성": "#22C55E"},
        text="달성률(%)",
        labels={"category": "카테고리"},
    )
    fig_goal.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_goal.add_vline(x=100, line_dash="dash", line_color="gray",
                       annotation_text="목표(100%)")
    fig_goal.update_layout(xaxis_range=[0, 200], yaxis_title=None)
    st.plotly_chart(fig_goal, use_container_width=True)

    # 상세 테이블
    st.markdown("##### 카테고리별 상세")
    display_goal = goal_df[["category", "target_budget", "actual", "달성률(%)", "초과여부"]].copy()
    display_goal.columns = ["카테고리", "권장예산(원)", "실제지출(원)", "달성률(%)", "판정"]
    st.dataframe(display_goal.reset_index(drop=True), use_container_width=True)            



with tab5:
    render_tab4(filtered)


# =========================================================
# TAB 6: 이상치 탐지 (G4)
# =========================================================

def render_tab5(filtered: pd.DataFrame):
    st.subheader("이상치 탐지")

    if filtered.empty:
        st.warning("선택한 조건에서 데이터가 없습니다.")
        return

    # --- (1) 월별 총지출 이상 탐지 ---
    st.markdown("#### 월별 총지출 이상 탐지 (평균 ±2σ)")
    monthly_total = filtered.groupby("year_month")["amount"].sum().reset_index()
    monthly_total.columns = ["year_month", "total"]
    mu  = monthly_total["total"].mean()
    sig = monthly_total["total"].std()

    def anomaly_label(v):
        if v > mu + 2 * sig: return "급증"
        if v < mu - 2 * sig: return "급감"
        return "정상"

    monthly_total["anomaly"] = monthly_total["total"].apply(anomaly_label)
    color_map_ano = {"급증": "#EF4444", "급감": "#3B82F6", "정상": "#D1D5DB"}

    fig_ano = px.bar(
        monthly_total, x="year_month", y="total",
        color="anomaly",
        color_discrete_map=color_map_ano,
        labels={"total": "월 총지출 (원)", "year_month": "월"},
        text="total",
    )
    fig_ano.update_traces(texttemplate="%{text:,.0f}원", textposition="outside")
    fig_ano.update_layout(yaxis_tickformat=",", xaxis_title=None)
    fig_ano.add_hline(y=mu + 2 * sig, line_dash="dash", line_color="#EF4444",
                      annotation_text="+2σ", annotation_position="top left")
    fig_ano.add_hline(y=max(0, mu - 2 * sig), line_dash="dash", line_color="#3B82F6",
                      annotation_text="-2σ", annotation_position="bottom left")
    st.plotly_chart(fig_ano, use_container_width=True)

    # --- (2) 카테고리 전월 대비 급증 탐지 ---
    st.markdown("#### 카테고리 전월 대비 급증 탐지 (기준: +30% 이상)")
    THRESHOLD_PCT = 30.0
    months_sorted = sorted(filtered["year_month"].unique())
    spike_rows = []
    for i in range(1, len(months_sorted)):
        prev_m, last_m = months_sorted[i-1], months_sorted[i]
        prev_sum = filtered[filtered["year_month"] == prev_m].groupby("category")["amount"].sum()
        last_sum = filtered[filtered["year_month"] == last_m].groupby("category")["amount"].sum()
        for cat in last_sum.index:
            p = prev_sum.get(cat, 0)
            l = last_sum[cat]
            if p > 0:
                rate = (l - p) / p * 100
                if rate >= THRESHOLD_PCT:
                    spike_rows.append({
                        "month": last_m, "category": cat,
                        "prev_amt": int(p), "last_amt": int(l),
                        "change_rate": round(rate, 1),
                    })

    if spike_rows:
        spike_df = pd.DataFrame(spike_rows)
        fig_bubble = px.scatter(
            spike_df, x="month", y="category",
            size="change_rate", color="change_rate",
            color_continuous_scale="Reds",
            labels={"change_rate": "증감률(%)", "month": "월", "category": "카테고리"},
            text="change_rate",
        )
        fig_bubble.update_traces(texttemplate="+%{text:.1f}%", textposition="top center")
        st.plotly_chart(fig_bubble, use_container_width=True)
    else:
        st.success(f"전월 대비 +{THRESHOLD_PCT:.0f}% 초과 카테고리가 없습니다.")

    # --- (3) 단건 고액 거래 탐지 ---
    st.markdown("#### 단건 고액 거래 탐지 (평균 + 2σ 초과)")
    amt_mean = filtered["amount"].mean()
    amt_std  = filtered["amount"].std()
    threshold_tx = amt_mean + 2 * amt_std
    big_tx = filtered[filtered["amount"] > threshold_tx].sort_values("amount", ascending=False)

    if not big_tx.empty:
        top3 = big_tx.head(3)
        cols = st.columns(len(top3))
        for i, (_, row) in enumerate(top3.iterrows()):
            cols[i].metric(
                label=f"{row['date'].strftime('%Y-%m-%d')} | {row['category']}",
                value=f"{row['amount']:,}원",
                help=str(row.get("description", "")),
            )

        show_cols = ["date", "category", "description", "amount", "memo"]
        show_cols = [c for c in show_cols if c in big_tx.columns]
        st.dataframe(
            big_tx[show_cols].reset_index(drop=True).head(20),
            use_container_width=True
        )
    else:
        st.success("고액 이상 거래가 감지되지 않았습니다.")

    # --- (4) 월별 최대 단건 지출 추이 ---
    st.markdown("#### 월별 최대 단건 지출 추이")
    monthly_max = filtered.groupby("year_month")["amount"].max().reset_index()
    monthly_max.columns = ["year_month", "max_amount"]
    fig_max = px.bar(
        monthly_max, x="year_month", y="max_amount",
        labels={"max_amount": "최대 지출 (원)", "year_month": "월"},
        text="max_amount",
    )
    fig_max.update_traces(texttemplate="%{text:,.0f}원", textposition="outside")
    fig_max.update_layout(yaxis_tickformat=",", xaxis_title=None)
    st.plotly_chart(fig_max, use_container_width=True)


with tab6:
    render_tab5(filtered)
    
# =========================================================
# TAB 7: 월간 리포트
# =========================================================
def render_tab_report(filtered: pd.DataFrame):
    st.subheader("월간 리포트 (F008)")
    st.caption("패턴 분석(F006)과 예산 추천(F007)을 먼저 실행하면 리포트가 완성됩니다.")

    available_months = sorted(filtered["year_month"].unique())
    if not available_months:
        st.warning("데이터가 없습니다.")
        return

    selected_month = st.selectbox(
        "리포트 대상 월",
        options=available_months[::-1],
        index=0,
        key="report_month_select",
    )

    monthly_kpi = build_monthly_kpi(filtered, selected_month)

    if monthly_kpi:
        m1, m2, m3 = st.columns(3)
        m1.metric(
            f"{selected_month} 총지출",
            f"{monthly_kpi['total_spend']:,}원",
            delta=f"{monthly_kpi['mom_rate']:+.1f}%"
                  if monthly_kpi["mom_rate"] is not None else "전월 없음",
        )
        m2.metric("거래 건수", f"{monthly_kpi['tx_count']}건")
        m3.metric("최대 단건", f"{monthly_kpi['max_single']:,}원")

    # F006/F007 완료 여부 표시
    st.markdown("---")
    st.markdown("#### 리포트 구성 상태")

    f006_done = "f006_result" in st.session_state
    f007_done = "f007_result" in st.session_state

    c1, c2 = st.columns(2)
    # ✅ 일반 if/else 블록으로 변경
    if f006_done:
        c1.success("✅ F006 패턴 분석 완료")
    else:
        c1.warning("⏳ F006 미완료 → 패턴 분석 탭에서 실행")

    if f007_done:
        c2.success("✅ F007 예산 추천 완료")
    else:
        c2.warning("⏳ F007 미완료 → 예산 추천 탭에서 실행")

    if st.button(
        "📄 월간 리포트 생성",
        disabled=not (f006_done and monthly_kpi),
        type="primary",
        help="F006은 필수, F007은 선택입니다.",
    ):
        budget_text = st.session_state.get(
            "f007_result",
            "예산 추천 탭에서 F007을 실행하면 여기에 포함됩니다."
        )
        report = f008_build_report(
            monthly_kpi,
            st.session_state["f006_result"],
            budget_text,
        )
        st.session_state["f008_result"] = report

    if "f008_result" in st.session_state:
        with st.expander("📄 리포트 미리보기", expanded=False):
            st.markdown(st.session_state["f008_result"])

        st.download_button(
            label="📥 리포트 다운로드 (.md)",
            data=st.session_state["f008_result"],
            file_name=f"report_{selected_month}.md",
            mime="text/markdown",
            key="f008_download",
        )
with tab7: render_tab_report(filtered)