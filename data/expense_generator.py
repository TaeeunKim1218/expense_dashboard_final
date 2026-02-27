# expense_generator.py
"""
개인 지출 시뮬레이터 (고도화 버전)
- SEED: 환경변수 EXPENSE_SEED 또는 인자로 주입 가능 (재현성 + 다양성)
- Employment Status: salaryman(직장인) / homemaker(전업주부)
- affluent 타입: A1(투자형)/A2(주담대형)/A3(라이프스타일형) 혼합
- 전자기기: PRICE_TABLE 기반 금액 + 기기군별 쿨다운(2년 데이터 내 중복 방지)
- 숙박: 일상 문화/여가 풀에서 완전 제거 → 여행/출장 시나리오에서만 생성
- 월 소프트캡 + 하드캡 normalize 함수로 폭주 방지
- 고정비 dict 구조: 만성질환 / 요양원 / 자영업 추가
- 해외여행: 항공권 사전 구매(30~90일 전) + 가족 인원 배수
- 출장: 직장인 전용, 법인카드 확률 포함
- build_grocery_schedule(): 주 단위 장보기 날짜 사전 확정
- cohort_engine.py의 LOGNORM_PARAMS와 동일한 파라미터 유지
"""

import os
import numpy as np
import pandas as pd
import json

# =========================================================
# 0) 기본 설정
# =========================================================

# SEED: 환경변수 EXPENSE_SEED > 기본값 42
SEED = int(os.environ.get("EXPENSE_SEED", "42"))
rng  = np.random.default_rng(SEED)

START_DATE = "2024-01-01"
END_DATE   = "2025-12-31"
dates      = pd.date_range(START_DATE, END_DATE, freq="D")

DEFAULT_PAYMENT  = "카드"
TRANSFER_PAYMENT = "계좌이체"
CORPORATE_CARD   = "법인카드"


# =========================================================
# 0-1) 유틸 함수
# =========================================================

def round_to_100(x: float) -> int:
    return int(round(x / 100) * 100)

def clamp01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))

def choose_start_month_with_seasonality() -> int:
    """연초(1~3월) 시작 확률이 높은 월 선택."""
    weights = np.array([2.2, 1.8, 1.5] + [1.0] * 9, dtype=float)
    weights /= weights.sum()
    return int(rng.choice(np.arange(1, 13), p=weights))

def month_range(start_month: int, length: int) -> list[int]:
    if start_month > 12:
        return []
    return list(range(start_month, min(12, start_month + length - 1) + 1))

def build_monthly_like_schedule(year: int) -> set:
    """매달 1회, 월초·월중·월말 근처 하루."""
    out = set()
    for m in range(1, 13):
        day_candidates = np.array([3, 4, 5, 12, 13, 14, 23, 24, 25], dtype=int)
        d = int(rng.choice(day_candidates))
        try:
            out.add(pd.Timestamp(year=year, month=m, day=d))
        except ValueError:
            pass
    return out

def build_bimonthly_schedule(year: int) -> set:
    """2달당 최대 1회(홀수월 대상)."""
    out = set()
    for m in [1, 3, 5, 7, 9, 11]:
        day_candidates = np.array([5, 8, 12, 15, 18, 22, 25], dtype=int)
        d = int(rng.choice(day_candidates))
        try:
            out.add(pd.Timestamp(year=year, month=m, day=d))
        except ValueError:
            pass
    return out

def build_monthly_from_start(year: int, start_m: int, until_m: int = 12) -> set:
    out = set()
    for m in range(start_m, until_m + 1):
        day_candidates = np.array([6, 9, 13, 16, 20, 24], dtype=int)
        d = int(rng.choice(day_candidates))
        try:
            out.add(pd.Timestamp(year=year, month=m, day=d))
        except ValueError:
            pass
    return out

def pick_n_unique_days_in_year(year_dates: pd.DatetimeIndex, n: int) -> set:
    n = max(0, int(n))
    if n == 0 or len(year_dates) == 0:
        return set()
    chosen = pd.to_datetime(
        rng.choice(year_dates, size=min(n, len(year_dates)), replace=False)
    )
    return set(chosen)

def make_row(date, amount, category, description, payment_method=DEFAULT_PAYMENT,
             is_fixed=False, memo="") -> dict:
    """행 딕셔너리 생성 헬퍼. None/NaN 방어 처리 포함."""
    return {
        "date":           date if isinstance(date, type(pd.Timestamp("2024-01-01").date())) else date.date(),
        "amount":         int(round_to_100(amount)),
        "category":       str(category)       if category       is not None else "",
        "description":    str(description)    if description    is not None else "",
        "payment_method": str(payment_method) if payment_method is not None else DEFAULT_PAYMENT,
        "is_fixed":       bool(is_fixed),
        "memo":           str(memo)            if memo           is not None else "",
    }


# =========================================================
# 1) 페르소나 결정
# =========================================================

spending_profile = str(rng.choice(["tight", "normal", "affluent"], p=[0.35, 0.45, 0.20]))

PROFILE_LAMBDA_SCALE = {"tight": 0.90, "normal": 1.00, "affluent": 1.12}
PROFILE_AMOUNT_SCALE = {"tight": 0.92, "normal": 1.00, "affluent": 1.12}

gender    = str(rng.choice(["female", "male"], p=[0.52, 0.48]))
age_group = str(rng.choice(["20s", "30s", "40s", "50s"], p=[0.42, 0.38, 0.14, 0.06]))

# Employment status
# 20대: 직장인 위주 / 30~50대 여성: 전업주부 가능
_emp_female_homemaker_prob = {"20s": 0.08, "30s": 0.28, "40s": 0.35, "50s": 0.40}
if gender == "female":
    _is_homemaker = rng.random() < _emp_female_homemaker_prob.get(age_group, 0.10)
else:
    _is_homemaker = False
employment_status = "homemaker" if _is_homemaker else "salaryman"

# 교통 수단
TRANSPORT_MODE_PROB = {
    "tight":    {"transit": 0.80, "car": 0.20},
    "normal":   {"transit": 0.70, "car": 0.30},
    "affluent": {"transit": 0.55, "car": 0.45},
}
# 전업주부는 버스 비중 높음
if employment_status == "homemaker":
    transport_mode = "transit"
else:
    transport_mode = str(rng.choice(
        ["transit", "car"],
        p=[TRANSPORT_MODE_PROB[spending_profile]["transit"],
           TRANSPORT_MODE_PROB[spending_profile]["car"]]
    ))

# 자녀
HAS_CHILD_PROB = {"20s": 0.05, "30s": 0.30, "40s": 0.60, "50s": 0.65}
has_children   = bool(rng.random() < HAS_CHILD_PROB[age_group])

children_count = 0
if has_children:
    _cp = {
        "20s": np.array([0.85, 0.14, 0.01]),
        "30s": np.array([0.62, 0.32, 0.06]),
        "40s": np.array([0.42, 0.45, 0.13]),
        "50s": np.array([0.45, 0.43, 0.12]),
    }[age_group]
    _cp /= _cp.sum()
    children_count = int(rng.choice([1, 2, 3], p=_cp))

# 반려동물
PET_PROB_BASE = {"20s": 0.28, "30s": 0.30, "40s": 0.18, "50s": 0.12}[age_group]
PET_ADJ       = {"tight": 0.92, "normal": 1.00, "affluent": 1.05}[spending_profile]
pet_owner     = bool(rng.random() < min(0.55, PET_PROB_BASE * PET_ADJ))

# 자영업 여부 (전업주부 제외, 30대 이상)
SELF_EMP_PROB = {"20s": 0.04, "30s": 0.12, "40s": 0.18, "50s": 0.16}
is_self_employed = (
    employment_status != "homemaker" and
    rng.random() < SELF_EMP_PROB.get(age_group, 0.0)
)

# 부모 부양 (40~50대, 요양원)
NURSING_PROB = {"20s": 0.00, "30s": 0.04, "40s": 0.14, "50s": 0.22}
has_nursing_expense = bool(rng.random() < NURSING_PROB.get(age_group, 0.0))

# 만성질환
CHRONIC_PROB = {"20s": 0.04, "30s": 0.08, "40s": 0.18, "50s": 0.30}
has_chronic_disease = bool(rng.random() < CHRONIC_PROB.get(age_group, 0.0))


# =========================================================
# affluent 서브타입 결정 (A1/A2/A3)
# - 일반/tight 프로파일에는 영향 없음
# =========================================================
# A1(투자형): 대출 없음, 저축/투자 비중 높음, 여행↑
# A2(주담대형): 주택담보대출 보유, 여행 보통, 안정적 소비
# A3(라이프스타일형): 구독/외식/뷰티 지출↑, 대출 적음
affluent_type: str = "none"  # non-affluent
if spending_profile == "affluent":
    _aff_p = np.array([0.25, 0.52, 0.23], dtype=float)   # A1 / A2 / A3 비중
    affluent_type = str(rng.choice(["A1", "A2", "A3"], p=_aff_p / _aff_p.sum()))

# affluent 주거 설정: A2는 주담대 확률↑, A1은 무대출↑
# (실제 고정비 결정은 섹션 11에서 affluent_type을 참조)


# =========================================================
# 전체 지출 스케일
# =========================================================

AGE_AMOUNT_MULT = {"20s": 0.95, "30s": 1.10, "40s": 1.25, "50s": 1.34}[age_group]
AGE_LAMBDA_MULT = {"20s": 1.04, "30s": 1.06, "40s": 1.10, "50s": 1.08}[age_group]

CHILD_AMOUNT_MULT = 1.00
CHILD_LAMBDA_MULT = 1.00
if has_children:
    CHILD_LAMBDA_MULT = 1.06 if age_group in ("30s", "40s", "50s") else 1.02
    CHILD_AMOUNT_MULT = (1.14 if age_group == "30s"
                         else (1.22 if age_group in ("40s", "50s") else 1.06))

CHILD_COUNT_MULT = 1.00 + (0.08 * max(0, children_count - 1))

TOTAL_AMOUNT_MULT = AGE_AMOUNT_MULT * CHILD_AMOUNT_MULT * CHILD_COUNT_MULT
TOTAL_LAMBDA_MULT = AGE_LAMBDA_MULT * CHILD_LAMBDA_MULT * (
    1.00 + 0.02 * max(0, children_count - 1)
)


# =========================================================
# 2) 카페 설명 샘플러
# =========================================================

cafe_brands = [
    "메가MGC커피", "컴포즈커피", "이디야", "스타벅스", "투썸플레이스",
    "공차", "더벤티", "매머드 익스프레스", "커피베이", "파스쿠찌", "개인카페"
]
_base_cafe_w = np.array(
    [0.12, 0.12, 0.09, 0.08, 0.07, 0.06, 0.10, 0.08, 0.06, 0.05, 0.17], dtype=float
)

def cafe_weights_by_profile(profile: str) -> np.ndarray:
    w = _base_cafe_w.copy()
    if profile == "tight":
        w[[0, 1, 6, 7]] *= 1.25
        w[[3, 4, 9]]    *= 0.75
    elif profile == "affluent":
        w[[0, 1, 6, 7]] *= 0.85
        w[[3, 4, 9]]    *= 1.25
        w[[10]]         *= 1.20
    w = np.clip(w, 1e-6, None)
    return w / w.sum()

def sample_cafe_description(weekday: int, profile: str) -> str:
    weights = cafe_weights_by_profile(profile)
    brand   = str(rng.choice(cafe_brands, p=weights))
    base_dp = 0.22 if weekday < 5 else 0.35
    if age_group == "20s":
        base_dp *= 1.10
    elif age_group in ("40s", "50s"):
        base_dp *= 0.90
    if profile == "tight":
        base_dp *= 0.85
    elif profile == "affluent":
        base_dp *= 1.10
    base_dp = float(np.clip(base_dp, 0.05, 0.65))
    item    = "음료+디저트" if rng.random() < base_dp else "음료"
    return f"{brand} {item}"


desc_pool = {
    "식비_일반":    ["점심 식사", "저녁 외식", "배달 음식", "편의점", "장보기(마트)"],
    "의료/건강":    ["병원 진료", "약국", "치과", "헬스장", "동물병원"],
    # 숙박(호텔/리조트/에어비앤비)은 일상 풀에서 완전 제거 → 여행/출장 시나리오에서만 생성
    "문화/여가":    ["영화 관람", "전시회", "공연", "입장권/티켓"],
    "교육_변동":    ["도서 구매", "자격증 응시료", "온라인 강의", "스터디 비용", "특강/세미나"],
    "기타":         ["경조사", "수수료", "기부", "벌금", "기타 지출", "명절 선물/용돈"],
}


# =========================================================
# 3) 쇼핑 플랫폼
# =========================================================

FASHION_PLATFORMS = ["무신사", "29CM", "W컨셉", "지그재그", "에이블리", "KREAM"]
SECONDHAND_PLATFORMS = ["당근마켓", "번개장터", "중고나라"]

shopping_platforms = {
    "무신사":        {"의류": 0.60, "신발": 0.25, "잡화": 0.15, "선물": 0.00, "생필품": 0.00, "화장품": 0.00, "반려동물": 0.00},
    "29CM":          {"의류": 0.40, "신발": 0.10, "잡화": 0.40, "선물": 0.10, "생필품": 0.00, "화장품": 0.00, "반려동물": 0.00},
    "W컨셉":         {"의류": 0.60, "신발": 0.10, "잡화": 0.25, "선물": 0.05, "생필품": 0.00, "화장품": 0.00, "반려동물": 0.00},
    "지그재그":      {"의류": 0.75, "신발": 0.10, "잡화": 0.10, "선물": 0.05, "생필품": 0.00, "화장품": 0.00, "반려동물": 0.00},
    "에이블리":      {"의류": 0.80, "신발": 0.10, "잡화": 0.06, "선물": 0.04, "생필품": 0.00, "화장품": 0.00, "반려동물": 0.00},
    "KREAM":         {"의류": 0.15, "신발": 0.75, "잡화": 0.10, "선물": 0.00, "생필품": 0.00, "화장품": 0.00, "반려동물": 0.00},
    "쿠팡":          {"의류": 0.07, "신발": 0.05, "잡화": 0.10, "선물": 0.06, "생필품": 0.62, "화장품": 0.06, "반려동물": 0.04},
    "네이버쇼핑":    {"의류": 0.22, "신발": 0.14, "잡화": 0.22, "선물": 0.10, "생필품": 0.22, "화장품": 0.07, "반려동물": 0.03},
    "카카오선물하기":{"의류": 0.04, "신발": 0.02, "잡화": 0.12, "선물": 0.76, "생필품": 0.04, "화장품": 0.02, "반려동물": 0.00},
    "오늘의집":      {"의류": 0.00, "신발": 0.00, "잡화": 0.45, "선물": 0.10, "생필품": 0.45, "화장품": 0.00, "반려동물": 0.00},
    "올리브영":      {"의류": 0.00, "신발": 0.00, "잡화": 0.05, "선물": 0.05, "생필품": 0.10, "화장품": 0.75, "반려동물": 0.05},
    "백화점":        {"의류": 0.40, "신발": 0.12, "잡화": 0.28, "선물": 0.10, "생필품": 0.05, "화장품": 0.05, "반려동물": 0.00},
    "아울렛":        {"의류": 0.55, "신발": 0.18, "잡화": 0.22, "선물": 0.03, "생필품": 0.02, "화장품": 0.00, "반려동물": 0.00},
    "당근마켓":      {"의류": 0.18, "신발": 0.18, "잡화": 0.50, "선물": 0.00, "생필품": 0.10, "화장품": 0.02, "반려동물": 0.02},
    "번개장터":      {"의류": 0.18, "신발": 0.35, "잡화": 0.43, "선물": 0.00, "생필품": 0.00, "화장품": 0.02, "반려동물": 0.00},
    "중고나라":      {"의류": 0.12, "신발": 0.22, "잡화": 0.56, "선물": 0.00, "생필품": 0.05, "화장품": 0.03, "반려동물": 0.02},
}

def secondhand_prob_by_profile(profile: str) -> float:
    return {"tight": 0.16, "normal": 0.12, "affluent": 0.0}[profile]

def sample_shopping_description(profile: str) -> str:
    if rng.random() < secondhand_prob_by_profile(profile):
        platform = str(rng.choice(SECONDHAND_PLATFORMS))
    else:
        candidates = [p for p in shopping_platforms if p not in SECONDHAND_PLATFORMS]
        if age_group in ("40s", "50s"):
            weights = []
            for p in candidates:
                if p in FASHION_PLATFORMS:
                    weights.append(0.02)
                elif p in ("백화점", "아울렛"):
                    weights.append(2.0 if age_group == "50s" else 1.8)
                elif p in ("쿠팡", "네이버쇼핑", "오늘의집", "카카오선물하기"):
                    weights.append(0.65)
                elif p == "올리브영":
                    weights.append(0.12 if gender == "male" else 0.50)
                else:
                    weights.append(0.30)
            w = np.array(weights, dtype=float)
            w /= w.sum()
            platform = str(rng.choice(candidates, p=w))
        else:
            platform = str(rng.choice(candidates))

    item_probs = shopping_platforms[platform].copy()
    if not pet_owner:
        item_probs["반려동물"] = 0.0
    if gender == "male":
        item_probs["화장품"] = 0.0

    items  = list(item_probs.keys())
    probs  = np.array(list(item_probs.values()), dtype=float)
    probs  = np.clip(probs, 0.0, None)
    if probs.sum() == 0:
        probs = np.ones(len(probs))
    probs /= probs.sum()
    item   = str(rng.choice(items, p=probs))

    if platform in SECONDHAND_PLATFORMS:
        return f"{platform} | 중고 {item} 구매"
    return f"{platform} | {item} 구매"


# =========================================================
# 4) 취미/덕질
# =========================================================

if spending_profile == "tight":
    hobby_active    = False
    hobby_intensity = "none"
    hobby_type      = None
else:
    HOBBY_ACTIVE_BASE = {"20s": 0.72, "30s": 0.62, "40s": 0.52, "50s": 0.45}[age_group]
    if spending_profile == "affluent":
        HOBBY_ACTIVE_BASE *= 1.08
    hobby_active = bool(rng.random() < clamp01(HOBBY_ACTIVE_BASE))

    hobby_intensity = "none"
    if hobby_active:
        _h_p = np.array(
            {"20s": [0.62, 0.38], "30s": [0.70, 0.30],
             "40s": [0.78, 0.22], "50s": [0.82, 0.18]}[age_group]
        )
        _h_p /= _h_p.sum()
        hobby_intensity = str(rng.choice(["light", "deep"], p=_h_p))

    HOBBY_TYPES = ["게임", "콘텐츠", "다꾸/문구", "공예", "베이킹", "요리",
                   "스포츠/운동", "사진/영상", "악기/음악"]

    def _hobby_type_probs():
        p = np.array([0.16, 0.12, 0.12, 0.10, 0.10, 0.08, 0.16, 0.08, 0.08], dtype=float)
        if gender == "female":
            p[0] *= 0.85; p[2] *= 1.20; p[3] *= 1.15; p[4] *= 1.15
        else:
            p[0] *= 1.15; p[2] *= 0.80; p[3] *= 0.90
        if age_group == "20s":
            p[0] *= 1.15; p[1] *= 1.10; p[7] *= 1.05
        elif age_group == "30s":
            p[5] *= 1.10; p[6] *= 1.10
        elif age_group in ("40s", "50s"):
            p[5] *= 1.15; p[6] *= 1.10; p[0] *= 0.85; p[1] *= 0.90
        p = np.clip(p, 1e-6, None)
        return p / p.sum()

    hobby_type = None
    if hobby_intensity != "none":
        hobby_type = str(rng.choice(HOBBY_TYPES, p=_hobby_type_probs()))

hobby_active_months_by_year: dict[int, set] = {2024: set(), 2025: set()}
if hobby_intensity != "none":
    _hy_start = int(rng.choice([2024, 2025], p=[0.62, 0.38]))
    _hy_sm    = choose_start_month_with_seasonality()
    _hy_len   = int(rng.integers(4, 13))
    if rng.random() < 0.18:
        _hy_len = int(rng.integers(2, max(3, _hy_len)))
    hobby_active_months_by_year[_hy_start] |= set(month_range(_hy_sm, _hy_len))
    if _hy_start == 2024:
        _carry_p = 0.45 if hobby_intensity == "deep" else 0.25
        if rng.random() < _carry_p:
            hobby_active_months_by_year[2025] |= set(month_range(1, int(rng.integers(3, 10))))

HOBBY_CHANNELS = ["온라인몰", "매장", "중고거래"]

def sample_hobby_channel() -> str:
    p = (np.array([0.62, 0.34, 0.04]) if spending_profile == "affluent"
         else np.array([0.60, 0.30, 0.10]))
    return str(rng.choice(HOBBY_CHANNELS, p=p))

def sample_hobby_purchase_desc(h_type: str) -> str:
    ch = sample_hobby_channel()
    _items_map = {
        "게임":        (["게임 구매", "DLC/확장팩", "인게임 결제(소액)", "콘솔/주변기기"],
                        [0.42, 0.18, 0.28, 0.12]),
        "콘텐츠":      (["웹툰/웹소설 결제", "영화/드라마 VOD", "전자책 결제", "유료 강의(취미)"],
                        [0.38, 0.30, 0.22, 0.10]),
        "다꾸/문구":   (["스티커 구매", "마스킹테이프 구매", "다이어리/노트 구매", "펜/형광펜 구매", "문구 세트"],
                        [0.28, 0.22, 0.20, 0.20, 0.10]),
        "공예":        (["비즈/레진 재료", "실/원단", "공예 도구", "페인트/붓", "DIY 키트"],
                        [0.25, 0.22, 0.22, 0.16, 0.15]),
        "베이킹":      (["베이킹 재료", "베이킹 도구", "틀/오븐용품", "포장재(상자/리본)"],
                        [0.35, 0.30, 0.22, 0.13]),
        "요리":        (["요리 도구", "조미료/소스", "식재료(특수)", "밀키트(취미)"],
                        [0.28, 0.22, 0.25, 0.25]),
        "스포츠/운동": (["운동복/소품", "러닝화/신발", "장비(라켓/골프 등)", "보충제/프로틴(취미)"],
                        [0.30, 0.22, 0.30, 0.18]),
        "사진/영상":   (["촬영 소품", "편집 앱/툴 결제", "삼각대/조명", "메모리/배터리"],
                        [0.22, 0.20, 0.34, 0.24]),
        "악기/음악":   (["악기 소모품(줄/리드)", "악보/음원 구매", "악기 액세서리", "중고 악기 구매"],
                        [0.30, 0.22, 0.28, 0.20]),
    }
    items, probs = _items_map.get(h_type, (["기타 취미 용품"], [1.0]))
    p = np.array(probs, dtype=float)
    p /= p.sum()
    item = str(rng.choice(items, p=p))
    return f"{ch} | {item}"

def sample_hobby_amount(h_type: str, intensity: str) -> int:
    mult = 1.00 if intensity == "light" else 1.60
    _ranges = {
        "게임":        (8_000, 180_000), "콘텐츠":      (3_000, 70_000),
        "다꾸/문구":   (3_000, 65_000),  "공예":        (8_000, 120_000),
        "베이킹":      (10_000, 140_000),"요리":        (8_000, 120_000),
        "스포츠/운동": (10_000, 220_000),"사진/영상":   (10_000, 180_000),
        "악기/음악":   (8_000, 170_000),
    }
    lo, hi = _ranges.get(h_type, (5_000, 80_000))
    amt    = int(rng.integers(lo, hi + 1)) * mult
    if spending_profile == "affluent":
        amt *= 1.15
    return round_to_100(amt)


# 덕질
_fandom_base = {
    ("female", "20s"): 0.22, ("female", "30s"): 0.15,
    ("female", "40s"): 0.07, ("female", "50s"): 0.04,
    ("male",   "20s"): 0.06,
}
FANDOM_BASE = _fandom_base.get((gender, age_group), 0.03)
if spending_profile == "tight":
    FANDOM_BASE *= 0.88
elif spending_profile == "affluent":
    FANDOM_BASE *= 1.08
fandom_persona = bool(rng.random() < clamp01(FANDOM_BASE))

FANDOM_TARGETS      = ["아이돌A", "아이돌B", "아이돌C", "아이돌D", "아이돌E"]
FANDOM_SWITCH_PROB  = 0.35

def build_fandom_target_schedule_12m() -> dict[int, str]:
    current  = str(rng.choice(FANDOM_TARGETS))
    schedule = {m: current for m in range(1, 13)}
    if rng.random() < FANDOM_SWITCH_PROB:
        switch_m = int(rng.integers(3, 11))
        next_t   = str(rng.choice([t for t in FANDOM_TARGETS if t != current]))
        for m in range(switch_m, 13):
            schedule[m] = next_t
    return schedule

def pick_concert_months() -> list[int]:
    candidates = [3, 4, 5, 6, 9, 10, 11, 12]
    k = 1 if rng.random() < 0.75 else 2
    return sorted(rng.choice(candidates, size=k, replace=False).tolist())

fandom_active_months_by_year: dict[int, set] = {2024: set(), 2025: set()}
fandom_target_by_month_by_year: dict[int, dict | None] = {2024: None, 2025: None}
concert_months_by_year: dict[int, list] = {2024: [], 2025: []}

if fandom_persona:
    _fy_start = int(rng.choice([2024, 2025], p=[0.60, 0.40]))
    _fy_sm    = choose_start_month_with_seasonality()
    fandom_active_months_by_year[_fy_start] = set(range(_fy_sm, 13))
    if _fy_start == 2024 and rng.random() < 0.72:
        fandom_active_months_by_year[2025] = set(range(1, 13))
    fandom_target_by_month_by_year[2024] = build_fandom_target_schedule_12m()
    fandom_target_by_month_by_year[2025] = build_fandom_target_schedule_12m()
    for y in (2024, 2025):
        if len(fandom_active_months_by_year[y]) > 0:
            cm = pick_concert_months()
            concert_months_by_year[y] = [m for m in cm if m in fandom_active_months_by_year[y]]

FANDOM_SMALL   = {"tight": (8_000,  30_000), "normal": (10_000,  80_000), "affluent": (20_000, 300_000)}
FANDOM_CONCERT = {"tight": (90_000, 180_000),"normal": (90_000, 300_000), "affluent": (90_000, 500_000)}


# =========================================================
# 5) 의료 / 교정 / 검진
# =========================================================

ortho_type = "none"
p_self, p_child = 0.0, 0.0
if age_group == "20s":
    p_self = 0.20 if gender == "female" else 0.12
elif age_group == "30s":
    p_self = 0.10 if gender == "female" else 0.06
elif age_group == "40s":
    p_self = 0.02
    if has_children:
        p_child = 0.10 + 0.05 * max(0, children_count - 1)
elif age_group == "50s":
    p_self = 0.005

if spending_profile == "tight":
    p_self *= 0.85; p_child *= 0.88
elif spending_profile == "affluent":
    p_self *= 1.10; p_child *= 1.08

p_self  = float(np.clip(p_self,  0.0, 0.40))
p_child = float(np.clip(p_child, 0.0, 0.35))

_r = rng.random()
if _r < p_self:
    ortho_type = "self"
elif _r < p_self + p_child:
    ortho_type = "child"

ortho_days: set = set()
if ortho_type != "none":
    ortho_days |= build_monthly_like_schedule(2024)
    ortho_days |= build_monthly_like_schedule(2025)

dental_check_days: set = set()
for _y in (2024, 2025):
    if ortho_type == "none" and rng.random() < 0.75:
        _yd = pd.date_range(f"{_y}-01-01", f"{_y}-12-31", freq="D")
        dental_check_days |= pick_n_unique_days_in_year(_yd, 1)

checkup_days: set = set()
for _y in (2024, 2025):
    _yd = pd.date_range(f"{_y}-01-01", f"{_y}-12-31", freq="D")
    if rng.random() < 0.65:
        checkup_days |= pick_n_unique_days_in_year(_yd, 1)

DENTAL_CHECK_RANGE = (50_000, 100_000)


# =========================================================
# 6) 미용 / 네일 / 피부관리
# =========================================================

beauty_persona = False
if gender == "female":
    _bp = {"20s": 0.42, "30s": 0.38, "40s": 0.26, "50s": 0.12}[age_group]
    if spending_profile == "tight":
        _bp *= 0.90
    elif spending_profile == "affluent":
        _bp *= 1.10
    beauty_persona = bool(rng.random() < clamp01(_bp))

nail_persona = False
skin_persona = False
if gender == "female" and beauty_persona:
    _np = {"20s": 0.42, "30s": 0.32, "40s": 0.06, "50s": 0.02}[age_group]
    _sp = {"20s": 0.34, "30s": 0.30, "40s": 0.22, "50s": 0.08}[age_group]
    if spending_profile == "tight":
        _np *= 0.88; _sp *= 0.90
    elif spending_profile == "affluent":
        _np *= 1.08; _sp *= 1.10
    nail_persona = bool(rng.random() < clamp01(_np))
    skin_persona = bool(rng.random() < clamp01(_sp))

hair_days: set = set()
nail_days: set = set()
skin_days: set = set()
derm_proc_days: set = set()

for _y in (2024, 2025):
    if gender == "male":
        hair_days |= build_monthly_like_schedule(_y)
    else:
        hair_days |= build_bimonthly_schedule(_y)

if nail_persona:
    _ny = int(rng.choice([2024, 2025], p=[0.60, 0.40]))
    _ns = choose_start_month_with_seasonality()
    nail_days |= build_monthly_from_start(_ny, _ns)
    if _ny == 2024 and rng.random() < 0.70:
        nail_days |= build_monthly_from_start(2025, 1)

if skin_persona:
    _sy = int(rng.choice([2024, 2025], p=[0.62, 0.38]))
    _ss = choose_start_month_with_seasonality()
    skin_days |= build_monthly_from_start(_sy, _ss)
    if _sy == 2024 and rng.random() < 0.75:
        skin_days |= build_monthly_from_start(2025, 1)

DERM_PROC_RANGE = {"tight": None, "normal": (50_000, 200_000), "affluent": (100_000, 1_000_000)}
NAIL_RANGE      = {"tight": None, "normal": (35_000, 70_000),  "affluent": (50_000, 150_000)}
SKINCARE_RANGE  = {"tight": None, "normal": (50_000, 100_000), "affluent": (100_000, 200_000)}

for _y in (2024, 2025):
    if DERM_PROC_RANGE[spending_profile] is None:
        continue
    if gender == "female" and (skin_persona or nail_persona or beauty_persona):
        _yd = pd.date_range(f"{_y}-01-01", f"{_y}-12-31", freq="D")
        if rng.random() < 0.65:
            derm_proc_days |= pick_n_unique_days_in_year(_yd, int(rng.integers(1, 3)))


# =========================================================
# 7) 장보기 스케줄 사전 확정
# =========================================================

def build_grocery_schedule(
    start: str,
    end: str,
    employment: str,
    has_child: bool,
) -> set:
    """
    주 단위로 장보기 날짜를 사전 확정한다.

    Parameters
    ----------
    start       : 시작일 문자열 (YYYY-MM-DD)
    end         : 종료일 문자열 (YYYY-MM-DD)
    employment  : "salaryman" | "homemaker"
    has_child   : 자녀 유무

    Returns
    -------
    set of pd.Timestamp: 장보기 발생 날짜 집합

    설계 규칙
    ----------
    - 자녀 가구: 주 1회 고정
    - 미혼/무자녀: 주 최대 2회
    - 직장인: 주말(토=5, 일=6) 가중치 높음
    - 전업주부: 평일(화=1, 목=3) 가중치 높음
    """
    all_dates    = pd.date_range(start, end, freq="D")
    weekly_groups: dict[tuple, list] = {}
    for d in all_dates:
        key = (d.isocalendar()[0], d.isocalendar()[1])  # (year, week)
        weekly_groups.setdefault(key, []).append(d)

    # 요일별 가중치 (0=월 ~ 6=일)
    if employment == "homemaker":
        day_weights = np.array([0.8, 1.5, 0.8, 1.5, 0.8, 0.5, 0.4], dtype=float)
    else:
        day_weights = np.array([0.3, 0.3, 0.3, 0.3, 0.5, 1.5, 1.2], dtype=float)

    result: set = set()
    max_per_week = 1 if has_child else 2

    for week_days in weekly_groups.values():
        if not week_days:
            continue
        n_buy = 1 if has_child else int(rng.choice([1, 2], p=[0.55, 0.45]))
        n_buy = min(n_buy, max_per_week, len(week_days))
        week_arr = np.array(week_days)
        wd_arr   = np.array([d.weekday() for d in week_days])
        w        = day_weights[wd_arr]
        w       /= w.sum()
        chosen   = rng.choice(week_arr, size=n_buy, replace=False, p=w)
        for c in chosen:
            result.add(pd.Timestamp(c))

    return result


grocery_days = build_grocery_schedule(
    start=START_DATE,
    end=END_DATE,
    employment=employment_status,
    has_child=has_children,
)


# =========================================================
# 8) 변동 카테고리 확률
# =========================================================

var_categories = ["식비", "교통비", "쇼핑", "의료/건강", "문화/여가", "교육", "기타"]
base_probs     = np.array([0.40, 0.18, 0.12, 0.06, 0.10, 0.05, 0.09], dtype=float)

def profile_category_bias(profile: str, mode: str) -> np.ndarray:
    p   = base_probs.copy()
    idx = {c: i for i, c in enumerate(var_categories)}

    # 전업주부: 식비·의료 높음, 쇼핑 낮음
    if employment_status == "homemaker":
        p[idx["식비"]]     += 0.05
        p[idx["의료/건강"]] += 0.02
        p[idx["쇼핑"]]     -= 0.03
        p[idx["교통비"]]   -= 0.02

    if profile == "tight":
        p[idx["식비"]]     += 0.03
        p[idx["쇼핑"]]     -= 0.02
        p[idx["문화/여가"]] -= 0.01
    elif profile == "affluent":
        p[idx["쇼핑"]]     += 0.02
        p[idx["문화/여가"]] += 0.02
        p[idx["식비"]]     -= 0.02
        p[idx["기타"]]     -= 0.02

    if mode == "car":
        p[idx["교통비"]]   += 0.03
        p[idx["식비"]]     -= 0.02
        p[idx["기타"]]     -= 0.01

    if age_group == "20s":
        p[idx["쇼핑"]]     += 0.04
        p[idx["문화/여가"]] += 0.02
        p[idx["교육"]]     -= 0.02
    elif age_group == "30s":
        p[idx["교육"]]     += 0.02
        p[idx["문화/여가"]] += 0.03
        p[idx["쇼핑"]]     -= 0.01
    elif age_group == "40s":
        p[idx["교육"]]     += 0.05
        p[idx["식비"]]     += 0.03
        p[idx["쇼핑"]]     -= 0.04
        p[idx["문화/여가"]] -= 0.01
    else:
        p[idx["교육"]]     += 0.04
        p[idx["식비"]]     += 0.03
        p[idx["쇼핑"]]     -= 0.05
        p[idx["문화/여가"]] -= 0.01

    if has_children:
        p[idx["교육"]]     += 0.03 + 0.01 * max(0, children_count - 1)
        p[idx["식비"]]     += 0.02 + 0.01 * max(0, children_count - 1)
        p[idx["문화/여가"]] -= 0.02
        p[idx["기타"]]     -= 0.03

    if pet_owner:
        p[idx["의료/건강"]] += 0.01
        p[idx["쇼핑"]]     += 0.01
        p[idx["기타"]]     -= 0.02

    p = np.clip(p, 0.01, None)
    return p / p.sum()

# 직장인: 평일 거래 많음 / 전업주부: 평일 낮 위주(주말 낮음)
if employment_status == "homemaker":
    lambda_by_weekday = {0: 2.4, 1: 2.3, 2: 2.2, 3: 2.3, 4: 2.0, 5: 1.2, 6: 1.0}
else:
    lambda_by_weekday = {0: 2.2, 1: 2.2, 2: 2.0, 3: 2.1, 4: 2.3, 5: 1.6, 6: 1.5}

# cohort_engine.py의 LOGNORM_PARAMS와 동일
lognorm_params = {
    "식비":     (9.15, 0.55),
    "교통비":   (8.30, 0.55),
    "쇼핑":     (10.15, 0.85),
    "의료/건강":(9.95, 0.75),
    "문화/여가":(9.80, 0.70),
    "교육":     (10.05, 0.70),
    "기타":     (9.40, 0.85),
}


# =========================================================
# 9) 명절 시즌성
# =========================================================

def make_holiday_window(year: int, kind: str) -> pd.DatetimeIndex:
    if kind == "seol":
        cands = pd.date_range(f"{year}-01-20", f"{year}-02-20", freq="D")
    else:
        cands = pd.date_range(f"{year}-09-10", f"{year}-10-20", freq="D")
    s = pd.Timestamp(rng.choice(cands))
    return pd.date_range(s, s + pd.Timedelta(days=2), freq="D")

HOLIDAY_WINDOWS = []
for _y in (2024, 2025):
    HOLIDAY_WINDOWS.append(make_holiday_window(_y, "seol"))
    HOLIDAY_WINDOWS.append(make_holiday_window(_y, "chuseok"))

def holiday_factor(d: pd.Timestamp) -> float:
    def _dist(date, window):
        if date < window.min(): return (window.min() - date).days
        if date > window.max(): return (date - window.max()).days
        return 0
    dist = min(_dist(d, w) for w in HOLIDAY_WINDOWS)
    if dist == 0:   return 1.18
    if dist <= 2:   return 1.12
    if dist <= 5:   return 1.06
    return 1.00

def adjust_category_probs(d: pd.Timestamp, wd: int, probs: np.ndarray) -> np.ndarray:
    p   = probs.copy()
    idx = {c: i for i, c in enumerate(var_categories)}

    if wd in (5, 6):
        p[idx["문화/여가"]] += 0.05
        p[idx["쇼핑"]]     += 0.03
        p[idx["식비"]]     -= 0.06
    if d.month in (7, 8):
        p[idx["문화/여가"]] += 0.04
        p[idx["식비"]]     -= 0.02
        p[idx["교통비"]]   -= 0.01
    if d.month == 12:
        p[idx["쇼핑"]]     += 0.06
        p[idx["기타"]]     += 0.01
        p[idx["식비"]]     -= 0.04
        p[idx["교육"]]     -= 0.01

    hf = holiday_factor(d)
    if hf > 1.0:
        p[idx["쇼핑"]]     += 0.04
        p[idx["교통비"]]   += 0.02
        p[idx["기타"]]     += 0.02
        p[idx["식비"]]     += 0.01
        p[idx["문화/여가"]] -= 0.03

    p = np.clip(p, 0.01, None)
    return p / p.sum()


# =========================================================
# 10) 교통 금액 규칙
# =========================================================

TAXI_MAX   = {"tight": 15_000, "normal": 20_000, "affluent": 50_000}
TRAIN_RANGE = (40_000, 80_000)

def taxi_prob(profile: str, mode: str) -> float:
    if profile == "tight":    return 0.01
    if profile == "affluent": return 0.16 if mode == "transit" else 0.18
    return 0.10 if mode == "transit" else 0.11

def sample_transport_description(mode: str, profile: str) -> str:
    # 전업주부: 버스 비중 매우 높음
    if employment_status == "homemaker":
        items  = ["지하철 교통카드", "버스", "택시"]
        probs  = np.array([0.25, 0.65, 0.10], dtype=float)
        probs /= probs.sum()
        return str(rng.choice(items, p=probs))

    if mode == "transit":
        p_taxi   = taxi_prob(profile, mode)
        p_train  = 0.05
        p_flight = 0.01
        p_subway = 0.56
        p_bus    = max(0.001, 1.0 - (p_subway + p_taxi + p_train + p_flight))
        items    = ["지하철 교통카드", "버스", "택시", "KTX/기차", "항공권"]
        probs    = np.array([p_subway, p_bus, p_taxi, p_train, p_flight], dtype=float)
        probs    = np.clip(probs, 0.001, None)
        probs   /= probs.sum()
        return str(rng.choice(items, p=probs))

    if rng.random() < (0.08 if profile == "tight" else 0.10):
        return str(rng.choice(["지하철 교통카드", "버스"], p=[0.6, 0.4]))
    if rng.random() < taxi_prob(profile, mode):
        return "택시"
    cp = np.array([0.50, 0.24, 0.10, 0.08, 0.08], dtype=float)
    cp /= cp.sum()
    return str(rng.choice(["주유", "주차", "통행료", "세차", "정비/오일"], p=cp))

def generate_amount(cat: str, profile: str, desc: str = "") -> int:
    """
    카테고리+프로파일 기반 금액을 생성한다.

    desc에 숙박 키워드가 포함된 경우, 별도 금액 구간 + 최솟값 하한을 강제한다.
    (숙박은 여행 시나리오에서만 나오지만, 방어층으로 desc 기반 체크 유지)
    """
    _LODGING_DESC_KEYWORDS = ["호텔", "리조트", "숙박", "에어비앤비"]
    if desc and any(k in desc for k in _LODGING_DESC_KEYWORDS):
        # 숙박 금액 강제 하한: 프로파일별
        _lo_map = {"tight": 80_000, "normal": 120_000, "affluent": 200_000}
        _hi_map = {"tight": 200_000, "normal": 400_000, "affluent": 1_200_000}
        lo = _lo_map.get(profile, 80_000)
        hi = _hi_map.get(profile, 400_000)
        return int(round_to_100(int(rng.integers(lo, hi + 1))))

    mean, sigma = lognorm_params[cat]
    amt = rng.lognormal(mean=mean, sigma=sigma) * PROFILE_AMOUNT_SCALE[profile]
    amt *= TOTAL_AMOUNT_MULT
    return int(min(2_000_000, max(1_000, round_to_100(amt))))

def sample_food_description(wd: int, profile: str, is_grocery_day: bool = False) -> str:
    """
    식비 description 샘플러.
    is_grocery_day=True이면 장보기(마트)를 우선 반환한다.
    """
    if is_grocery_day:
        return "장보기(마트)"

    # 전업주부: 직접 조리 → 배달/외식 확률 낮음
    if employment_status == "homemaker":
        base_cafe_p   = 0.18 if wd < 5 else 0.12
        delivery_prob = 0.08
    else:
        base_cafe_p   = 0.25 if wd < 5 else 0.32
        delivery_prob = 0.20

    if age_group == "20s":
        base_cafe_p *= 1.12
    elif age_group in ("40s", "50s"):
        base_cafe_p *= 0.86
    if profile == "tight":
        base_cafe_p *= 0.90
    elif profile == "affluent":
        base_cafe_p *= 1.08
    base_cafe_p = float(np.clip(base_cafe_p, 0.05, 0.60))

    if rng.random() < base_cafe_p:
        return sample_cafe_description(wd, profile)
    if rng.random() < delivery_prob:
        return "배달 음식"
    return str(rng.choice(["점심 식사", "저녁 외식", "편의점"]))


# =========================================================
# 11) 고정비 dict 구조 (확장)
# =========================================================

PHONE_PAYDAY   = int(rng.integers(3, 8))
HOUSING_PAYDAY = int(rng.integers(1, 6))
SUB_PAYDAY     = int(rng.integers(20, 29))

# 고정비 플랜 리스트 — dict 구조 통일
fixed_plans: list[dict] = []

# --- 휴대폰 요금
fixed_plans.append({
    "payday":          PHONE_PAYDAY,
    "amount":          55_000,
    "category":        "주거/통신",
    "description":     "휴대폰 요금",
    "payment_method":  DEFAULT_PAYMENT,
    "memo":            "",
})

# --- 주거비 (대출 vs 월세, 상호배타)
if age_group in ("40s", "50s"):
    p_mortgage = 0.62 if spending_profile != "tight" else 0.50
    p_rent     = 0.25 if spending_profile != "tight" else 0.30
else:
    p_mortgage = 0.12 if spending_profile != "tight" else 0.08
    p_rent     = 0.60 if spending_profile != "tight" else 0.55

# affluent 타입별 주거 확률 조정
if spending_profile == "affluent":
    if affluent_type == "A1":   # 투자형: 주담대 낮음, 전세/무주택 많음
        p_mortgage = min(0.30, p_mortgage)
        p_rent     = max(p_rent, 0.45)
    elif affluent_type == "A2": # 주담대형: 주담대↑
        p_mortgage = min(0.92, p_mortgage + 0.35)
        p_rent     = max(0.05, p_rent - 0.20)
    # A3: 기본값 유지

if has_children:
    p_mortgage = min(0.85, p_mortgage + 0.10)
    p_rent     = min(0.80, p_rent + 0.06)

has_mortgage = bool(rng.random() < p_mortgage)
has_rent     = False if has_mortgage else bool(rng.random() < p_rent)

if has_mortgage:
    # affluent 주담대 금액: 타입별 현실화
    if spending_profile == "affluent":
        if affluent_type == "A2":
            _lo, _hi = (1_500_000, 3_500_000)   # A2: 주담대 크다
        else:
            _lo, _hi = (800_000, 2_200_000)
    elif spending_profile == "tight":
        _lo, _hi = (350_000, 900_000)
    else:
        _lo, _hi = (600_000, 2_000_000)

    _base = int(rng.integers(_lo, _hi + 1))
    if age_group in ("40s", "50s"):
        _base = int(_base * rng.uniform(1.05, 1.30))
    if has_children:
        _base = int(_base * (1.10 + 0.06 * max(0, children_count - 1)))
    fixed_plans.append({
        "payday":         HOUSING_PAYDAY,
        "amount":         round_to_100(min(5_000_000, _base)),
        "category":       "주거/통신",
        "description":    "대출 상환",
        "payment_method": TRANSFER_PAYMENT,
        "memo":           "",
    })
elif has_rent:
    if spending_profile == "affluent":
        _lo, _hi = (1_200_000, 2_500_000)
    elif spending_profile == "tight":
        _lo, _hi = (450_000, 900_000)
    else:
        _lo, _hi = (650_000, 1_250_000)

    _base = int(rng.integers(_lo, _hi + 1))
    if has_children:
        _bump = rng.uniform(1.20, 1.60) if age_group in ("40s", "50s") else rng.uniform(1.12, 1.40)
        _base = int(_base * _bump * (1.05 + 0.06 * max(0, children_count - 1)))
    fixed_plans.append({
        "payday":         HOUSING_PAYDAY,
        "amount":         round_to_100(min(4_500_000, _base)),
        "category":       "주거/통신",
        "description":    "월세",
        "payment_method": TRANSFER_PAYMENT,
        "memo":           "",
    })

# --- 자녀 용돈 (40~50대)
if age_group in ("40s", "50s") and has_children:
    _lo, _hi = {"tight": (100_000, 250_000), "normal": (150_000, 420_000),
                "affluent": (250_000, 600_000)}[spending_profile]
    _base = int(rng.integers(_lo, _hi + 1))
    _base = int(_base * (1.00 + 0.12 * max(0, children_count - 1)))
    if age_group == "50s":
        _base = int(_base * 1.08)
    fixed_plans.append({
        "payday":         int(rng.integers(12, 18)),
        "amount":         round_to_100(_base),
        "category":       "기타",
        "description":    "자녀 용돈",
        "payment_method": TRANSFER_PAYMENT,
        "memo":           "",
    })

# --- [신규] 만성질환 병원·약국 고정
if has_chronic_disease:
    _chronic_hosp_amt = round_to_100(int(rng.integers(15_000, 45_001)))
    _chronic_pharm_amt= round_to_100(int(rng.integers(8_000,  20_001)))
    _chronic_payday   = int(rng.integers(4, 15))
    fixed_plans.append({
        "payday":         _chronic_payday,
        "amount":         _chronic_hosp_amt,
        "category":       "의료/건강",
        "description":    "병원 진료",
        "payment_method": DEFAULT_PAYMENT,
        "memo":           "만성질환(고정)",
    })
    fixed_plans.append({
        "payday":         min(28, _chronic_payday + 1),
        "amount":         _chronic_pharm_amt,
        "category":       "의료/건강",
        "description":    "약국",
        "payment_method": DEFAULT_PAYMENT,
        "memo":           "만성질환(고정)",
    })

# --- [신규] 부모 요양원 부양
if has_nursing_expense:
    _lo, _hi = {"tight": (300_000, 600_000), "normal": (600_000, 1_200_000),
                "affluent": (1_000_000, 2_500_000)}[spending_profile]
    _nursing_amt = round_to_100(int(rng.integers(_lo, _hi + 1)))
    fixed_plans.append({
        "payday":         int(rng.integers(1, 5)),
        "amount":         _nursing_amt,
        "category":       "기타",
        "description":    "부모님 요양원",
        "payment_method": TRANSFER_PAYMENT,
        "memo":           "부모님 부양(고정)",
    })

# --- [신규] 자영업 (상가 월세 + 직원 급여)
if is_self_employed:
    # 상가 월세
    _s_lo, _s_hi = {"tight": (300_000, 700_000), "normal": (700_000, 1_500_000),
                    "affluent": (1_500_000, 3_500_000)}[spending_profile]
    _store_rent = round_to_100(int(rng.integers(_s_lo, _s_hi + 1)))
    fixed_plans.append({
        "payday":         1,
        "amount":         _store_rent,
        "category":       "기타",
        "description":    "상가 월세",
        "payment_method": TRANSFER_PAYMENT,
        "memo":           "자영업(고정)",
    })
    # 직원 월급 (확률적: 40% 직원 보유)
    if rng.random() < 0.40:
        _w_lo, _w_hi = {"tight": (1_800_000, 2_500_000), "normal": (2_200_000, 3_500_000),
                        "affluent": (3_000_000, 5_500_000)}[spending_profile]
        _wage = round_to_100(int(rng.integers(_w_lo, _w_hi + 1)))
        fixed_plans.append({
            "payday":         int(rng.integers(25, 29)),
            "amount":         _wage,
            "category":       "기타",
            "description":    "직원 급여",
            "payment_method": TRANSFER_PAYMENT,
            "memo":           "자영업(고정)",
        })


# =========================================================
# 12) 구독 계획
# =========================================================

SUB_PROB_SCALE = {"tight": 0.90, "normal": 1.00, "affluent": 1.10}
sub_scale      = SUB_PROB_SCALE[spending_profile]

subscription_catalog = [
    ("구독", "유튜브 프리미엄",  10_450, 0.82),
    ("구독", "넷플릭스",         17_000, 0.72),
    ("구독", "쿠팡 와우",         7_900, 0.68),
    ("구독", "티빙",             13_900, 0.25),
    ("구독", "웨이브",           10_900, 0.18),
    ("구독", "디즈니+",          13_900, 0.15),
    ("구독", "왓챠",              9_900, 0.10),
    ("구독", "애플뮤직",         10_900, 0.22),
    ("구독", "스포티파이",       10_900, 0.20),
    ("구독", "네이버 멤버십",     4_900, 0.25),
    ("구독", "우주패스",          9_900, 0.12),
    ("구독", "iCloud",            1_100, 0.28),
    ("구독", "ChatGPT Plus",     27_000, 0.08),
    ("구독", "Gemini",           27_000, 0.05),
]

SUB_LIMIT = {"tight": 4, "normal": 5, "affluent": 6}[spending_profile]
_sub_priority = {
    "유튜브 프리미엄": 100, "넷플릭스": 95, "쿠팡 와우": 90,
    "iCloud": 60, "네이버 멤버십": 55, "애플뮤직": 50, "스포티파이": 48,
    "티빙": 40, "웨이브": 35, "디즈니+": 30, "왓챠": 20,
    "우주패스": 18, "ChatGPT Plus": 15, "Gemini": 12,
}

def build_subscription_active_months(name: str, profile: str) -> set:
    core = name in ("유튜브 프리미엄", "넷플릭스", "쿠팡 와우")
    sm   = choose_start_month_with_seasonality()
    flen = int(rng.integers(5, 11)) if core else int(rng.integers(3, 9))
    first_months = set(month_range(sm, flen))
    active       = set(first_months)

    base_churn = {"tight": 0.30, "normal": 0.22, "affluent": 0.16}[profile]
    if core:
        base_churn *= 0.65
    if rng.random() < base_churn and first_months:
        gap         = int(rng.integers(1, 3))
        rejoin_start= max(first_months) + gap
        if rejoin_start <= 12:
            rlen   = int(rng.integers(3, 7)) if core else int(rng.integers(2, 6))
            rmonths= set(month_range(rejoin_start, rlen))
            base_r = {"tight": 0.75 * 0.92, "normal": 0.75, "affluent": 0.75 * 1.08}[profile]
            if rng.random() < min(0.95, base_r):
                active |= rmonths
    return active

picked_subs = []
for _cat, _name, _price, _bp in subscription_catalog:
    _p = min(0.95, _bp * sub_scale)
    if _name in ("ChatGPT Plus", "Gemini"):
        _p *= 0.60 if spending_profile == "tight" else (1.30 if spending_profile == "affluent" else 1.0)
    if rng.random() < _p:
        picked_subs.append((_cat, _name, _price))

_has_yt = any(n == "유튜브 프리미엄" for _, n, _ in picked_subs)
if _has_yt:
    _music = [(c, n, p) for c, n, p in picked_subs if n in ("애플뮤직", "스포티파이")]
    if len(_music) >= 2:
        picked_subs.remove(_music[int(rng.integers(0, 2))])
    _music = [(c, n, p) for c, n, p in picked_subs if n in ("애플뮤직", "스포티파이")]
    if len(_music) == 1 and rng.random() < 0.25:
        picked_subs.remove(_music[0])

picked_subs.sort(key=lambda x: _sub_priority.get(x[1], 1), reverse=True)
picked_subs = picked_subs[:SUB_LIMIT]

subscription_plans = []
for _cat, _name, _price in picked_subs:
    subscription_plans.append({
        "category":       _cat,
        "description":    _name,
        "payday":         SUB_PAYDAY,
        "months_by_year": {2024: build_subscription_active_months(_name, spending_profile),
                           2025: build_subscription_active_months(_name, spending_profile)},
        "amount":         round_to_100(_price),
        "memo":           "구독(해지/재가입 가능)",
    })


# =========================================================
# 13) 준고정비 (학원)
# =========================================================

AMOUNT_RANGE_ACADEMY = {
    "self_academy":  (120_000, 650_000),
    "child_academy": (250_000, 1_200_000),
}
FIXED_PAYDAY_ACADEMY = {
    "self_academy":  int(rng.integers(3, 10)),
    "child_academy": int(rng.integers(5, 11)),
}

optional_plans: list[dict] = []

def build_optional_active_months(profile: str) -> dict[int, set]:
    result = {}
    for y in (2024, 2025):
        sm        = choose_start_month_with_seasonality()
        planned   = int(rng.integers(3, 7))
        quit_p    = {"tight": 0.36, "normal": 0.26, "affluent": 0.18}[profile]
        actual    = int(rng.integers(1, planned)) if (rng.random() < quit_p and planned >= 2) else planned
        first     = set(month_range(sm, actual))
        active    = set(first)
        rejoin_p  = {"tight": 0.22, "normal": 0.24, "affluent": 0.20}[profile]
        if rng.random() < rejoin_p and first:
            gap  = int(rng.integers(1, 3))
            rs   = max(first) + gap
            if rs <= 12:
                rlen = int(rng.integers(2, 5))
                if rng.random() < 0.70:
                    active |= set(month_range(rs, rlen))
        result[y] = active
    return result

def add_optional_fixed_plan(key: str, category: str, desc: str,
                             base_prob: float, amount_mult: float = 1.0):
    if rng.random() >= min(0.95, float(base_prob)):
        return
    months_by_year = build_optional_active_months(spending_profile)
    lo, hi = AMOUNT_RANGE_ACADEMY[key]
    amt    = int(rng.integers(lo, hi + 1)) * amount_mult
    if key == "child_academy" and age_group in ("40s", "50s"):
        amt *= 1.35
    if age_group == "30s" and key == "self_academy":
        amt *= 1.10
    amt = round_to_100(amt * (1.00 + 0.08 * max(0, children_count - 1)))
    optional_plans.append({
        "key":            key,
        "category":       category,
        "description":    desc,
        "payday":         FIXED_PAYDAY_ACADEMY[key],
        "months_by_year": months_by_year,
        "amount":         int(amt),
        "memo":           "준고정비(기간제) | 해지/재가입 가능",
    })

_sa_prob = {"20s": 0.28, "30s": 0.34, "40s": 0.12, "50s": 0.06}[age_group]
if spending_profile == "tight":   _sa_prob *= 0.80
elif spending_profile == "affluent": _sa_prob *= 1.10
add_optional_fixed_plan("self_academy",  "교육", "학원비(본인)", _sa_prob)

if has_children:
    _ca_prob = {"20s": 0.20, "30s": 0.55, "40s": 0.82, "50s": 0.82}.get(age_group, 0.20)
    if spending_profile == "tight": _ca_prob *= 0.85
    add_optional_fixed_plan("child_academy", "교육", "학원비(자녀)", _ca_prob)


# =========================================================
# 14) 반려동물 정기
# =========================================================

def build_pet_supply_schedule(year: int) -> list:
    schedule = []
    for m in range(1, 13):
        k = int(rng.integers(1, 3))
        day_pool = np.array(list(range(2, 7)) + list(range(12, 17)) + list(range(23, 28)), dtype=int)
        for d in sorted(rng.choice(day_pool, size=k, replace=False)):
            try:
                schedule.append(pd.Timestamp(year=year, month=m, day=int(d)))
            except ValueError:
                pass
    return schedule

def sample_pet_supply_desc_and_amount(profile: str) -> tuple[str, int]:
    items = ["반려동물(사료/간식)", "반려동물(모래/배변패드)", "반려동물(장난감)"]
    probs = np.array([0.50, 0.35, 0.15], dtype=float)
    item  = str(rng.choice(items, p=probs))
    _ranges = {
        "반려동물(사료/간식)":    (25_000, 140_000) if profile == "affluent" else (18_000, 95_000),
        "반려동물(모래/배변패드)":(18_000, 110_000) if profile == "affluent" else (12_000, 70_000),
        "반려동물(장난감)":       (12_000, 85_000)  if profile == "affluent" else (8_000, 55_000),
    }
    lo, hi = _ranges[item]
    if profile == "tight": hi = int(hi * 0.85)
    amt = round_to_100(int(rng.integers(lo, hi + 1)))
    plat = str(rng.choice(["쿠팡", "네이버쇼핑", "당근마켓"],
                           p=[0.78, 0.20, 0.02] if profile == "tight" else [0.72, 0.23, 0.05]))
    return f"{plat} | {item} 구매", amt

pet_supply_days: list = []
if pet_owner:
    pet_supply_days = build_pet_supply_schedule(2024) + build_pet_supply_schedule(2025)


# =========================================================
# 15) 등록금 (학기 2회)
# =========================================================

TUITION_RANGE = {
    "tight":    (2_000_000, 3_200_000),
    "normal":   (2_200_000, 4_000_000),
    "affluent": (2_600_000, 4_800_000),
}
tuition_persona = "none"
p_tuition_self  = 0.0
p_tuition_child = 0.0

if age_group in ("20s", "30s"):
    p_tuition_self = 0.35 if age_group == "20s" else 0.18
if age_group in ("40s", "50s") and has_children:
    p_tuition_child = 0.45 if age_group == "40s" else 0.35

if spending_profile == "tight":
    p_tuition_self *= 0.75; p_tuition_child *= 0.85
elif spending_profile == "affluent":
    p_tuition_self *= 1.10; p_tuition_child *= 1.10

_r2 = rng.random()
if _r2 < p_tuition_self:
    tuition_persona = "self"
elif _r2 < p_tuition_self + p_tuition_child:
    tuition_persona = "child"

def pick_tuition_days_for_year(year: int) -> list:
    days = []
    for months in [(2, 3), (8, 9)]:
        m = int(rng.choice(list(months)))
        d = int(rng.integers(10, 26))
        try:
            days.append(pd.Timestamp(year=year, month=m, day=d))
        except ValueError:
            pass
    return days

tuition_days: set = set()
if tuition_persona != "none":
    tuition_days |= set(pick_tuition_days_for_year(2024))
    tuition_days |= set(pick_tuition_days_for_year(2025))


# =========================================================
# 16) 쇼핑 금액 규칙
# =========================================================

GIFT_RANGE     = {"tight": (10_000, 100_000),  "normal": (10_000, 300_000),  "affluent": (20_000, 800_000)}
CLOTH_RANGE    = {"tight": (25_000, 100_000),  "normal": (25_000, 150_000),  "affluent": (50_000, 400_000)}
CLOTH_WINTER_MAX = 250_000
SHOES_RANGE    = {"tight": (50_000, 150_000),  "normal": (50_000, 150_000),  "affluent": (50_000, 400_000)}
ACC_RANGE      = {"tight": (10_000, 50_000),   "normal": (10_000, 50_000),   "affluent": (20_000, 300_000)}
COSMETIC_RANGE = {"tight": (10_000, 30_000),   "normal": (10_000, 60_000),   "affluent": (20_000, 100_000)}
SECONDHAND_RANGE={"tight": (3_000, 100_000),   "normal": (50_000, 300_000),  "affluent": None}


# =========================================================
# 17) 의료 후속 (돌발 → 약국)
# =========================================================

def maybe_add_pharmacy_followup(rows: list, d: pd.Timestamp,
                                 hospital_amount: int, profile: str,
                                 reason: str = "돌발 진료(급성)"):
    base_p = 0.80
    if profile == "tight":    base_p *= 0.92
    elif profile == "affluent": base_p *= 1.05
    p = float(np.clip(base_p, 0.35, 0.95))
    if rng.random() >= p:
        return
    follow_day = d if rng.random() < 0.85 else (d + pd.Timedelta(days=1))
    if follow_day < dates.min() or follow_day > dates.max():
        return
    ratio = rng.uniform(0.20, 0.65)
    cap   = {"tight": 90_000, "normal": 130_000, "affluent": 180_000}[profile]
    amt   = round_to_100(max(3_000, min(cap, int(hospital_amount * ratio))))
    rows.append(make_row(follow_day, amt, "의료/건강", "약국",
                         DEFAULT_PAYMENT, False, f"병원 후속(약국) | {reason}"))


# =========================================================
# 18) 해외 여행 / 출장 스케줄
# =========================================================

# 숙박 금액 최솟값 방어 (여행 시나리오에서만 사용)
LODGING_MIN = {
    "tight":    80_000,
    "normal":   120_000,
    "affluent": 200_000,
}

# 여행 여부 결정
TRAVEL_ABROAD_PROB = {
    "tight":    0.15,
    "normal":   0.35,
    "affluent": 0.55,
}
# 출장: 직장인 전용
BUSINESS_TRIP_PROB = {"tight": 0.10, "normal": 0.20, "affluent": 0.18}

travel_rows: list[dict] = []  # 항공권 사전 구매 행 별도 관리

def build_abroad_travel(year: int):
    """
    해외 여행 1회 시뮬레이션.
    - 항공권을 출발 30~90일 전에 사전 결제
    - 여행 기간 중 식비·숙박·문화비 생성
    - 가족 여행이면 인원 배수 적용
    """
    prob = TRAVEL_ABROAD_PROB[spending_profile]
    if has_children and age_group in ("30s", "40s"):
        prob *= 1.15
    if rng.random() >= prob:
        return

    # 여행 출발일: 봄/여름/겨울방학 시즌
    candidate_months = [3, 4, 7, 8, 12]
    travel_month = int(rng.choice(candidate_months))
    travel_day   = int(rng.integers(5, 25))
    try:
        depart_date = pd.Timestamp(year=year, month=travel_month, day=travel_day)
    except ValueError:
        return

    if depart_date < dates.min() or depart_date > dates.max():
        return

    trip_nights = int(rng.integers(3, 8))  # 3~7박

    # 인원수 결정
    if has_children and age_group in ("30s", "40s", "50s"):
        travel_headcount = 2 + children_count  # 본인+배우자+자녀
        is_family_travel = True
    else:
        travel_headcount = 1 if age_group == "20s" else 2
        is_family_travel = False

    # --- 항공권 사전 구매 (30~90일 전)
    advance_days = int(rng.integers(30, 91))
    purchase_date = depart_date - pd.Timedelta(days=advance_days)
    if purchase_date < dates.min():
        purchase_date = dates.min()

    _flight_lo, _flight_hi = {
        "tight":    (150_000, 450_000),
        "normal":   (300_000, 900_000),
        "affluent": (600_000, 2_000_000),
    }[spending_profile]
    flight_per_person = round_to_100(int(rng.integers(_flight_lo, _flight_hi + 1)))
    flight_total      = flight_per_person * travel_headcount

    travel_rows.append(make_row(
        purchase_date, flight_total, "교통비", "항공권",
        DEFAULT_PAYMENT, False,
        f"해외여행 항공권 사전구매 | {'가족' if is_family_travel else '개인'} {travel_headcount}인",
    ))

    # --- 여행 기간 중 지출
    for n in range(trip_nights + 1):
        travel_d = depart_date + pd.Timedelta(days=n)
        if travel_d > dates.max():
            break

        # 숙박비 (매일) - LODGING_MIN 최솟값 방어
        _hotel_lo, _hotel_hi = {
            "tight":    (80_000, 180_000),     # 최솟값 80,000원 보장
            "normal":   (120_000, 350_000),
            "affluent": (250_000, 1_000_000),
        }[spending_profile]
        _hotel_per = max(
            LODGING_MIN[spending_profile],
            round_to_100(int(rng.integers(_hotel_lo, _hotel_hi + 1)))
        )
        hotel_amt = _hotel_per * travel_headcount
        travel_rows.append(make_row(
            travel_d, hotel_amt, "문화/여가", "호텔 숙박비",
            DEFAULT_PAYMENT, False,
            f"해외여행({'가족' if is_family_travel else '개인'})",
        ))

        # 식비
        _food_lo, _food_hi = {
            "tight":    (20_000, 60_000),
            "normal":   (40_000, 100_000),
            "affluent": (80_000, 250_000),
        }[spending_profile]
        food_amt = round_to_100(int(rng.integers(_food_lo, _food_hi + 1)) * travel_headcount)
        travel_rows.append(make_row(
            travel_d, food_amt, "식비", "점심 식사",
            DEFAULT_PAYMENT, False,
            f"해외여행 식비({'가족' if is_family_travel else '개인'})",
        ))

        # 문화/관광 (확률적)
        if rng.random() < 0.55:
            _culture_amt = round_to_100(
                int(rng.integers(20_000, 120_001)) * travel_headcount
            )
            travel_rows.append(make_row(
                travel_d, _culture_amt, "문화/여가", "입장권/티켓",
                DEFAULT_PAYMENT, False,
                f"해외여행 관광({'가족' if is_family_travel else '개인'})",
            ))


def build_business_trip(year: int):
    """
    출장 시뮬레이션 (직장인 전용).
    - 식비·교통비 비중 높음, 쇼핑 없음
    - 법인카드 결제 확률 적용
    """
    if employment_status != "salaryman":
        return
    if rng.random() >= BUSINESS_TRIP_PROB[spending_profile]:
        return

    # 분기당 최대 1회, 1~3박
    trip_months = int(rng.choice([3, 6, 9, 11]))
    trip_day    = int(rng.integers(5, 22))
    try:
        depart_date = pd.Timestamp(year=year, month=trip_months, day=trip_day)
    except ValueError:
        return
    if depart_date < dates.min() or depart_date > dates.max():
        return

    nights = int(rng.integers(1, 4))
    corp_card_prob = 0.65  # 법인카드 확률

    # KTX/항공 왕복
    use_flight  = rng.random() < 0.30
    trans_desc  = "항공권" if use_flight else "KTX/기차"
    _t_lo, _t_hi= (100_000, 400_000) if use_flight else (50_000, 150_000)
    trans_amt   = round_to_100(int(rng.integers(_t_lo, _t_hi + 1)) * 2)  # 왕복
    t_pm        = CORPORATE_CARD if rng.random() < corp_card_prob else DEFAULT_PAYMENT
    travel_rows.append(make_row(
        depart_date, trans_amt, "교통비", trans_desc, t_pm, False, "출장(교통)",
    ))

    for n in range(nights + 1):
        t_d = depart_date + pd.Timedelta(days=n)
        if t_d > dates.max():
            break

        # 숙박
        _h_pm  = CORPORATE_CARD if rng.random() < corp_card_prob else DEFAULT_PAYMENT
        _h_amt = round_to_100(int(rng.integers(80_000, 200_001)))
        travel_rows.append(make_row(
            t_d, _h_amt, "문화/여가", "호텔 숙박비", _h_pm, False, "출장(숙박)",
        ))

        # 식비 (회식 포함)
        _f_pm  = CORPORATE_CARD if rng.random() < corp_card_prob else DEFAULT_PAYMENT
        _f_amt = round_to_100(int(rng.integers(15_000, 80_001)))
        travel_rows.append(make_row(
            t_d, _f_amt, "식비", "점심 식사", _f_pm, False, "출장(식비)",
        ))

# 연도별 여행·출장 생성
for _y in (2024, 2025):
    build_abroad_travel(_y)
    build_business_trip(_y)


# =========================================================
# 19) 이벤트성 고액
# =========================================================

# =========================================================
# 전자기기 PRICE_TABLE (기기군 × 브랜드 × 프로파일)
# =========================================================

# 브랜드 가중치: 연령대별 Apple/Samsung 선호 반영
def _device_brand_weights() -> tuple[list, np.ndarray]:
    """(brands, probs) 반환. 20s/30s → Apple↑, 40s/50s → Samsung↑"""
    brands = ["apple", "samsung"]
    if age_group in ("20s", "30s"):
        p = np.array([0.62, 0.38])
    elif age_group in ("40s", "50s"):
        p = np.array([0.35, 0.65])
    else:
        p = np.array([0.50, 0.50])
    return brands, p / p.sum()


# (group, brand) → (lo, hi) 기본 금액 구간 (원)
# profile 보정은 ±10~15% 스케일만 적용 (과도한 흔들림 방지)
_DEVICE_PRICE_TABLE: dict[tuple, tuple] = {
    # 스마트폰
    ("phone", "apple"):   (1_200_000, 1_800_000),   # 아이폰 구매
    ("phone", "samsung"): (  900_000, 1_500_000),   # 갤럭시폰 구매
    # 스마트워치
    ("watch", "apple"):   (  450_000,   800_000),   # 애플워치 구매
    ("watch", "samsung"): (  350_000,   600_000),   # 갤럭시 워치 구매
    # 태블릿
    ("tablet", "apple"):  (  800_000, 1_500_000),   # 아이패드 구매
    ("tablet", "samsung"):(  500_000, 1_100_000),   # 갤럭시탭 구매
    # 노트북
    ("laptop", "apple"):  (1_800_000, 3_500_000),   # 맥북 구매
    ("laptop", "samsung"):(1_200_000, 2_300_000),   # 갤럭시북 구매
    # PC/데스크탑
    ("pc", "apple"):      (1_800_000, 3_000_000),   # iMac/Mac mini 구매
    ("pc", "samsung"):    (  800_000, 2_000_000),   # 데스크탑 구매
    # 기타(에어팟/버즈/소형가전) - 하한 200,000원 강제
    ("other", "apple"):   (  250_000,   450_000),   # 에어팟 구매
    ("other", "samsung"): (  200_000,   380_000),   # 갤럭시 버즈 구매
}

# 프로파일별 스케일 (과도하지 않게 ±12% 정도)
_DEVICE_PROFILE_SCALE = {"tight": 0.88, "normal": 1.00, "affluent": 1.12}

# 기기군별 쿨다운 (일 수) - 2년(730일) 내 중복 방지 목적
_DEVICE_COOLDOWN_DAYS: dict[str, int] = {
    "laptop": 900,   # 사실상 2년 내 1회 이하
    "pc":     1100,  # 3년 주기
    "phone":  700,   # 약 2년
    "tablet": 900,
    "watch":  730,
    "other":  180,   # 6개월 (버즈류는 비교적 자주)
}

# 기기군별 발생 확률 (이벤트 전자기기 선택 시)
_DEVICE_GROUP_PROBS_BASE = np.array([0.30, 0.25, 0.12, 0.12, 0.08, 0.13], dtype=float)
_DEVICE_GROUPS = ["phone", "watch", "other", "laptop", "pc", "tablet"]

# 전자기기 구매 이력 추적 (쿨다운)
_device_last_purchase: dict[str, pd.Timestamp | None] = {g: None for g in _DEVICE_GROUPS}


def sample_device_event(candidate_date: pd.Timestamp) -> tuple[str, str, int] | None:
    """
    전자기기 이벤트 1건을 샘플링한다.
    쿨다운 미충족 시 None 반환.

    Returns
    -------
    (description, memo_label, amount) | None
    """
    brands, brand_w = _device_brand_weights()

    # 발생 가능한 기기군 (쿨다운 통과)
    eligible = []
    eligible_p = []
    for i, group in enumerate(_DEVICE_GROUPS):
        last = _device_last_purchase.get(group)
        cooldown = _DEVICE_COOLDOWN_DAYS[group]
        if last is None or (candidate_date - last).days >= cooldown:
            eligible.append(group)
            eligible_p.append(_DEVICE_GROUP_PROBS_BASE[i])

    if not eligible:
        return None

    probs = np.array(eligible_p, dtype=float)
    probs /= probs.sum()
    group = str(rng.choice(eligible, p=probs))

    brand = str(rng.choice(brands, p=brand_w))
    lo, hi = _DEVICE_PRICE_TABLE[(group, brand)]
    scale  = _DEVICE_PROFILE_SCALE[spending_profile]
    lo_s   = int(lo * scale)
    hi_s   = int(hi * scale)
    amt    = round_to_100(int(rng.integers(lo_s, hi_s + 1)))

    # description 현실화
    _desc_map = {
        ("phone", "apple"):    "아이폰 구매",
        ("phone", "samsung"):  "갤럭시폰 구매",
        ("watch", "apple"):    "애플워치 구매",
        ("watch", "samsung"):  "갤럭시 워치 구매",
        ("tablet", "apple"):   "아이패드 구매",
        ("tablet", "samsung"): "갤럭시탭 구매",
        ("laptop", "apple"):   "맥북 구매",
        ("laptop", "samsung"): "갤럭시북 구매",
        ("pc", "apple"):       "맥 데스크탑 구매",
        ("pc", "samsung"):     "데스크탑 구매",
        ("other", "apple"):    "에어팟 구매",
        ("other", "samsung"):  "갤럭시 버즈 구매",
    }
    desc = _desc_map.get((group, brand), "전자기기 구매")

    # 쿨다운 갱신
    _device_last_purchase[group] = candidate_date

    return desc, f"이벤트성 지출 | 전자기기({group})", amt


# 기타 이벤트성 지출 범위
EVENT_RANGE_NON_DEVICE = {
    "경조사": {
        "tight":    (50_000, 100_000),
        "normal":   (50_000, 100_000),
        "affluent": (50_000, 300_000),
    },
    "항공권": {
        "tight":    (100_000, 200_000),
        "normal":   (100_000, 800_000),
        "affluent": (300_000, 2_000_000),
    },
}



def add_event_spikes(year: int, rows: list):
    """
    이벤트성 고액 지출 생성.
    - 전자기기: sample_device_event() → 쿨다운 + PRICE_TABLE
    - 경조사/항공권: 기존 범위 유지
    - 리조트 숙박비: 이벤트에서 제거 (여행 시나리오에서만 발생)
    """
    year_dates       = dates[dates.year == year]
    event_candidates = year_dates[year_dates.month.isin([3, 6, 9, 12])]
    event_n          = {"tight": 4, "normal": 5, "affluent": 6}[spending_profile]
    if len(event_candidates) == 0:
        return
    event_days = pd.to_datetime(
        rng.choice(event_candidates, size=min(event_n, len(event_candidates)), replace=False)
    )

    # 비(非)전자기기 이벤트 풀
    non_device_items = list(EVENT_RANGE_NON_DEVICE.keys())
    # 전자기기 이벤트 비중: tight 30%, normal 40%, affluent 45%
    device_prob = {"tight": 0.30, "normal": 0.40, "affluent": 0.45}[spending_profile]

    for day in sorted(event_days):
        if rng.random() < device_prob:
            # 전자기기 이벤트 시도
            result = sample_device_event(day)
            if result is not None:
                desc, memo, amt = result
                rows.append(make_row(day, amt, "쇼핑", desc, DEFAULT_PAYMENT, False, memo))
                continue
            # 쿨다운 실패 시 비장치 이벤트로 폴백

        # 비전자기기 이벤트
        ev_desc = str(rng.choice(non_device_items))
        lo, hi  = EVENT_RANGE_NON_DEVICE[ev_desc][spending_profile]
        amt     = round_to_100(int(rng.integers(lo, hi + 1)))
        cat     = "기타" if ev_desc == "경조사" else "교통비"
        rows.append(make_row(day, amt, cat, ev_desc, DEFAULT_PAYMENT, False, "이벤트성 지출"))


# =========================================================
# 20) 데이터 생성 메인 루프
# =========================================================

rows: list[dict] = []

profile_base_probs = profile_category_bias(spending_profile, transport_mode)

for d in dates:
    year  = int(d.year)
    month = int(d.month)
    wd    = int(d.weekday())

    # --- 20-1) 고정비 (dict 구조)
    for plan in fixed_plans:
        if d.day == plan["payday"]:
            rows.append(make_row(
                d, plan["amount"], plan["category"],
                plan["description"], plan["payment_method"],
                True, plan["memo"],
            ))

    # --- 20-2) 구독
    for plan in subscription_plans:
        if d.day == plan["payday"] and month in plan["months_by_year"][year]:
            rows.append(make_row(
                d, plan["amount"], plan["category"],
                plan["description"], DEFAULT_PAYMENT,
                True, plan["memo"],
            ))

    # --- 20-3) 준고정비 (학원)
    for plan in optional_plans:
        if d.day == plan["payday"] and month in plan["months_by_year"][year]:
            rows.append(make_row(
                d, plan["amount"], plan["category"],
                plan["description"], DEFAULT_PAYMENT,
                True, plan["memo"],
            ))

# --- 20-4) 등록금
if tuition_persona != "none":
    lo, hi = TUITION_RANGE[spending_profile]
    for day in sorted(tuition_days):
        amt = int(rng.integers(lo, hi + 1))
        if tuition_persona == "child":
            amt = int(amt * (1.05 + 0.06 * max(0, children_count - 1)))
        rows.append(make_row(
            day, amt, "교육", "등록금", DEFAULT_PAYMENT, False,
            f"등록금(학기) | {'본인' if tuition_persona == 'self' else '자녀'}",
        ))

# --- 20-5) 반려동물 정기
if pet_owner:
    for day in pet_supply_days:
        desc, amt = sample_pet_supply_desc_and_amount(spending_profile)
        pm = TRANSFER_PAYMENT if "당근마켓" in desc else DEFAULT_PAYMENT
        rows.append(make_row(day, amt, "쇼핑", desc, pm, False, "반려동물 정기(월 1~2회)"))

# --- 20-6) 교정 / 검진 / 미용 / 네일 / 피부
for d in dates:
    if d in ortho_days:
        _lo, _hi = (50_000, 140_000) if spending_profile == "affluent" else (30_000, 90_000)
        mult = 1.0
        if ortho_type == "child":
            mult *= 1.15 * (1.00 + 0.06 * max(0, children_count - 1))
        rows.append(make_row(
            d, round_to_100(int(rng.integers(_lo, _hi + 1)) * mult),
            "의료/건강", "치과", DEFAULT_PAYMENT, False,
            "치과 교정(월1) | " + ("본인" if ortho_type == "self" else "자녀"),
        ))

    if d in checkup_days:
        rows.append(make_row(
            d, round_to_100(int(rng.integers(80_000, 250_000))),
            "의료/건강", "병원 진료", DEFAULT_PAYMENT, False, "정기 건강검진(연1회)",
        ))

    if d in dental_check_days:
        lo, hi = DENTAL_CHECK_RANGE
        rows.append(make_row(
            d, round_to_100(int(rng.integers(lo, hi + 1))),
            "의료/건강", "치과", DEFAULT_PAYMENT, False, "치과 정기검진(연1회)",
        ))

    if d in hair_days:
        if gender == "male":
            _h_amt = round_to_100(int(rng.integers(30_000, 50_001)))
            _h_memo = "미용실(월1)"
        else:
            _lo, _hi = (60_000, 260_000) if spending_profile == "affluent" else (45_000, 180_000)
            if spending_profile == "tight": _hi = int(_hi * 0.85)
            _h_amt  = round_to_100(int(rng.integers(_lo, _hi + 1)))
            _h_memo = "미용실(2달당 최대1회)"
        rows.append(make_row(d, _h_amt, "기타", "미용실", DEFAULT_PAYMENT, False, _h_memo))

    if d in nail_days and NAIL_RANGE[spending_profile] is not None:
        lo, hi = NAIL_RANGE[spending_profile]
        rows.append(make_row(
            d, round_to_100(int(rng.integers(lo, hi + 1))),
            "기타", "네일/왁싱", DEFAULT_PAYMENT, False, "네일(주기)",
        ))

    if d in skin_days and SKINCARE_RANGE[spending_profile] is not None:
        lo, hi = SKINCARE_RANGE[spending_profile]
        rows.append(make_row(
            d, round_to_100(int(rng.integers(lo, hi + 1))),
            "기타", "피부관리", DEFAULT_PAYMENT, False, "피부관리(주기)",
        ))

    if d in derm_proc_days and DERM_PROC_RANGE[spending_profile] is not None:
        lo, hi = DERM_PROC_RANGE[spending_profile]
        rows.append(make_row(
            d, round_to_100(int(rng.integers(lo, hi + 1))),
            "의료/건강", "병원 진료", DEFAULT_PAYMENT, False, "피부과(시술)",
        ))

# --- 20-7) 변동지출 메인 루프
for d in dates:
    wd    = int(d.weekday())
    year  = int(d.year)
    month = int(d.month)

    is_grocery_day = (d in grocery_days)

    lam = lambda_by_weekday[wd]
    if month == 12:   lam *= 1.10
    elif month in (1, 2): lam *= 0.95
    lam *= holiday_factor(d)
    lam *= PROFILE_LAMBDA_SCALE[spending_profile]
    lam *= TOTAL_LAMBDA_MULT

    n_tx  = int(rng.poisson(lam))
    probs = adjust_category_probs(d, wd, profile_base_probs)
    chosen = rng.choice(var_categories, size=n_tx, replace=True, p=probs)

    for cat in chosen:
        memo = ""
        desc = None
        pm   = DEFAULT_PAYMENT

        # 식비: desc 먼저 결정 후 금액 계산
        if cat == "식비":
            desc = sample_food_description(wd, spending_profile, is_grocery_day)
            amt  = generate_amount(cat, spending_profile, desc=desc)
            # 장보기 금액
            if desc == "장보기(마트)":
                if gender == "female" and age_group in ("30s", "40s", "50s"):
                    _lo, _hi = {"tight": (40_000, 180_000), "normal": (55_000, 300_000),
                                "affluent": (80_000, 420_000)}[spending_profile]
                else:
                    _lo, _hi = (25_000, 160_000) if spending_profile == "affluent" else (18_000, 120_000)
                mult = 1.00
                if has_children:
                    mult *= 1.18 + 0.10 * max(0, children_count - 1)
                    if age_group in ("40s", "50s"): mult *= 1.05
                amt = round_to_100(int(rng.integers(_lo, _hi + 1)) * mult)

            # 카페 금액
            elif any(b in desc for b in cafe_brands):
                if any(k in desc for k in ["메가", "컴포즈", "더벤티", "매머드"]):
                    _bmin, _bmax = 2500, 12000
                elif any(k in desc for k in ["이디야", "커피베이", "공차"]):
                    _bmin, _bmax = 3500, 18000
                elif any(k in desc for k in ["스타벅스", "투썸", "파스쿠찌"]):
                    _bmin, _bmax = 4000, 25000
                else:
                    _bmin, _bmax = 3000, 28000
                if "음료+디저트" in desc:
                    _bmin = int(_bmin * 1.2); _bmax = int(_bmax * 1.6)
                _bmin = int(_bmin * PROFILE_AMOUNT_SCALE[spending_profile] * AGE_AMOUNT_MULT)
                _bmax = int(_bmax * PROFILE_AMOUNT_SCALE[spending_profile] * AGE_AMOUNT_MULT)
                amt   = round_to_100(int(rng.integers(_bmin, _bmax + 1)))

            # 자녀 가구 외식 배수
            if has_children and desc in ("저녁 외식", "배달 음식", "점심 식사"):
                bump = 1.10 + 0.06 * max(0, children_count - 1)
                if age_group in ("40s", "50s"): bump *= 1.06
                amt = round_to_100(amt * bump)

        # 교통비
        elif cat == "교통비":
            desc = sample_transport_description(transport_mode, spending_profile)
            amt  = generate_amount(cat, spending_profile, desc=desc)
            if desc in ("지하철 교통카드", "버스"):
                amt = round_to_100(min(80_000, max(1_200, amt)))
            elif desc == "택시":
                amt = round_to_100(min(TAXI_MAX[spending_profile], max(5_000, amt)))
            elif desc in ("주유", "정비/오일"):
                cap = {"tight": 200_000, "normal": 250_000, "affluent": 340_000}[spending_profile]
                amt = round_to_100(min(cap, max(30_000, amt)))
            elif desc in ("주차", "통행료", "세차"):
                cap = {"tight": 90_000, "normal": 120_000, "affluent": 160_000}[spending_profile]
                amt = round_to_100(min(cap, max(3_000, amt)))
            elif desc == "KTX/기차":
                lo, hi = TRAIN_RANGE
                amt    = round_to_100(int(rng.integers(lo, hi + 1)))
            elif desc == "항공권":
                cap = {"tight": 900_000, "normal": 1_300_000, "affluent": 1_700_000}[spending_profile]
                amt = round_to_100(min(cap, max(120_000, amt)))

        # 쇼핑
        elif cat == "쇼핑":
            desc = sample_shopping_description(spending_profile)
            if gender == "male" and "| 화장품 구매" in desc:
                continue
            is_secondhand = "중고" in desc
            if is_secondhand:
                pm = TRANSFER_PAYMENT
                if SECONDHAND_RANGE[spending_profile] is None:
                    continue
                lo, hi = SECONDHAND_RANGE[spending_profile]
                amt = round_to_100(int(rng.integers(lo, hi + 1)))
            else:
                if "| 선물 구매" in desc:
                    lo, hi = GIFT_RANGE[spending_profile]
                    amt = round_to_100(int(rng.integers(lo, hi + 1)))
                elif "| 의류 구매" in desc:
                    lo, hi = CLOTH_RANGE[spending_profile]
                    if spending_profile == "tight" and month in (11, 12, 1, 2):
                        hi = max(hi, CLOTH_WINTER_MAX)
                    amt = round_to_100(int(rng.integers(lo, hi + 1)))
                elif "| 신발 구매" in desc:
                    lo, hi = SHOES_RANGE[spending_profile]
                    amt = round_to_100(int(rng.integers(lo, hi + 1)))
                elif "| 잡화 구매" in desc:
                    lo, hi = ACC_RANGE[spending_profile]
                    amt = round_to_100(int(rng.integers(lo, hi + 1)))
                elif "| 화장품 구매" in desc:
                    lo, hi = COSMETIC_RANGE[spending_profile]
                    amt = round_to_100(int(rng.integers(lo, hi + 1)))
                elif "| 생필품 구매" in desc:
                    cap = 220_000 if spending_profile == "affluent" else 200_000
                    amt = round_to_100(min(cap, max(5_000, amt)))
                else:
                    amt = round_to_100(min(300_000, max(5_000, amt)))

        # 교육
        elif cat == "교육":
            desc = str(rng.choice(desc_pool["교육_변동"]))
            amt  = round_to_100(min(200_000, max(3_000, amt)))

        # 의료/건강
        elif cat == "의료/건강":
            acute_p = 0.30
            if (d in ortho_days) or (d in checkup_days) or (d in dental_check_days):
                acute_p *= 0.25
            if pet_owner and rng.random() < 0.10:
                desc = "동물병원"
                _lo, _hi = (30_000, 450_000) if spending_profile == "affluent" else (20_000, 250_000)
                amt  = round_to_100(int(rng.integers(_lo, _hi + 1)))
            elif rng.random() < acute_p:
                desc = "병원 진료"
                _lo, _hi = (10_000, 110_000) if spending_profile == "affluent" else (8_000, 70_000)
                amt  = round_to_100(int(rng.integers(_lo, _hi + 1)))
                memo = "돌발 진료(급성)"
                maybe_add_pharmacy_followup(rows, d, amt, spending_profile)
            else:
                _pool = [x for x in desc_pool["의료/건강"] if x != "동물병원"]
                desc  = str(rng.choice(_pool))

        # 문화/여가 (덕질 + 취미)
        elif cat == "문화/여가":
            if fandom_persona and (month in fandom_active_months_by_year[year]):
                target = fandom_target_by_month_by_year[year][month]
                if (month in concert_months_by_year[year]) and (d.day == int(rng.integers(8, 23))):
                    desc = f"덕질(콘서트/팬미팅) | {target}"
                    lo, hi = FANDOM_CONCERT[spending_profile]
                    amt  = round_to_100(int(rng.integers(lo, hi + 1)))
                    memo = "덕질 시즌(콘서트)"
                else:
                    p_small = {"20s": 0.20, "30s": 0.16}.get(age_group, 0.10)
                    if rng.random() < p_small:
                        base = str(rng.choice(["덕질(굿즈 구매)", "덕질(앨범 구매)", "덕질(팝업스토어)"]))
                        desc = f"{base} | {target}"
                        lo, hi = FANDOM_SMALL[spending_profile]
                        amt  = round_to_100(int(rng.integers(lo, hi + 1)))
                        memo = "덕질(소액 반복)"

            if hobby_intensity != "none" and (month in hobby_active_months_by_year[year]):
                p_h = 0.22 if hobby_intensity == "light" else 0.45
                if rng.random() < p_h:
                    desc = sample_hobby_purchase_desc(hobby_type)
                    amt  = sample_hobby_amount(hobby_type, hobby_intensity)
                    memo = f"취미({hobby_type}|{hobby_intensity})"
                    if desc.startswith("중고거래"):
                        pm = TRANSFER_PAYMENT

            if desc is None:
                desc = str(rng.choice(desc_pool["문화/여가"]))
                amt  = round_to_100(min(250_000, max(10_000, amt)))

        # 기타
        else:
            pool = desc_pool["기타"].copy()
            if age_group == "20s":
                pool = [x for x in pool if x != "명절 선물/용돈"]
            desc = str(rng.choice(pool))
            if desc == "명절 선물/용돈":
                hf     = holiday_factor(d)
                p_holi = 0.25 if hf > 1.0 else 0.05
                if rng.random() >= p_holi:
                    _alt  = [x for x in pool if x != "명절 선물/용돈"]
                    desc  = str(rng.choice(_alt)) if _alt else "기타 지출"
                    amt   = round_to_100(min(200_000, max(3_000, amt)))
                else:
                    base  = int(rng.integers(30_000, 260_000))
                    mult  = 1.00
                    if spending_profile == "affluent": mult *= 1.25
                    if age_group in ("40s", "50s"):    mult *= 1.12
                    if has_children: mult *= 1.05 + 0.05 * max(0, children_count - 1)
                    amt   = round_to_100(base * mult)
                    pm    = TRANSFER_PAYMENT
            else:
                amt = round_to_100(min(200_000, max(3_000, amt)))

        rows.append(make_row(d, amt, cat, desc, pm, False, memo))

# --- 20-8) 이벤트성 고액
add_event_spikes(2024, rows)
add_event_spikes(2025, rows)

# --- 20-9) 여행·출장 행 병합
rows.extend(travel_rows)


# =========================================================
# 21) 월 소프트캡 normalize (폭주 방지)
# =========================================================

def normalize_monthly_cap(df: pd.DataFrame) -> pd.DataFrame:
    """
    월별 총지출이 소프트캡/하드캡을 초과하지 않도록 조정한다.

    설계 원칙:
    - is_fixed=True 항목은 절대 건드리지 않음
    - 1단계: 일상 변동비(이벤트 아닌 항목) 비례 축소
    - 2단계: 그래도 초과 시 이벤트성 지출도 최대 0.75배까지만 축소
    - 이벤트 달(전자기기/항공권 포함)은 캡을 약간 완화 (+20%)

    월 목표치 (원):
        tight:    2,500,000 (±500,000)
        normal:   4,000,000 (±800,000)
        affluent: 6,000,000 (±1,200,000)
    """
    TARGET_CENTER = {"tight": 2_500_000, "normal": 4_000_000, "affluent": 6_000_000}
    TARGET_STD    = {"tight":   500_000, "normal":   800_000, "affluent": 1_200_000}
    SOFT_CAP_MULT = 1.10   # target의 110% = soft cap
    HARD_CAP_MULT = 1.40   # target의 140% = hard cap

    # 이벤트성 지출 판정 키워드
    EVENT_MEMO_KEYWORDS = ["이벤트성 지출", "해외여행", "출장"]

    rows_out = []
    center = TARGET_CENTER[spending_profile]
    std    = TARGET_STD[spending_profile]

    for ym, grp in df.groupby(df["date"].dt.strftime("%Y-%m")):
        # 월별 목표 샘플 (정규분포, 음수 방지)
        monthly_target = float(np.clip(
            rng.normal(center, std), center * 0.6, center * 1.6
        ))

        # 이벤트 달 여부 (캡 완화)
        has_event = grp["memo"].str.contains("|".join(EVENT_MEMO_KEYWORDS), na=False).any()
        boost     = 1.20 if has_event else 1.00
        soft_cap  = monthly_target * SOFT_CAP_MULT * boost
        hard_cap  = monthly_target * HARD_CAP_MULT * boost

        total = grp["amount"].sum()

        if total <= soft_cap:
            rows_out.append(grp)
            continue

        # 초과량
        excess = total - hard_cap
        if excess <= 0:
            # soft~hard 사이: 그냥 통과
            rows_out.append(grp)
            continue

        # 고정비 분리
        fixed_mask  = grp["is_fixed"] == True
        event_mask  = grp["memo"].str.contains("|".join(EVENT_MEMO_KEYWORDS), na=False) & ~fixed_mask
        var_mask    = ~fixed_mask & ~event_mask

        fixed_sum = grp.loc[fixed_mask, "amount"].sum()
        event_sum = grp.loc[event_mask, "amount"].sum()
        var_sum   = grp.loc[var_mask,   "amount"].sum()

        # 1단계: 변동비 비례 축소
        if var_sum > 0:
            needed_cut = min(excess, var_sum * 0.60)  # 최대 60% 삭감
            scale_var  = max(0.40, 1.0 - needed_cut / var_sum)
            grp = grp.copy()
            grp.loc[var_mask, "amount"] = (
                grp.loc[var_mask, "amount"] * scale_var
            ).round(-2).astype(int).clip(lower=100)
            excess = grp["amount"].sum() - hard_cap

        # 2단계: 이벤트 지출 최대 0.75배까지 축소
        if excess > 0 and event_sum > 0:
            scale_evt = max(0.75, 1.0 - excess / event_sum)
            grp.loc[event_mask, "amount"] = (
                grp.loc[event_mask, "amount"] * scale_evt
            ).round(-2).astype(int).clip(lower=100)

        rows_out.append(grp)

    result = pd.concat(rows_out, ignore_index=True)
    # 100원 단위 재보정
    result["amount"] = (result["amount"] / 100).round().astype(int) * 100
    result["amount"] = result["amount"].clip(lower=100)
    return result


# =========================================================
# 22) DataFrame 생성 및 품질 검증
# =========================================================

df = pd.DataFrame(rows)
df["date"]           = pd.to_datetime(df["date"])
df["amount"]         = pd.to_numeric(df["amount"], errors="coerce").astype(int)
df["category"]       = df["category"].astype(str)
df["description"]    = df["description"].astype(str)
df["payment_method"] = df["payment_method"].astype(str)
df["is_fixed"]       = df["is_fixed"].astype(bool)
df["memo"]           = df["memo"].astype(str)

df = df.sort_values("date").reset_index(drop=True)

# 월 소프트캡 normalize 적용
df = normalize_monthly_cap(df)
df = df.sort_values("date").reset_index(drop=True)

# 품질 검증
assert df["date"].isna().sum()   == 0, "date 결측값 존재"
assert df["amount"].isna().sum() == 0, "amount 결측값 존재"
assert (df["amount"] > 0).all(),        "음수/0 금액 존재"
assert (df["amount"] % 100 == 0).all(), "100원 단위 아닌 금액 존재"
assert set(df.columns) == {
    "date", "amount", "category", "description",
    "payment_method", "is_fixed", "memo"
}, "컬럼 스키마 불일치"


# =========================================================
# 22) 저장
# =========================================================

meta = {
    "SEED":                SEED,
    "spending_profile":    spending_profile,
    "affluent_type":       affluent_type,
    "transport_mode":      transport_mode,
    "gender":              gender,
    "age_group":           age_group,
    "employment_status":   employment_status,
    "has_children":        bool(has_children),
    "children_count":      int(children_count),
    "pet_owner":           bool(pet_owner),
    "is_self_employed":    bool(is_self_employed),
    "has_nursing_expense": bool(has_nursing_expense),
    "has_chronic_disease": bool(has_chronic_disease),
    "TOTAL_AMOUNT_MULT":   float(TOTAL_AMOUNT_MULT),
    "TOTAL_LAMBDA_MULT":   float(TOTAL_LAMBDA_MULT),
}

print("[persona]", meta)

RAW_PATH = "expense_raw_2024_2025.csv"
df.to_csv(RAW_PATH, index=False, encoding="utf-8-sig", na_rep="")
with open("persona_meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print(f"저장 완료(원천): {RAW_PATH}  ({len(df):,}행)")
print("저장 완료(메타): persona_meta.json")
print(f"카테고리별 건수:\n{df.groupby('category')['amount'].count().sort_values(ascending=False).to_string()}")
print(f"고용형태: {employment_status} | 자영업: {is_self_employed} | 만성질환: {has_chronic_disease} | 요양원: {has_nursing_expense}")
