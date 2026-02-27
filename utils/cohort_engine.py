# cohort_engine.py
"""
seed 기반 가상 레퍼런스 코호트 생성 + 사용자 지출 비교 분석 모듈.
Streamlit 의존 없음. 순수 Pandas/NumPy/SciPy 연산만 수행.
"""

import numpy as np
import pandas as pd
from scipy import stats

# =========================================================
# 상수 정의
# =========================================================

COHORT_SEED = 2024
COHORT_SIZE = 500

# expense_generator.py의 lognorm_params와 동일한 파라미터 사용
# 단위: 월평균 지출 (원)
LOGNORM_PARAMS: dict[str, tuple[float, float]] = {
    "식비":     (9.15, 0.55),
    "교통비":   (8.30, 0.55),
    "쇼핑":     (10.15, 0.85),
    "의료/건강":(9.95, 0.75),
    "문화/여가":(9.80, 0.70),
    "교육":     (10.05, 0.70),
    "기타":     (9.40, 0.85),
    "주거/통신":(13.10, 0.35),
    "구독":     (10.30, 0.45),
}

# 프로파일별 금액 스케일
PROFILE_SCALE: dict[str, float] = {
    "tight":    0.92,
    "normal":   1.00,
    "affluent": 1.12,
}

# lift 판정 기준
LIFT_OVERSPEND = 1.30   # 과소비
LIFT_SAVING    = 0.80   # 절약


# =========================================================
# 1. 사용자 프로파일 추출
# =========================================================

def extract_user_profile(df: pd.DataFrame) -> dict:
    """
    업로드 데이터에서 사용자 지출 프로파일을 추출한다.

    Parameters
    ----------
    df : 전처리 완료된 DataFrame (year_month, category, amount 컬럼 필수)

    Returns
    -------
    dict:
        monthly_avg_total   : 월 평균 총지출 (float)
        monthly_avg_by_cat  : 카테고리별 월평균 지출 dict
        months              : 분석에 포함된 year_month 목록 list
        total_months        : 월 수 int
    """
    required = {"year_month", "category", "amount"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"extract_user_profile: 필수 컬럼 누락 → {missing}")

    monthly_cat = (
        df.groupby(["year_month", "category"])["amount"]
        .sum()
        .reset_index()
    )
    months = sorted(monthly_cat["year_month"].unique().tolist())
    total_months = len(months)

    monthly_total = monthly_cat.groupby("year_month")["amount"].sum()
    monthly_avg_total = float(monthly_total.mean()) if total_months > 0 else 0.0

    monthly_avg_by_cat = (
        monthly_cat.groupby("category")["amount"]
        .mean()
        .to_dict()
    )

    return {
        "monthly_avg_total":  monthly_avg_total,
        "monthly_avg_by_cat": monthly_avg_by_cat,
        "months":             months,
        "total_months":       total_months,
    }


# =========================================================
# 2. 가상 레퍼런스 코호트 생성
# =========================================================

def generate_reference_cohort(
    profile: str = "normal",
    size: int = COHORT_SIZE,
    seed: int = COHORT_SEED,
) -> pd.DataFrame:
    """
    seed 기반으로 재현 가능한 가상 코호트를 생성한다.

    Parameters
    ----------
    profile : "tight" | "normal" | "affluent"
    size    : 코호트 인원 수
    seed    : NumPy random seed

    Returns
    -------
    DataFrame: 행=코호트 구성원, 열=카테고리 (월평균 지출, 원)
    """
    if profile not in PROFILE_SCALE:
        raise ValueError(f"profile은 {list(PROFILE_SCALE.keys())} 중 하나여야 합니다.")

    rng = np.random.default_rng(seed)
    scale = PROFILE_SCALE[profile]

    data: dict[str, np.ndarray] = {}
    for cat, (mu, sigma) in LOGNORM_PARAMS.items():
        raw = rng.lognormal(mean=mu, sigma=sigma, size=size)
        data[cat] = np.round(raw * scale / 100) * 100  # 100원 단위 반올림

    cohort_df = pd.DataFrame(data)
    # 월 총지출 컬럼 추가
    cohort_df["total"] = cohort_df[list(LOGNORM_PARAMS.keys())].sum(axis=1)
    return cohort_df


# =========================================================
# 3. 사용자 소비 순위(percentile) 계산
# =========================================================

def calc_user_percentile(
    user_monthly_avg: float,
    cohort_df: pd.DataFrame,
) -> dict:
    """
    사용자 월평균 총지출이 코호트 내 상위 몇 %인지 계산한다.

    Parameters
    ----------
    user_monthly_avg : 사용자 월평균 총지출 (원)
    cohort_df        : generate_reference_cohort() 결과

    Returns
    -------
    dict:
        percentile_rank  : 상위 X% (낮을수록 소비 많음)
        cohort_mean      : 코호트 평균 (원)
        cohort_median    : 코호트 중앙값 (원)
        user_value       : 입력값 그대로 반환
    """
    cohort_totals = cohort_df["total"].values
    # percentileofscore: 사용자보다 낮은(더 적게 쓴) 사람의 비율
    pct_below = stats.percentileofscore(cohort_totals, user_monthly_avg, kind="rank")
    # "상위 X%" = 100 - pct_below
    top_pct = round(100.0 - pct_below, 1)

    return {
        "percentile_rank": top_pct,          # 상위 X%
        "cohort_mean":     float(np.mean(cohort_totals)),
        "cohort_median":   float(np.median(cohort_totals)),
        "user_value":      user_monthly_avg,
    }


# =========================================================
# 4. 카테고리별 lift 계산
# =========================================================

def calc_category_lift(
    user_avg_by_cat: dict[str, float],
    cohort_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    카테고리별 사용자 지출을 코호트 평균 대비 lift로 계산한다.

    lift = 사용자 월평균 / 코호트 카테고리 평균
    lift >= 1.30 → 과소비
    lift <= 0.80 → 절약
    그 외      → 보통

    Parameters
    ----------
    user_avg_by_cat : extract_user_profile()의 monthly_avg_by_cat
    cohort_df       : generate_reference_cohort() 결과

    Returns
    -------
    DataFrame: category / user_avg / cohort_avg / lift / label 컬럼
    """
    cats = list(LOGNORM_PARAMS.keys())
    rows = []
    for cat in cats:
        user_val    = user_avg_by_cat.get(cat, 0.0)
        cohort_avg  = float(cohort_df[cat].mean()) if cat in cohort_df.columns else 1.0
        lift        = round(user_val / cohort_avg, 3) if cohort_avg > 0 else 0.0

        if lift >= LIFT_OVERSPEND:
            label = "과소비"
        elif lift <= LIFT_SAVING:
            label = "절약"
        else:
            label = "보통"

        rows.append({
            "category":   cat,
            "user_avg":   round(user_val),
            "cohort_avg": round(cohort_avg),
            "lift":       lift,
            "label":      label,
        })

    return pd.DataFrame(rows).sort_values("lift", ascending=False).reset_index(drop=True)


# =========================================================
# 5. 서브카테고리 월평균 Top-N
# =========================================================

def get_top_subcategory_by_amount(
    df: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    서브카테고리별 월평균 지출 Top-N을 반환한다.

    Parameters
    ----------
    df    : 전처리 완료 DataFrame (sub_category, year_month, amount 필수)
    top_n : 반환할 항목 수

    Returns
    -------
    DataFrame: sub_category / monthly_avg 컬럼, 내림차순 정렬
    """
    required = {"sub_category", "year_month", "amount"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"get_top_subcategory_by_amount: 필수 컬럼 누락 → {missing}")

    result = (
        df.groupby(["year_month", "sub_category"])["amount"]
        .sum()
        .reset_index()
        .groupby("sub_category")["amount"]
        .mean()
        .rename("monthly_avg")
        .reset_index()
        .sort_values("monthly_avg", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    result["monthly_avg"] = result["monthly_avg"].round().astype(int)
    return result


# =========================================================
# 6. 통합 실행 함수
# =========================================================

def run_cohort_analysis(
    df: pd.DataFrame,
    profile: str = "normal",
    top_n: int = 5,
) -> dict:
    """
    코호트 분석 전체를 한 번에 실행하는 통합 함수.

    Parameters
    ----------
    df      : 전처리 완료 DataFrame
    profile : "tight" | "normal" | "affluent"
    top_n   : 서브카테고리 Top-N 수

    Returns
    -------
    dict:
        user_profile     : extract_user_profile() 결과
        cohort_df        : generate_reference_cohort() 결과
        percentile       : calc_user_percentile() 결과
        lift_df          : calc_category_lift() 결과 DataFrame
        top_subcat_df    : get_top_subcategory_by_amount() 결과 DataFrame
        profile          : 사용된 프로파일 문자열
    """
    user_profile = extract_user_profile(df)
    cohort_df    = generate_reference_cohort(profile=profile)
    percentile   = calc_user_percentile(
        user_monthly_avg=user_profile["monthly_avg_total"],
        cohort_df=cohort_df,
    )
    lift_df      = calc_category_lift(
        user_avg_by_cat=user_profile["monthly_avg_by_cat"],
        cohort_df=cohort_df,
    )
    top_subcat_df = get_top_subcategory_by_amount(df, top_n=top_n)

    return {
        "user_profile":  user_profile,
        "cohort_df":     cohort_df,
        "percentile":    percentile,
        "lift_df":       lift_df,
        "top_subcat_df": top_subcat_df,
        "profile":       profile,
    }


# =========================================================
# 7. 소득 밴드 기반 코호트 생성 / 로드
# =========================================================

# 소득 밴드 정의 (만원/월 기준)
INCOME_BANDS: list[str] = [
    "300~400만",
    "400~550만",
    "550~700만",
    "700만+",
]

# 소득 밴드 → spending_profile 매핑
_BAND_TO_PROFILE: dict[str, str] = {
    "300~400만":  "tight",
    "400~550만":  "normal",
    "550~700만":  "affluent",
    "700만+":     "affluent",
}

# ── 소득 밴드별 카테고리 월평균 (원) ─────────────────────────────────────
# 각 값은 (중심값, 표준편차). 로그스케일 아닌 실제 원화 단위.
# 설계 기준: 400~550만 기준 월 총지출 380~420만, 300~400만은 240~280만
_BAND_CAT_PARAMS: dict[str, dict[str, tuple[int, int]]] = {
    "300~400만": {
        "식비":     (700_000,  150_000),
        "교통비":   (150_000,   50_000),
        "쇼핑":     (200_000,   80_000),
        "의료/건강":(100_000,   50_000),
        "문화/여가":(100_000,   50_000),
        "교육":     (150_000,   80_000),
        "기타":     (150_000,   60_000),
        "주거/통신":(700_000,  200_000),
        "구독":      (50_000,   20_000),
    },
    "400~550만": {
        "식비":     (950_000,  200_000),
        "교통비":   (220_000,   70_000),
        "쇼핑":     (350_000,  130_000),
        "의료/건강":(150_000,   70_000),
        "문화/여가":(200_000,   90_000),
        "교육":     (220_000,  110_000),
        "기타":     (250_000,  100_000),
        "주거/통신":(1_000_000, 300_000),
        "구독":      (70_000,   25_000),
    },
    "550~700만": {
        "식비":     (1_300_000, 280_000),
        "교통비":   (280_000,   90_000),
        "쇼핑":     (550_000,  200_000),
        "의료/건강":(200_000,   90_000),
        "문화/여가":(350_000,  140_000),
        "교육":     (350_000,  160_000),
        "기타":     (400_000,  160_000),
        "주거/통신":(1_500_000, 400_000),
        "구독":      (90_000,   30_000),
    },
    "700만+": {
        "식비":     (1_700_000, 400_000),
        "교통비":   (380_000,  120_000),
        "쇼핑":     (900_000,  350_000),
        "의료/건강":(300_000,  130_000),
        "문화/여가":(600_000,  250_000),
        "교육":     (550_000,  250_000),
        "기타":     (600_000,  250_000),
        "주거/통신":(2_000_000, 600_000),
        "구독":      (120_000,  40_000),
    },
}

_BAND_CATS = list(list(_BAND_CAT_PARAMS.values())[0].keys())


def generate_band_cohort(
    band: str,
    size: int = 80,
    seed: int = COHORT_SEED,
) -> pd.DataFrame:
    """소득 밴드별 가상 코호트를 생성한다. (카테고리별 월평균 기준)"""
    if band not in INCOME_BANDS:
        raise ValueError(f"band는 {INCOME_BANDS} 중 하나여야 합니다.")

    rng_c = np.random.default_rng(seed + INCOME_BANDS.index(band) * 7)
    params = _BAND_CAT_PARAMS[band]

    data: dict[str, np.ndarray] = {}
    for cat, (center, std) in params.items():
        raw = rng_c.normal(center, std, size=size)
        # 음수/0 방지: 중심의 10% 이하는 클리핑
        raw = np.clip(raw, center * 0.10, center * 3.0)
        data[cat] = np.round(raw / 1000) * 1000   # 1000원 단위 반올림

    df_cohort = pd.DataFrame(data)
    df_cohort["total"]       = df_cohort[_BAND_CATS].sum(axis=1)
    df_cohort["income_band"] = band
    return df_cohort


def generate_all_bands_cohort(size_per_band: int = 80, seed: int = COHORT_SEED) -> pd.DataFrame:
    """전체 소득 밴드 코호트를 합쳐서 반환한다."""
    frames = [generate_band_cohort(band, size=size_per_band, seed=seed) for band in INCOME_BANDS]
    return pd.concat(frames, ignore_index=True)


def save_cohort_parquet(
    path: str = "data/cohort.parquet",
    size_per_band: int = 80,
    seed: int = COHORT_SEED,
) -> None:
    """
    전체 소득 밴드 코호트를 저장한다.
    parquet 라이브러리(pyarrow) 없으면 자동으로 CSV 폴백.
    """
    import os
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    df_out = generate_all_bands_cohort(size_per_band=size_per_band, seed=seed)

    try:
        df_out.to_parquet(path, index=False)
        print(f"[cohort_engine] saved parquet → {path}  ({len(df_out):,}rows)")
    except ImportError:
        csv_path = path.rsplit(".", 1)[0] + ".csv"
        df_out.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"[cohort_engine] pyarrow 없음 → CSV 폴백 {csv_path}  ({len(df_out):,}rows)")


def load_cohort_parquet(path: str = "data/cohort.parquet") -> pd.DataFrame:
    """
    저장된 코호트를 로드한다.
    파일 없으면 즉석 생성. parquet 불가 시 CSV 폴백.
    """
    import os
    csv_path = path.rsplit(".", 1)[0] + ".csv"

    # 1) parquet 시도
    if os.path.exists(path):
        try:
            return pd.read_parquet(path)
        except ImportError:
            pass  # pyarrow 없음 → CSV 확인

    # 2) CSV 폴백
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path, encoding="utf-8-sig")

    # 3) 즉석 생성 (두 저장 모두 실패하면 메모리에서만 반환)
    return generate_all_bands_cohort()


# =========================================================
# 8. 소득 밴드 퍼센타일 / lift
# =========================================================

def calc_user_percentile_by_band(
    user_monthly_avg: float,
    band: str,
    cohort_all: pd.DataFrame,
) -> dict:
    """특정 소득 밴드 코호트 내 사용자 퍼센타일을 계산한다."""
    band_df = cohort_all[cohort_all["income_band"] == band]
    if band_df.empty:
        return {"percentile_rank": 50.0, "cohort_mean": 0.0,
                "cohort_median": 0.0, "user_value": user_monthly_avg, "band": band}
    totals   = band_df["total"].values
    pct_below= stats.percentileofscore(totals, user_monthly_avg, kind="rank")
    return {
        "percentile_rank": round(100.0 - pct_below, 1),
        "cohort_mean":     float(np.mean(totals)),
        "cohort_median":   float(np.median(totals)),
        "user_value":      user_monthly_avg,
        "band":            band,
    }


def calc_category_lift_by_band(
    user_avg_by_cat: dict[str, float],
    band: str,
    cohort_all: pd.DataFrame,
) -> pd.DataFrame:
    """특정 소득 밴드 코호트 기준 카테고리별 lift를 계산한다."""
    band_df = cohort_all[cohort_all["income_band"] == band]
    # 코호트에 존재하는 카테고리 열만 사용
    cats = [c for c in _BAND_CATS if c in cohort_all.columns]
    rows = []
    for cat in cats:
        user_val   = user_avg_by_cat.get(cat, 0.0)
        cohort_avg = float(band_df[cat].mean()) if (cat in band_df.columns and not band_df.empty) else 1.0
        lift       = round(user_val / cohort_avg, 3) if cohort_avg > 0 else 0.0
        label      = "과소비" if lift >= LIFT_OVERSPEND else ("절약" if lift <= LIFT_SAVING else "보통")
        rows.append({"category": cat, "user_avg": round(user_val),
                     "cohort_avg": round(cohort_avg), "lift": lift, "label": label})
    return pd.DataFrame(rows).sort_values("lift", ascending=False).reset_index(drop=True)


# =========================================================
# 9. 소득 밴드 기반 통합 분석
# =========================================================

def run_band_cohort_analysis(
    df: pd.DataFrame,
    band: str,
    cohort_all: pd.DataFrame,
    top_n: int = 5,
    exclude_event: bool = False,
) -> dict:
    """
    소득 밴드 기반 코호트 분석 통합 함수.

    Parameters
    ----------
    df            : 전처리 완료 DataFrame
    band          : 선택한 소득 밴드
    cohort_all    : load_cohort_parquet() 결과
    top_n         : 서브카테고리 Top-N
    exclude_event : True이면 이벤트성 지출(여행/출장/전자기기) 제외
    """
    df_work = df.copy()
    if exclude_event and "memo" in df_work.columns:
        event_mask = df_work["memo"].str.contains("이벤트성 지출|해외여행|출장", na=False)
        df_work    = df_work[~event_mask]

    user_profile  = extract_user_profile(df_work)
    percentile    = calc_user_percentile_by_band(user_profile["monthly_avg_total"], band, cohort_all)
    lift_df       = calc_category_lift_by_band(user_profile["monthly_avg_by_cat"], band, cohort_all)
    top_subcat_df = get_top_subcategory_by_amount(df_work, top_n=top_n)

    return {
        "user_profile":  user_profile,
        "percentile":    percentile,
        "lift_df":       lift_df,
        "top_subcat_df": top_subcat_df,
        "band":          band,
        "exclude_event": exclude_event,
    }


if __name__ == "__main__":
    # python cohort_engine.py 로 실행 시 parquet 생성
    save_cohort_parquet(path="data/cohort.parquet", size_per_band=80)
