# utils/data_processor.py
# sub_category_rules.py + preprocess + kpi 함수 통합

import pandas as pd
import streamlit as st
from  utils.sub_category_rules import assign_sub_category  # 기존 파일 재활용


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """원본 DataFrame → 분석용 DataFrame"""
    required_cols = ["date", "amount", "category"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"필수 컬럼 누락: {missing}")
        st.stop()

    df = df.copy()
    df["date"]   = pd.to_datetime(df["date"],  errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    # dropna 먼저 (NaN → "nan" 문자열 변환 방지)
    df = df.dropna(subset=["date", "amount", "category"])
    df["category"] = df["category"].astype(str).str.strip()
    df = df[df["category"].ne("")]

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
    df["day_type"]     = df["weekday"].apply(
                             lambda x: "WEEKEND" if x >= 5 else "WEEKDAY"
                         )
    df["season"]       = df["date"].dt.month.map(
        {12:"겨울",1:"겨울",2:"겨울",
          3:"봄",  4:"봄",  5:"봄",
          6:"여름",7:"여름",8:"여름",
          9:"가을",10:"가을",11:"가을"}
    )
    df["sub_category"] = df.apply(assign_sub_category, axis=1)
    return df.sort_values("date").reset_index(drop=True)


def calc_kpi(df: pd.DataFrame) -> dict:
    """전체 필터 기준 KPI 계산"""
    if df.empty:
        return {}
    total_spend   = df["amount"].sum()
    monthly_avg   = df.groupby("year_month")["amount"].sum().mean()
    max_tx        = df["amount"].max()
    fixed_ratio   = (
        df[df["is_fixed"] == True]["amount"].sum() / total_spend * 100
        if total_spend > 0 else 0
    )
    monthly_total = df.groupby("year_month")["amount"].sum().sort_index()
    mom_rate = 0.0
    if len(monthly_total) >= 2:
        last, prev = monthly_total.iloc[-1], monthly_total.iloc[-2]
        mom_rate = round((last - prev) / prev * 100, 1) if prev != 0 else 0.0

    cat_share = (
        df.groupby("category")["amount"].sum() / total_spend * 100
    ).round(1).to_dict()

    return {
        "total_spend": int(total_spend),
        "monthly_avg": int(monthly_avg),
        "max_tx":      int(max_tx),
        "fixed_ratio": round(fixed_ratio, 1),
        "mom_rate":    mom_rate,
        "cat_share":   cat_share,
    }


def build_monthly_kpi(df: pd.DataFrame, target_month: str) -> dict:
    """target_month 기준 월간 KPI (전월 비교 포함)"""
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
        if not prev_df.empty and prev_df["amount"].sum() > 0 else None
    )

    cat_share = (
        this_df.groupby("category")["amount"].sum()
        .sort_values(ascending=False)
        .apply(lambda x: round(x / total * 100, 1))
        .to_dict()
    )
    sub_top5 = (
        this_df.groupby("sub_category")["amount"].sum()
        .sort_values(ascending=False).head(5).to_dict()
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
    """월간 총지출 기준 코호트 밴드 자동 추론"""
    for limit, band in [
        (3_000_000, "300~400만"),
        (5_500_000, "400~550만"),
        (7_000_000, "550~700만"),
        (float("inf"), "700만+"),
    ]:
        if monthly_total < limit:
            return band
    return "700만+"