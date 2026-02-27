# utils/ai_analyzer.py
# F006: 패턴 분석 / F007: 예산 추천 / F008: 월간 리포트

import pandas as pd
import streamlit as st
from openai import OpenAI


def _client() -> OpenAI:
    return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


def f006_pattern_analysis(monthly_kpi: dict) -> str:

    sub_top5_text = "\n".join(
        f"  - {k}: {v:,}원"
        for k, v in list(monthly_kpi["sub_top5"].items())[:5]
    )
    cat_share_text = "\n".join(
        f"  - {k}: {v:.1f}%"
        for k, v in sorted(monthly_kpi["cat_share"].items(),
                           key=lambda x: x[1], reverse=True)
    )
    mom_text = (
        f"{monthly_kpi['mom_rate']:+.1f}%"
        if monthly_kpi["mom_rate"] is not None
        else "전월 데이터 없음"
    )

    # 단건 고액 여부 사전 계산 (총지출의 20% 이상이면 언급)
    max_single = monthly_kpi["max_single"]
    total      = monthly_kpi["total_spend"]
    high_single_note = (
        f"이번 달 최대 단건 지출은 {max_single:,}원으로, 총지출의 "
        f"{max_single / total * 100:.1f}%를 차지합니다."
        if total > 0 and max_single / total >= 0.2
        else ""
    )

    prompt = f"""
당신은 개인 재무 분석 전문가입니다.
아래 {monthly_kpi['target_month']} 지출 데이터를 분석하고,
반드시 아래 [인사이트 규칙]을 적용해 작성하세요.

[지출 요약]
- 총지출: {total:,}원
- 전월 대비: {mom_text}
- 거래 건수: {monthly_kpi['tx_count']}건
- 최대 단건: {max_single:,}원
{high_single_note}

[카테고리별 비중]
{cat_share_text}

[서브카테고리 Top5]
{sub_top5_text}

---

[인사이트 규칙 - 해당 항목이 있을 때만 사용]

1. 전월 대비 변화:
   → "전월 대비 총지출이 +X원(+Y%) 증가했습니다." 또는 "감소했습니다."

2. 구독 지출 증가 시:
   → "구독 지출이 최근 3개월 평균 대비 Z% 증가했습니다. 사용하지 않는 구독 점검을 권장합니다."

3. 단건 고액 지출 시 (최대 단건이 총지출의 20% 이상):
   → "이번 달 1회성 고액 지출(카테고리: OOO, X원)이 있어 총지출이 상승했습니다."

4. 증가 기여 카테고리:
   → "지출 증가 기여 1위는 OOO(+X원)입니다."

---

[출력 형식]
① 이번 달 주목할 지출 패턴 2~3가지 (수치 포함, 위 규칙 문장 적극 활용)
② 소비 습관 한 줄 총평
③ 다음 달 개선 제안 1가지

친근하고 구체적인 말투로 작성하세요.
"""

    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "당신은 친절한 개인 재무 분석 전문가입니다."},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=800,
        temperature=0.7,
    )
    return response.choices[0].message.content

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
    cat_str = "\n".join(f"  - {k}: {v:,}원" for k, v in cat_avg.items())

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
    resp = _client().chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "냉철하고 분석적인 재무 컨설턴트입니다."},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.4,
        max_tokens=500,
    )
    return resp.choices[0].message.content


def f008_build_report(
    monthly_kpi: dict,
    pattern: str,
    budget: str,
) -> str:
    """F008: F006 + F007 결과 합쳐 마크다운 리포트 생성"""
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