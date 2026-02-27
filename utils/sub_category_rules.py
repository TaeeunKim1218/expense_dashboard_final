# sub_category_rules.py
# -*- coding: utf-8 -*-
"""
서브카테고리 분류 규칙 모듈.

분류 우선순위
-----------
1순위: memo 기반 특수 고정비 판정
       (만성질환, 자영업, 부모님 부양, 출장, 해외여행)
2순위: description 신규 키워드 판정
       (요양원, 상가 월세, 항공권/비행기/왕복, 직원 급여/직원월급/급여)
3순위: 카테고리별 기존 세분류 로직
폴백:  "미분류"

설계 원칙
---------
- 이모지 사용 금지 (모든 문자열)
- UTF-8 인코딩 준수
- 신규 expense_generator.py 출력 데이터와 완전 호환
"""

import pandas as pd


# =========================================================
# 내부 상수: 키워드 집합
# =========================================================

# memo 기반 특수 고정비 식별자
_MEMO_CHRONIC       = "만성질환(고정)"
_MEMO_SELF_EMP      = "자영업(고정)"
_MEMO_NURSING       = "부모님 부양(고정)"
_MEMO_BUSINESS_TRIP = "출장"
_MEMO_ABROAD_TRAVEL = "해외여행"

# 카페 브랜드 키워드 (식비-카페 판정용)
_CAFE_KEYWORDS = [
    "스타벅스", "투썸", "이디야", "메가", "컴포즈",
    "더벤티", "커피", "카페", "공차", "파스쿠찌",
    "커피베이", "매머드",
]

# 구독 서비스 키워드
_OTT_KEYWORDS   = ["넷플릭스", "티빙", "웨이브", "디즈니", "왓챠"]
_MUSIC_KEYWORDS = ["애플뮤직", "스포티파이"]
_MEMBER_KEYWORDS= ["쿠팡", "네이버", "우주패스"]
_AI_KEYWORDS    = ["ChatGPT", "Gemini"]

# 쇼핑 플랫폼 키워드
_SECONDHAND_PLATFORMS = ["당근마켓", "번개장터", "중고나라"]
_ONLINE_PLATFORMS     = ["쿠팡", "네이버", "오늘의집", "카카오선물하기", "올리브영"]
_OFFLINE_PLATFORMS    = ["백화점", "아울렛"]

# 교육 변동 키워드
_BOOK_KEYWORDS     = ["도서", "책"]
_CERT_KEYWORDS     = ["자격증", "응시료"]
_LECTURE_KEYWORDS  = ["온라인 강의", "특강", "세미나"]

# 문화/여가: 문화생활 키워드
_CULTURE_KEYWORDS  = ["영화", "전시", "공연", "입장권", "티켓", "VOD"]
# 문화/여가: 숙박 키워드
_LODGING_KEYWORDS  = ["호텔", "리조트", "숙박", "에어비앤비"]

# 교통: 차량 관련 키워드
_CAR_KEYWORDS      = ["주유", "주차", "통행료", "세차", "정비"]

# 의료: 항공 관련 키워드 (신규 - 교통비 카테고리 내 판정 보조)
_FLIGHT_KEYWORDS   = ["항공권", "항공", "비행기", "왕복"]

# 쇼핑: 전자기기 키워드 (PRICE_TABLE 기반 description + 구버전 호환)
_DEVICE_KEYWORDS   = [
    "아이폰", "갤럭시폰", "맥북", "갤럭시북", "아이패드", "갤럭시탭",
    "애플워치", "갤럭시 워치", "에어팟", "갤럭시 버즈",
    "맥 데스크탑", "데스크탑 구매", "전자기기 구매",
    # 구버전 호환 키워드
    "가전/전자기기", "온라인 노트북",
]


# =========================================================
# 내부 헬퍼 함수
# =========================================================

def _any_in(text: str, keywords: list[str]) -> bool:
    """keywords 중 하나라도 text에 포함되면 True."""
    return any(k in text for k in keywords)


def _classify_by_memo_prefix(memo: str) -> str | None:
    """
    memo 기반 특수 고정비 서브카테고리를 반환한다.
    해당 없으면 None 반환.

    Parameters
    ----------
    memo : row["memo"] 값 (정규화된 문자열)

    Returns
    -------
    str | None : 서브카테고리 문자열 또는 None
    """
    if _MEMO_CHRONIC in memo:
        return "의료-만성질환(고정)"
    if _MEMO_NURSING in memo:
        return "기타-부모님요양(고정)"
    if _MEMO_SELF_EMP in memo:
        # desc로 상가/급여 구분은 하위 로직에서 처리
        return None  # 자영업은 desc 키워드로 추가 세분류
    if _MEMO_BUSINESS_TRIP in memo:
        return "여가-출장"
    if _MEMO_ABROAD_TRAVEL in memo:
        # 숙박/식비/항공 각각 하위 로직에서 구분
        return None
    return None


def _classify_abroad_travel(desc: str, cat: str) -> str | None:
    """
    해외여행 memo가 붙은 행을 카테고리별로 세분류한다.

    Parameters
    ----------
    desc : row["description"]
    cat  : row["category"]

    Returns
    -------
    str | None
    """
    if cat == "교통비":
        return "교통-항공"
    if cat == "문화/여가":
        return "여가-해외여행(숙박/관광)"
    if cat == "식비":
        return "식비-해외여행(식비)"
    return None


# =========================================================
# 공개 함수: assign_sub_category
# =========================================================

def assign_sub_category(row: pd.Series) -> str:
    """
    지출 행(row)을 서브카테고리 문자열로 분류한다.

    Parameters
    ----------
    row : pd.Series
        필수 컬럼: category, description
        선택 컬럼: memo, payment_method

    Returns
    -------
    str : 서브카테고리 레이블 (예: "식비-카페", "교통-항공" 등)
    """
    cat  = str(row.get("category", "")).strip()
    desc = str(row.get("description", "")).strip()
    memo = str(row.get("memo", "")).strip()   if "memo"           in row.index else ""
    pm   = str(row.get("payment_method", "")).strip() if "payment_method" in row.index else ""

    # =========================================================
    # [1순위] memo 기반 특수 고정비 분류
    # =========================================================

    # 만성질환 고정
    if _MEMO_CHRONIC in memo:
        if "약국" in desc:
            return "의료-만성질환(약국)"
        return "의료-만성질환(병원)"

    # 부모님 요양원 고정
    if _MEMO_NURSING in memo:
        return "기타-부모님요양(고정)"

    # 자영업 고정 - desc로 세분류
    if _MEMO_SELF_EMP in memo:
        if any(k in desc for k in ["상가 월세", "상가월세", "월세"]):
            return "기타-자영업(상가월세)"
        if any(k in desc for k in ["직원 급여", "직원급여", "직원월급", "급여"]):
            return "기타-자영업(급여)"
        return "기타-자영업(기타)"

    # 출장
    if _MEMO_BUSINESS_TRIP in memo:
        if cat == "교통비":
            return "교통-출장(교통)"
        if cat == "문화/여가":
            return "여가-출장(숙박)"
        if cat == "식비":
            return "식비-출장(식비)"
        return "기타-출장(기타)"

    # 해외여행 (memo에 "해외여행" 포함)
    if _MEMO_ABROAD_TRAVEL in memo:
        abroad_sub = _classify_abroad_travel(desc, cat)
        if abroad_sub:
            return abroad_sub

    # =========================================================
    # [2순위] description 신규 키워드 분류
    # =========================================================

    # 요양원: desc에 "요양원" 포함
    # - category가 "기타"(부모님 부양)  -> 기타-부모님요양(고정)
    # - category가 "의료/건강"          -> 의료-요양원
    if "요양원" in desc:
        if cat == "기타":
            return "기타-부모님요양(고정)"
        return "의료-요양원"

    # 상가 월세 / 자영업 관련
    if any(k in desc for k in ["상가 월세", "상가월세"]):
        return "기타-자영업(상가월세)"

    # 직원 급여
    if any(k in desc for k in ["직원 급여", "직원급여", "직원월급", "급여"]):
        return "기타-자영업(급여)"

    # 항공권 / 비행기 / 왕복: 교통비 서브카테고리로 판정
    # (category가 교통비가 아닌 경우에도 desc로 판정 가능하도록 배치)
    if _any_in(desc, _FLIGHT_KEYWORDS):
        return "교통-항공"

    # =========================================================
    # [3순위] 카테고리별 기존 세분류 로직
    # =========================================================

    # ---- 구독
    if cat == "구독":
        if "유튜브" in desc:
            return "구독-유튜브"
        if _any_in(desc, _OTT_KEYWORDS):
            return "구독-OTT"
        if _any_in(desc, _MUSIC_KEYWORDS):
            return "구독-음악"
        if _any_in(desc, _MEMBER_KEYWORDS):
            return "구독-멤버십"
        if "iCloud" in desc:
            return "구독-클라우드"
        if _any_in(desc, _AI_KEYWORDS):
            return "구독-AI"
        return "구독-기타"

    # ---- 식비
    if cat == "식비":
        if "장보기" in desc or "마트" in desc:
            return "식비-장보기"
        if _any_in(desc, _CAFE_KEYWORDS):
            return "식비-카페"
        if "배달" in desc:
            return "식비-배달"
        if "편의점" in desc:
            return "식비-편의점"
        return "식비-외식/일반"

    # ---- 교통비
    if cat == "교통비":
        if "지하철" in desc or "버스" in desc:
            return "교통-대중교통"
        if "택시" in desc:
            return "교통-택시"
        if "KTX" in desc or "기차" in desc:
            return "교통-기차"
        # 항공권은 2순위에서 이미 처리되지만, 카테고리 블록 내 안전망으로 유지
        if _any_in(desc, _FLIGHT_KEYWORDS):
            return "교통-항공"
        if _any_in(desc, _CAR_KEYWORDS):
            return "교통-차량"
        return "교통-기타"

    # ---- 쇼핑
    if cat == "쇼핑":
        # 전자기기 우선 판정
        if _any_in(desc, _DEVICE_KEYWORDS):
            return "쇼핑-전자기기"

        # 중고 거래 우선
        if "중고" in desc or (
            pm == "계좌이체" and _any_in(desc, _SECONDHAND_PLATFORMS)
        ):
            return "쇼핑-중고"

        # "플랫폼 | 품목 구매" 형태 파싱
        if " | " in desc:
            right = desc.split(" | ", 1)[1]
            if "의류" in right:       return "쇼핑-의류"
            if "신발" in right:       return "쇼핑-신발"
            if "잡화" in right:       return "쇼핑-잡화"
            if "선물" in right:       return "쇼핑-선물"
            if "생필품" in right:     return "쇼핑-생필품"
            if "화장품" in right:     return "쇼핑-화장품"
            if "반려동물" in right:   return "쇼핑-반려동물"

        # 플랫폼 기반 폴백
        if _any_in(desc, _ONLINE_PLATFORMS):
            return "쇼핑-온라인기타"
        if _any_in(desc, _OFFLINE_PLATFORMS):
            return "쇼핑-오프라인"
        return "쇼핑-기타"

    # ---- 의료/건강
    if cat == "의료/건강":
        if "약국" in desc or "약국" in memo:
            return "의료-약국"
        if "치과" in desc:
            if "정기검진" in memo:   return "의료-치과검진"
            if "교정" in memo:       return "의료-치과교정"
            return "의료-치과"
        if "동물병원" in desc:
            return "의료-반려동물"
        if "건강검진" in memo:
            return "의료-건강검진"
        if "피부과" in memo or "시술" in memo:
            return "의료-피부과"
        return "의료-병원/일반"

    # ---- 교육
    if cat == "교육":
        if "등록금" in desc:
            return "교육-등록금"
        if "학원비(자녀)" in desc:
            return "교육-학원(자녀)"
        if "학원비(본인)" in desc:
            return "교육-학원(본인)"
        if _any_in(desc, _BOOK_KEYWORDS):
            return "교육-도서"
        if _any_in(desc, _CERT_KEYWORDS):
            return "교육-자격증"
        if _any_in(desc, _LECTURE_KEYWORDS):
            return "교육-강의"
        return "교육-기타"

    # ---- 문화/여가
    if cat == "문화/여가":
        if "덕질(" in desc:
            if any(k in desc for k in ["콘서트", "팬미팅"]):
                return "여가-덕질(공연)"
            return "여가-덕질(소액)"
        if "취미(" in memo or "취미(" in desc:
            return "여가-취미"
        if _any_in(desc, _CULTURE_KEYWORDS):
            return "여가-문화생활"
        if _any_in(desc, _LODGING_KEYWORDS):
            return "여가-여행/숙박"
        return "여가-기타"

    # ---- 기타
    if cat == "기타":
        if "경조사" in desc:        return "기타-경조사"
        if "미용실" in desc:        return "기타-미용"
        if "네일" in desc or "왁싱" in desc:
            return "기타-네일"
        if "피부관리" in desc:      return "기타-피부관리"
        if "명절" in desc:          return "기타-명절"
        if "자녀 용돈" in desc:     return "기타-자녀용돈"
        # 신규: 부모님 요양원 (memo 없이 desc만 있는 경우 안전망)
        if "요양원" in desc:        return "기타-부모님요양(고정)"
        # 신규: 자영업 항목 (memo 없이 desc만 있는 경우 안전망)
        if any(k in desc for k in ["상가 월세", "상가월세"]):
            return "기타-자영업(상가월세)"
        if any(k in desc for k in ["직원 급여", "직원급여", "직원월급", "급여"]):
            return "기타-자영업(급여)"
        return "기타-기타"

    # ---- 주거/통신
    if cat == "주거/통신":
        if "휴대폰" in desc:        return "주거/통신-통신"
        # 신규: 상가 월세 (자영업) 구분
        if any(k in desc for k in ["상가 월세", "상가월세"]):
            return "주거/통신-자영업월세"
        if "월세" in desc:          return "주거/통신-주거"
        if "대출" in desc or "상환" in desc:
            return "주거/통신-주거"
        return "주거/통신-기타"

    return "미분류"
