# 💰 개인 지출 분석 대시보드

> 개인 지출 데이터(CSV/Excel)를 업로드하면 소비 패턴을 자동 분석하고,  
> AI 인사이트와 맞춤형 예산을 추천해주는 인터랙티브 대시보드입니다.

🔗 **[라이브 데모 보기](https://taeeunkim1218-expense-dashboard-final-app-xxxx.streamlit.app)**

---

## 📌 주요 기능

| 탭 | 기능 |
|---|---|
| 내 지출 현황 | KPI 요약 · 카테고리 비중 · 월별 추이 |
| 구독/고정비 | 구독 항목별 결제일 · 고정비 비중 분석 |
| 패턴 분석 | 서브카테고리 드릴다운 · 전월/전년 비교 · 개인 소비 스타일 분석 |
| 코호트 비교 | 소득 밴드 평균 대비 내 지출 비교 |
| 예산 추천 | 최근 3개월 평균 기반 예산 · 목표 달성 여부 게이지 |
| AI 리포트 | GPT 기반 패턴 분석 · 월간 리포트 생성 및 다운로드 |

---

## 🛠 기술 스택

```
Python 3.12
Streamlit     대시보드 프레임워크
Pandas        데이터 전처리
Plotly        인터랙티브 시각화
OpenAI API    AI 패턴 분석 · 예산 추천 · 월간 리포트
Scipy         코호트 통계 분석
```

---

## 📂 폴더 구조

```
expense_dashboard/
├── app.py                      메인 앱
├── requirements.txt
├── data/
│   └── expense_raw_2024_2025.csv  샘플 데이터
└── utils/
    ├── data_processor.py       전처리 · KPI 계산
    ├── ai_analyzer.py          F006 · F007 · F008 AI 분석
    ├── cohort_engine.py        코호트 비교 엔진
    └── sub_category_rules.py   서브카테고리 분류 규칙
```

---

## 📊 데이터 스키마

| 컬럼 | 타입 | 설명 |
|---|---|---|
| date | datetime | 지출 발생일 |
| amount | int | 지출 금액 (원) |
| category | string | 대분류 (식비, 교통비 등) |
| description | string | 지출 상세 설명 |
| payment_method | string | 결제 수단 |
| is_fixed | boolean | 고정지출 여부 |
| memo | string | 메모 |

---

## ⚙️ 로컬 실행 방법

```bash
# 1. 저장소 클론
git clone https://github.com/TaeeunKim1218/expense_dashboard_final.git
cd expense_dashboard_final

# 2. 패키지 설치
pip install -r requirements.txt

# 3. API 키 설정
# .streamlit/secrets.toml 생성 후 아래 내용 입력
# OPENAI_API_KEY = "sk-proj-..."

# 4. 실행
streamlit run app.py
```

---

## 🤖 AI 분석 기능

### F006: 패턴 분석
월간 지출 데이터를 분석해 소비 패턴 인사이트 제공
```
"전월 대비 총지출이 +X원(+Y%) 증가했습니다."
"이번 달 1회성 고액 지출(카테고리: OOO)이 있어 총지출이 상승했습니다."
```

### F007: 예산 추천
최근 3개월 평균 기반 카테고리별 권장 예산 제안

### F008: 월간 리포트
F006 + F007 결과를 통합한 마크다운 리포트 자동 생성 및 다운로드

---

## 📈 분석 목표 (기획안 기준)

| 목표 | 내용 | 구현 |
|---|---|---|
| G1 | 월간/연간/계절성 패턴 분석 | ✅ |
| G2 | 구독/고정비 분석 | ✅ |
| G3 | 예산 관리 · 목표 달성 여부 | ✅ |
| G4 | 이상치 탐지 | ✅ |

---

## 👩‍💻 개발자

**김태은** · [GitHub](https://github.com/TaeeunKim1218)
