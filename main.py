import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import gspread
from google.oauth2.service_account import Credentials
import google.generativeai as genai

# 1. 초기 설정 및 프리미엄 스타일링
st.set_page_config(
    page_title="Wonju AI Quant Lab Pro", 
    layout="wide", 
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS: 다크 모드 최적화 및 메트릭 카드 디자인
st.markdown("""
<style>
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; border: 1px solid #3e4461; }
    [data-testid="stMetricValue"] { font-size: 26px; color: #00ffcc; font-weight: bold; }
    .stAlert { border-radius: 10px; }
    .stButton>button { width: 100%; border-radius: 8px; font-weight: bold; height: 3em; transition: 0.3s; }
    .stButton>button:hover { border-color: #00ffcc; color: #00ffcc; }
</style>
""", unsafe_allow_html=True)

# [안정화] 404 모델 찾기 에러 해결을 위한 동적 로더
def get_stable_model():
    """API 키 확인 및 가용 모델 중 최적의 모델을 자동 선택합니다."""
    if "GOOGLE_API_KEY" not in st.secrets:
        st.info("💡 **안내:** AI 분석을 위해 Secrets에 'GOOGLE_API_KEY'를 추가해주세요.")
        return None
    
    try:
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
        # 현재 키에서 'generateContent'를 지원하는 모델 목록 가져오기
        valid_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        
        # 선호 순위: Flash 1.5 -> Pro 1.5 -> Pro 1.0
        preferred = ['models/gemini-1.5-flash', 'models/gemini-1.5-pro', 'models/gemini-pro']
        target = next((p for p in preferred if p in valid_models), valid_models[0] if valid_models else None)
        
        if target:
            return genai.GenerativeModel(target)
        return None
    except Exception as e:
        st.error(f"❌ AI 시스템 초기화 실패: {e}")
        return None

model = get_stable_model()

# 2. 핵심 유틸리티 함수 (예외 처리 강화)
def get_robust_news(ticker):
    """Yahoo Finance에서 최신 뉴스를 가져옵니다."""
    try:
        stock = yf.Ticker(ticker)
        news_data = stock.news
        if not news_data: return "최근 관련 뉴스 없음"
        return "\n".join([f"- {n['title']} ({n.get('publisher', 'News')})" for n in news_data[:5]])
    except: return "뉴스 로드 실패"

def save_to_google_sheet(url, data):
    """구글 스프레드시트에 분석 데이터를 기록합니다."""
    try:
        if "gcp_service_account" not in st.secrets:
            return False, "❌ Secrets에 서비스 계정 정보가 없습니다."
        
        scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        creds_info = dict(st.secrets["gcp_service_account"])
        creds = Credentials.from_service_account_info(creds_info, scopes=scope)
        client = gspread.authorize(creds)
        sheet = client.open_by_url(url).sheet1
        sheet.append_row(data)
        return True, "✅ 구글 시트 기록 성공!"
    except Exception as e:
        return False, f"❌ 저장 실패: {str(e)}"

@st.cache_data(ttl=3600)
def load_and_process(ticker, period):
    """주가 데이터 로드 및 퀀트 지표(MA20, BB, RSI) 계산"""
    try:
        df = yf.Ticker(ticker).history(period=period)
        if df.empty: return None
        
        # 지표 계산: 20일 이평선 및 볼린저 밴드
        df['MA20'] = df['Close'].rolling(window=20).mean()
        std = df['Close'].rolling(window=20).std()
        df['Upper'], df['Lower'] = df['MA20'] + (std * 2), df['MA20'] - (std * 2)
        
        # RSI 지표 계산
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14).mean()
        rs = gain / loss.replace(0, 0.001)
        df['RSI'] = 100 - (100 / (1 + rs))
        return df
    except: return None

# 3. 사이드바 제어판
with st.sidebar:
    st.header("🔍 분석 설정")
    target_ticker = st.text_input("종목 코드 (Ticker)", value="005930.KS").upper()
    period_choice = st.selectbox("분석 기간", ["6mo", "1y", "3y", "max"], index=1)
    
    st.divider()
    st.header("💾 데이터 연동")
    sheet_url = st.text_input("기록용 시트 URL", value=st.secrets.get("DEFAULT_SHEET_URL", ""))
    st.caption("패드 사용 시 좌측 상단 '>'를 눌러 설정을 변경하세요.")

# 4. 메인 대시보드 로직
df = load_and_process(target_ticker, period_choice)

if df is not None:
    last = df.iloc[-1]
    st.title(f"🚀 {target_ticker} Pro 퀀트 분석")
    
    # 상단 요약 지표
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("현재가", f"{last['Close']:,.0f}")
    m2.metric("RSI (14)", f"{last['RSI']:.1f}")
    m3.metric("볼린저 상단", f"{last['Upper']:,.0f}")
    m4.metric("볼린저 하단", f"{last['Lower']:,.0f}")

    # 3층 통합 차트 (가로폭 반응형 적용)
    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True, 
        row_heights=[0.5, 0.2, 0.3], 
        vertical_spacing=0.05,
        subplot_titles=("주가 및 기술적 지표", "거래량 분석", "RSI 강도")
    )
    
    # 1층: 캔들스틱 + 볼린저 밴드
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name="MA20", line=dict(color='yellow', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Upper'], name="Upper", line=dict(color='rgba(255,255,255,0.2)', dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Lower'], name="Lower", line=dict(color='rgba(255,255,255,0.2)', dash='dot')), row=1, col=1)
    
    # 2층: 거래량 (상승/하락 색상 구분)
    colors = ['#ff4d4d' if r.Open < r.Close else '#4d94ff' for _, r in df.iterrows()]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name="Volume", marker_color=colors), row=2, col=1)
    
    # 3층: RSI 강도 및 기준선
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name="RSI", line=dict(color='#a64dff', width=2)), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    fig.update_layout(height=800, template="plotly_dark", xaxis_rangeslider_visible=False, showlegend=False, margin=dict(l=10, r=10, t=40, b=10))
    # [Fix] 최신 버전 대응 가로폭 설정
    st.plotly_chart(fig, width="stretch")

    # 5. AI 분석 및 기록 액션
    st.divider()
    col_l, col_r = st.columns([2, 1])
    
    with col_l:
        st.subheader("🤖 AI 전략 리포트")
        if st.button("AI 정밀 분석 실행 (뉴스 포함)", type="primary"):
            if model:
                news = get_robust_news(target_ticker)
                prompt = f"""당신은 수석 퀀트 애널리스트입니다.
                종목: {target_ticker} | 현재가: {last['Close']:,.0f} | RSI: {last['RSI']:.1f}
                기술적 상태: 볼린저 밴드 상단 {last['Upper']:,.0f}, 하단 {last['Lower']:,.0f}
                최근 뉴스 요약:
                {news}
                
                [지침] 분석 의견을 [적극 매수/관망/매도] 중 하나로 시작하고, 그 근거를 지표와 뉴스를 섞어 투자 전략으로 3줄 요약하세요."""
                
                with st.spinner("AI가 데이터를 분석 중입니다..."):
                    try:
                        res = model.generate_content(prompt)
                        st.markdown(f"### 📋 분석 결과\n{res.text}")
                    except Exception as e:
                        st.error(f"분석 생성 중 오류: {e}")
            else:
                st.warning("AI 모델이 준비되지 않았습니다. Secrets를 확인하세요.")

    with col_r:
        st.subheader("📝 투자 일지")
        user_memo = st.text_input("메모", placeholder="매수 진입 근거 등")
        if st.button("현재 상태 시트에 기록"):
            if not sheet_url:
                st.warning("시트 URL을 입력해주세요.")
            else:
                log = [datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), target_ticker, last['Close'], round(last['RSI'], 1), user_memo]
                success, msg = save_to_google_sheet(sheet_url, log)
                if success:
                    st.success(msg)
                    st.balloons()
                else: st.error(msg)
else:
    st.error("데이터를 불러오지 못했습니다. 종목 코드가 정확한지 확인하세요.")
