import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import gspread
from google.oauth2.service_account import Credentials
import google.generativeai as genai

# 1. 초기 설정 및 스타일링
st.set_page_config(page_title="Wonju AI Quant Lab Pro", layout="wide", page_icon="📈")

st.markdown("""
<style>
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; border: 1px solid #3e4461; }
    [data-testid="stMetricValue"] { font-size: 24px; color: #00ffcc; }
</style>
""", unsafe_allow_html=True)

# [안정화] AI 모델 로드 함수 (404 에러 방지)
def get_stable_model():
    try:
        if "GOOGLE_API_KEY" not in st.secrets: return None
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        target = 'models/gemini-1.5-flash'
        return genai.GenerativeModel(target if target in available_models else available_models[0])
    except:
        return None

model = get_stable_model()

# 2. 핵심 유틸리티 함수
def get_robust_news(ticker):
    try:
        stock = yf.Ticker(ticker)
        news_data = stock.news
        return "\n".join([f"- {n['title']} ({n.get('publisher', 'News')})" for n in news_data[:5]]) if news_data else "최근 뉴스 없음"
    except: return "뉴스 로드 실패"

def save_to_google_sheet(url, data):
    try:
        scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        creds = Credentials.from_service_account_info(dict(st.secrets["gcp_service_account"]), scopes=scope)
        client = gspread.authorize(creds)
        sheet = client.open_by_url(url).sheet1
        sheet.append_row(data)
        return True, "✅ 시트 저장 성공!"
    except Exception as e: return False, f"❌ 저장 실패: {e}"

@st.cache_data(ttl=3600)
def load_and_process(ticker, period):
    df = yf.Ticker(ticker).history(period=period)
    if df.empty: return None
    # 지표 계산
    df['MA20'] = df['Close'].rolling(window=20).mean()
    std = df['Close'].rolling(window=20).std()
    df['Upper'], df['Lower'] = df['MA20'] + (std * 2), df['MA20'] - (std * 2)
    delta = df['Close'].diff()
    gain, loss = (delta.where(delta > 0, 0)).ewm(alpha=1/14).mean(), (-delta.where(delta < 0, 0)).ewm(alpha=1/14).mean()
    df['RSI'] = 100 - (100 / (1 + gain/loss))
    return df

# 3. 사이드바 및 레이아웃
with st.sidebar:
    st.header("🔍 분석 설정")
    target_ticker = st.text_input("종목 코드", value="005930.KS").upper()
    period_choice = st.selectbox("기간", ["6mo", "1y", "3y", "max"], index=0)
    st.divider()
    st.header("💾 연동")
    default_url = "https://docs.google.com/spreadsheets/d/1cDwpOaZfEDJY6v7aZa92A9KgRHFqT8S7jy9jywc5rRY/edit?usp=sharing"
    sheet_url = st.text_input("시트 URL", value=default_url)

# 4. 메인 분석 로직
df = load_and_process(target_ticker, period_choice)

if df is not None:
    last = df.iloc[-1]
    st.title(f"🚀 {target_ticker} 뉴스+퀀트 통합 리포트")
    
    # 상단 지표
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("현재가", f"{last['Close']:,.0f}")
    c2.metric("RSI(14)", f"{last['RSI']:.1f}")
    c3.metric("볼린저 상단", f"{last['Upper']:,.0f}")
    c4.metric("볼린저 하단", f"{last['Lower']:,.0f}")

    # 3층 전문 차트
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.5, 0.2, 0.3], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="주가"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name="20일선", line=dict(color='yellow')), row=1, col=1)
    
    colors = ['#ff4d4d' if r.Open < r.Close else '#4d94ff' for _, r in df.iterrows()]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name="거래량", marker_color=colors), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name="RSI", line=dict(color='#a64dff')), row=3, col=1)
    
    fig.update_layout(height=800, template="plotly_dark", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    # 5. AI 분석 및 기록 버튼
    st.divider()
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.subheader("🤖 AI 전략 리포트")
        news = get_robust_news(target_ticker)
        if st.button("뉴스 포함 정밀 분석 실행", type="primary"):
            if model:
                prompt = f"당신은 수석 퀀트입니다. {target_ticker} 분석. 가격:{last['Close']}, RSI:{last['RSI']:.1f}\n뉴스:\n{news}\n[필수] 의견을 [적극 매수/눌림목 대기/매도] 중 하나로 시작해."
                with st.spinner("AI 분석 중..."):
                    res = model.generate_content(prompt)
                    st.session_state['report'] = res.text
                    st.info(res.text)
            else: st.error("AI 모델 로드 실패")

    with col_right:
        st.subheader("💾 기록")
        user_memo = st.text_input("한 줄 메모", placeholder="매수 진입 등")
        if st.button("결과 시트에 기록"):
            log = [datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), target_ticker, last['Close'], last['RSI'], user_memo]
            success, msg = save_to_google_sheet(sheet_url, log)
            if success: st.success(msg); st.balloons()
            else: st.error(msg)