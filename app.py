import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import gspread
from google.oauth2.service_account import Credentials
import google.generativeai as genai

# 1. 초기 설정 및 보안 연결
st.set_page_config(page_title="Wonju AI Quant Lab Pro v3.1", layout="wide", page_icon="🔥")

# [수정] 모델을 안전하게 불러오는 함수 (404 에러 방지)
def get_stable_model():
    try:
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        target = 'models/gemini-1.5-flash'
        # 리스트에 해당 모델이 있으면 사용, 없으면 첫 번째 가용 모델 선택
        return genai.GenerativeModel(target if target in available_models else available_models[0])
    except Exception:
        return genai.GenerativeModel('gemini-pro')

if "GOOGLE_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    model = get_stable_model()
else:
    st.error("⚠️ secrets.toml에 GOOGLE_API_KEY가 없습니다.")

# 2. 구글 시트 저장 함수
def save_to_google_sheet(url, data):
    try:
        scope = ["https://www.googleapis.com/auth/spreadsheets"]
        # secrets.toml의 [gcp_service_account] 정보를 사용합니다.
        creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scope)
        client = gspread.authorize(creds)
        sheet = client.open_by_url(url).sheet1
        sheet.append_row(data)
        return True
    except Exception as e:
        st.error(f"시트 저장 실패: {e}")
        return False

# 3. 뉴스 수집 함수 (견고함 강화)
def get_robust_news(ticker):
    try:
        stock = yf.Ticker(ticker)
        news_data = stock.news
        if not news_data: return "최근 관련 뉴스가 없습니다."
        # 제목과 출처를 결합하여 더 정보량 많은 텍스트 생성
        return "\n".join([f"- {n['title']} ({n.get('publisher', 'News')})" for n in news_data[:5]])
    except Exception:
        return "뉴스를 불러오는 중 오류가 발생했습니다."

# 4. 데이터 및 지표 계산
@st.cache_data(ttl=3600)
def get_advanced_data(ticker, period):
    df = yf.Ticker(ticker).history(period=period)
    if df.empty: return None
    df['MA20'] = df['Close'].rolling(window=20).mean()
    std = df['Close'].rolling(window=20).std()
    df['Upper'], df['Lower'] = df['MA20'] + (std * 2), df['MA20'] - (std * 2)
    delta = df['Close'].diff()
    gain, loss = (delta.where(delta > 0, 0)).ewm(alpha=1/14).mean(), (-delta.where(delta < 0, 0)).ewm(alpha=1/14).mean()
    df['RSI'] = 100 - (100 / (1 + gain/loss))
    return df

# 5. 사이드바 구성
with st.sidebar:
    st.header("🔍 분석 설정")
    target_ticker = st.text_input("종목 코드 (예: NVDA, 005930.KS)", value="005930.KS").upper()
    period_choice = st.selectbox("기간", ["6mo", "1y", "3y"], index=0)
    sheet_url = st.text_input("구글 시트 URL", value="https://docs.google.com/spreadsheets/d/1cDwpOaZfEDJY6v7aZa92A9KgRHFqT8S7jy9jywc5rRY/edit?usp=sharing")

df = get_advanced_data(target_ticker, period_choice)

if df is not None:
    last = df.iloc[-1]
    st.title(f"🔥 {target_ticker} 뉴스+퀀트 통합 대시보드")
    
    # --- 📊 3층 통합 차트 ---
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.5, 0.2, 0.3], vertical_spacing=0.03)
    
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="주가"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Upper'], name="상단", line=dict(color='rgba(255,255,255,0.2)', dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name="20일선", line=dict(color='yellow')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Lower'], name="하단", line=dict(color='rgba(255,255,255,0.2)', dash='dot')), row=1, col=1)

    colors = ['red' if row['Open'] < row['Close'] else 'blue' for _, row in df.iterrows()]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name="거래량", marker_color=colors), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name="RSI", line=dict(color='orange')), row=3, col=1)
    
    fig.update_layout(height=850, template="plotly_dark", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    # --- 🤖 6. AI 분석 및 저장 섹션 ---
    st.divider()
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📢 AI 전략 리포트")
        news_headlines = get_robust_news(target_ticker)
        
        ai_prompt = f"""
        당신은 원주 연구소의 수석 퀀트 트레이더입니다. {target_ticker}에 대한 공격적인 투자 의견을 제시하세요.
        [지표] 가격: {last['Close']:,.0f}, RSI: {last['RSI']:.1f}, BB상단: {last['Upper']:,.0f}
        [뉴스] {news_headlines}
        [필수] 의견을 [적극 매수 / 눌림목 대기 / 매도] 중 하나로 시작하고 구체적 가격대를 제시하세요.
        """
        
        if st.button("🤖 뉴스 포함 정밀 분석 실행", type="primary"):
            with st.spinner("AI 분석 중..."):
                try:
                    response = model.generate_content(ai_prompt)
                    st.session_state['ai_analysis'] = response.text
                    st.info(response.text)
                except Exception as e:
                    st.error(f"분석 오류: {e}")

    with col2:
        st.subheader("💾 기록소")
        if st.button("🚀 결과를 구글 시트에 기록"):
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            # 시트에는 날짜, 종목, 가격, RSI를 저장합니다.
            log_data = [now, target_ticker, last['Close'], last['RSI']]
            if save_to_google_sheet(sheet_url, log_data):
                st.success("✅ 시트 기록 성공!")