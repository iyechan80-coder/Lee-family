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

# 테마 스타일링 개선
st.markdown("""
<style>
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; border: 1px solid #3e4461; }
    [data-testid="stMetricValue"] { font-size: 24px; color: #00ffcc; }
    .stAlert { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# [안정화] AI 모델 로드 함수 (디버깅 강화)
def get_stable_model():
    # 1. API 키 존재 여부 확인
    if "GOOGLE_API_KEY" not in st.secrets:
        st.error("🔑 Secrets 설정에 'GOOGLE_API_KEY'가 없습니다.")
        return None
    
    try:
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
        # 모델 목록 확인 및 최적 모델 선택
        target_model = 'gemini-1.5-flash'
        model = genai.GenerativeModel(target_model)
        # 테스트 호출 (연결 확인)
        return model
    except Exception as e:
        st.error(f"❌ AI 초기화 오류: {str(e)}")
        return None

# 전역 모델 설정
model = get_stable_model()

# 2. 핵심 유틸리티 함수
def get_robust_news(ticker):
    try:
        stock = yf.Ticker(ticker)
        news_data = stock.news
        if not news_data: return "최근 뉴스 없음"
        return "\n".join([f"- {n['title']} ({n.get('publisher', 'News')})" for n in news_data[:5]])
    except Exception as e: 
        return f"뉴스 로드 실패: {str(e)}"

def save_to_google_sheet(url, data):
    try:
        if "gcp_service_account" not in st.secrets:
            return False, "❌ Secrets에 gcp_service_account 설정이 없습니다."
            
        scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        # 서비스 계정 정보를 dict로 변환하여 인증
        creds_info = dict(st.secrets["gcp_service_account"])
        creds = Credentials.from_service_account_info(creds_info, scopes=scope)
        client = gspread.authorize(creds)
        sheet = client.open_by_url(url).sheet1
        sheet.append_row(data)
        return True, "✅ 시트 저장 성공!"
    except Exception as e: 
        return False, f"❌ 저장 실패: {str(e)}"

@st.cache_data(ttl=3600)
def load_and_process(ticker, period):
    try:
        df = yf.Ticker(ticker).history(period=period)
        if df.empty: return None
        
        # 지표 계산
        df['MA20'] = df['Close'].rolling(window=20).mean()
        std = df['Close'].rolling(window=20).std()
        df['Upper'], df['Lower'] = df['MA20'] + (std * 2), df['MA20'] - (std * 2)
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14).mean()
        
        # 0 나누기 방지
        rs = gain / loss.replace(0, 0.001)
        df['RSI'] = 100 - (100 / (1 + rs))
        return df
    except:
        return None

# 3. 사이드바 구성
with st.sidebar:
    st.header("🔍 분석 설정")
    target_ticker = st.text_input("종목 코드 (예: 005930.KS)", value="005930.KS").upper()
    period_choice = st.selectbox("기간", ["6mo", "1y", "3y", "max"], index=0)
    st.divider()
    st.header("💾 연동")
    # 기본 URL은 예시이므로 사용자의 것으로 교체 가능
    sheet_url = st.text_input("시트 URL", value=st.secrets.get("DEFAULT_SHEET_URL", ""))

# 4. 메인 분석 로직
df = load_and_process(target_ticker, period_choice)

if df is not None:
    last = df.iloc[-1]
    st.title(f"🚀 {target_ticker} 퀀트 리포트")
    
    # 상단 메트릭
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("현재가", f"{last['Close']:,.0f}")
    m2.metric("RSI(14)", f"{last['RSI']:.1f}")
    m3.metric("볼린저 상단", f"{last['Upper']:,.0f}")
    m4.metric("볼린저 하단", f"{last['Lower']:,.0f}")

    # 차트 생성 (3단 구성)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                        row_heights=[0.5, 0.2, 0.3], vertical_spacing=0.05)
    
    # 1단: 주가 및 볼린저 밴드
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="주가"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name="20일선", line=dict(color='yellow', width=1)), row=1, col=1)
    
    # 2단: 거래량
    colors = ['#ff4d4d' if r.Open < r.Close else '#4d94ff' for _, r in df.iterrows()]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name="거래량", marker_color=colors), row=2, col=1)
    
    # 3단: RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name="RSI", line=dict(color='#a64dff')), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    # 차트 레이아웃 (Deprecation 경고 해결: width='stretch')
    fig.update_layout(height=700, template="plotly_dark", xaxis_rangeslider_visible=False, margin=dict(t=30, b=10))
    st.plotly_chart(fig, width="stretch")

    # 5. AI 분석 영역
    st.divider()
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.subheader("🤖 AI 전략 리포트")
        if st.button("AI 정밀 분석 실행", type="primary", use_container_width=True):
            if model:
                news = get_robust_news(target_ticker)
                prompt = f"""당신은 전문 퀀트 애널리스트입니다. 
                종목: {target_ticker}
                현재 주가: {last['Close']:,.0f}
                RSI: {last['RSI']:.1f}
                최근 뉴스: {news}
                
                위 데이터를 바탕으로 [적극 매수/관망/매도] 의견 중 하나를 선택하고 이유를 3줄 내외로 설명하세요."""
                
                with st.spinner("Gemini AI가 분석 중입니다..."):
                    try:
                        response = model.generate_content(prompt)
                        st.markdown(f"### 분석 결과\n{response.text}")
                        st.session_state['last_report'] = response.text
                    except Exception as e:
                        st.error(f"분석 중 오류 발생: {e}")
            else:
                st.error("AI 모델이 로드되지 않았습니다. Secrets 설정을 확인하세요.")

    with col_right:
        st.subheader("💾 기록")
        user_memo = st.text_input("메모", placeholder="매수 진입 지점 등")
        if st.button("구글 시트에 기록", use_container_width=True):
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            log_data = [now, target_ticker, last['Close'], round(last['RSI'], 2), user_memo]
            
            if not sheet_url:
                st.warning("시트 URL을 입력해주세요.")
            else:
                success, msg = save_to_google_sheet(sheet_url, log_data)
                if success:
                    st.success(msg)
                    st.balloons()
                else:
                    st.error(msg)
else:
    st.error("데이터를 불러올 수 없습니다. 종목 코드를 확인하세요.")

