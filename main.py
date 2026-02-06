iimport streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import gspread
from google.oauth2.service_account import Credentials
import google.generativeai as genai

# 1. 초기 설정 및 스타일링
# 모바일 및 패드 환경을 고려하여 레이아웃을 최적화합니다.
st.set_page_config(
    page_title="Wonju AI Quant Lab Pro", 
    layout="wide", 
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS: 메트릭 카드 및 경고창 디자인
st.markdown("""
<style>
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; border: 1px solid #3e4461; }
    [data-testid="stMetricValue"] { font-size: 24px; color: #00ffcc; }
    .stAlert { border-radius: 10px; }
    .stButton>button { width: 100%; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# [안정화] AI 모델 로드 함수
# Secrets 설정이 누락되었을 때 앱이 멈추지 않고 안내 메시지를 출력합니다.
def get_stable_model():
    # 1. API 키 확인
    if "GOOGLE_API_KEY" not in st.secrets:
        st.info("💡 **알림:** AI 기능을 사용하려면 Streamlit Cloud 설정의 'Secrets'에 `GOOGLE_API_KEY`를 추가해주세요.")
        return None
    
    try:
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
        # 모델 목록을 확인하여 가장 안정적인 모델을 선택합니다.
        # 로그에 나타난 'google.generativeai' 지원 종료 예고에 대비하여 예외 처리를 강화합니다.
        model = genai.GenerativeModel('gemini-1.5-flash')
        return model
    except Exception as e:
        st.error(f"❌ AI 초기화 오류: {str(e)}")
        return None

# 전역 모델 객체 생성
model = get_stable_model()

# 2. 핵심 유틸리티 함수
def get_robust_news(ticker):
    """야후 파이낸스에서 최신 뉴스를 가져옵니다."""
    try:
        stock = yf.Ticker(ticker)
        news_data = stock.news
        if not news_data:
            return "최근 관련 뉴스가 없습니다."
        return "\n".join([f"- {n['title']} ({n.get('publisher', 'News')})" for n in news_data[:5]])
    except Exception as e: 
        return f"뉴스 로드 중 오류 발생: {str(e)}"

def save_to_google_sheet(url, data):
    """분석 결과를 구글 스프레드시트에 기록합니다."""
    try:
        if "gcp_service_account" not in st.secrets:
            return False, "❌ Secrets에 `gcp_service_account` 설정이 없습니다."
            
        scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        # 서비스 계정 인증 정보 로드
        creds_info = dict(st.secrets["gcp_service_account"])
        creds = Credentials.from_service_account_info(creds_info, scopes=scope)
        client = gspread.authorize(creds)
        
        # URL에서 시트 열기
        sheet = client.open_by_url(url).sheet1
        sheet.append_row(data)
        return True, "✅ 구글 시트에 성공적으로 기록되었습니다!"
    except Exception as e: 
        return False, f"❌ 시트 저장 실패: {str(e)}"

@st.cache_data(ttl=3600)
def load_and_process(ticker, period):
    """주가 데이터를 로드하고 기술적 지표를 계산합니다."""
    try:
        df = yf.Ticker(ticker).history(period=period)
        if df.empty: return None
        
        # 20일 이동평균선 및 볼린저 밴드 계산
        df['MA20'] = df['Close'].rolling(window=20).mean()
        std = df['Close'].rolling(window=20).std()
        df['Upper'] = df['MA20'] + (std * 2)
        df['Lower'] = df['MA20'] - (std * 2)
        
        # RSI (Relative Strength Index) 계산
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14).mean()
        # 0으로 나누기 방지 처리
        rs = gain / loss.replace(0, 0.001)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        return df
    except Exception:
        return None

# 3. 사이드바 구성
with st.sidebar:
    st.header("🔍 분석 설정")
    target_ticker = st.text_input("종목 코드 (예: 005930.KS)", value="005930.KS").upper()
    period_choice = st.selectbox("데이터 분석 기간", ["6mo", "1y", "3y", "max"], index=1)
    
    st.divider()
    st.header("💾 데이터 연동")
    # Secrets에 저장된 기본 URL이 있다면 사용하고, 없으면 예시 URL 표시
    default_sheet = st.secrets.get("DEFAULT_SHEET_URL", "https://docs.google.com/spreadsheets/...")
    sheet_url = st.text_input("구글 시트 URL", value=default_sheet)
    
    st.info("패드/모바일 사용 시 좌측 상단 '>' 버튼을 눌러 설정을 변경할 수 있습니다.")

# 4. 메인 분석 대시보드
df = load_and_process(target_ticker, period_choice)

if df is not None:
    last = df.iloc[-1]
    st.title(f"🚀 {target_ticker} AI 퀀트 리포트")
    
    # 상단 핵심 지표 요약
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("현재가", f"{last['Close']:,.0f}")
    m2.metric("RSI (14)", f"{last['RSI']:.1f}")
    m3.metric("볼린저 상단", f"{last['Upper']:,.0f}")
    m4.metric("볼린저 하단", f"{last['Lower']:,.0f}")

    # 차트 구성 (3단 Subplots)
    # 로그 경고 해결: 캔버스의 반응형 너비를 위해 width='stretch' 적용
    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True, 
        row_heights=[0.5, 0.2, 0.3], 
        vertical_spacing=0.05,
        subplot_titles=("주가 및 볼린저 밴드", "거래량", "RSI 지표")
    )
    
    # 1단: 캔들스틱 및 이동평균선
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], 
        name="Price"
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name="MA20", line=dict(color='yellow', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Upper'], name="Upper", line=dict(color='gray', width=1, dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Lower'], name="Lower", line=dict(color='gray', width=1, dash='dot')), row=1, col=1)
    
    # 2단: 거래량 (가격 상승/하락에 따른 색상 구분)
    bar_colors = ['#ff4d4d' if r.Open < r.Close else '#4d94ff' for _, r in df.iterrows()]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name="Volume", marker_color=bar_colors), row=2, col=1)
    
    # 3단: RSI 지표 및 과매수/과매도 기준선
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name="RSI", line=dict(color='#a64dff', width=2)), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    fig.update_layout(
        height=800, 
        template="plotly_dark", 
        xaxis_rangeslider_visible=False,
        showlegend=False,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    
    # [Fix] Streamlit 1.54.0+ 버전 규격에 맞게 width 설정
    st.plotly_chart(fig, width="stretch")

    # 5. 하단 액션 영역 (AI 분석 및 기록)
    st.divider()
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.subheader("🤖 AI 전략 리포트")
        if st.button("AI 정밀 분석 실행 (뉴스 포함)", type="primary"):
            if model:
                news_context = get_robust_news(target_ticker)
                prompt = f"""당신은 원주 퀀트 연구소의 수석 애널리스트입니다.
                종목: {target_ticker}
                데이터: 현재가 {last['Close']:,.0f}, RSI {last['RSI']:.1f}, 볼린저밴드 상단 {last['Upper']:,.0f}
                최근 뉴스 요약:
                {news_context}
                
                위 지표와 뉴스를 종합하여 [적극 매수 / 관망 / 매도] 중 하나를 선택하고, 그 이유를 투자 전략 관점에서 3줄로 요약하세요."""
                
                with st.spinner("AI가 시장 데이터를 분석 중입니다..."):
                    try:
                        response = model.generate_content(prompt)
                        st.markdown(f"### 📋 분석 결과\n{response.text}")
                        # 세션에 결과 저장하여 유지
                        st.session_state['last_analysis'] = response.text
                    except Exception as e:
                        st.error(f"분석 생성 중 오류가 발생했습니다: {e}")
            else:
                st.warning("AI 모델이 준비되지 않았습니다. 사이드바의 안내를 확인하세요.")

    with col_right:
        st.subheader("📝 투자 기록")
        user_memo = st.text_input("메모", placeholder="매수 진입 근거 등 입력")
        if st.button("현재 상태 시트에 기록"):
            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            # 저장할 데이터 배열 생성
            record = [now_str, target_ticker, last['Close'], round(last['RSI'], 2), user_memo]
            
            if not sheet_url or "spreadsheets" not in sheet_url:
                st.warning("유효한 구글 시트 URL을 입력해주세요.")
            else:
                with st.spinner("데이터 기록 중..."):
                    success, msg = save_to_google_sheet(sheet_url, record)
                    if success:
                        st.success(msg)
                        st.balloons()
                    else:
                        st.error(msg)
else:
    st.error(f"❌ '{target_ticker}' 데이터를 불러올 수 없습니다. 종목 코드가 올바른지(예: 005930.KS) 확인해주세요.")

