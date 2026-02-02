import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import gspread
from google.oauth2.service_account import Credentials

# ---------------------------------------------------------
# 1. 페이지 설정 및 스타일
# ---------------------------------------------------------
st.set_page_config(page_title="Pro 퀀트 분석 대시보드", layout="wide", page_icon="📈")

st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
    }
    [data-testid="stMetricValue"] {
        font-size: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 월가 퀀트 스타일 주식 분석 대시보드 (Pro)")

# ---------------------------------------------------------
# 2. 사이드바 설정
# ---------------------------------------------------------
with st.sidebar:
    st.header("🔍 분석 파라미터")
    target_ticker = st.text_input("종목 코드", value="005930.KS").upper()
    period_choice = st.selectbox("조회 기간", ["6mo", "1y", "3y", "5y", "max"], index=1)
    
    st.divider()
    st.header("💾 구글 시트 연동")
    
    # [수정 완료] 제공해주신 구글 시트 URL을 기본값으로 설정했습니다.
    default_url = "https://docs.google.com/spreadsheets/d/1cDwpOaZfEDJY6v7aZa92A9KgRHFqT8S7jy9jywc5rRY/edit?usp=sharing" 
    
    sheet_url = st.text_input("구글 시트 URL", value=default_url, placeholder="https://docs.google.com/spreadsheets/d/...")
    st.caption("※ `secrets.toml`에 키 설정이 선행되어야 합니다.")

# ---------------------------------------------------------
# 3. 데이터 로드 및 지표 계산
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def load_data(ticker, period):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        if df.empty:
            return None
        return df
    except Exception:
        return None

def calculate_indicators(df):
    data = df.copy()
    
    # 볼린저 밴드
    data['MA20'] = data['Close'].rolling(window=20).mean()
    std_dev = data['Close'].rolling(window=20).std()
    data['Upper_Band'] = data['MA20'] + (std_dev * 2)
    data['Lower_Band'] = data['MA20'] - (std_dev * 2)
    data['Band_Width'] = (data['Upper_Band'] - data['Lower_Band']) / data['MA20']

    # RSI (Wilder's Smoothing 정밀 계산)
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.ewm(alpha=1/14, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14).mean()
    
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    return data

# ---------------------------------------------------------
# 4. 헬퍼 함수: 프롬프트 생성 & 시트 저장
# ---------------------------------------------------------
def generate_gems_prompt(ticker, df):
    last_row = df.iloc[-1]
    close = last_row['Close']
    rsi = last_row['RSI']
    ma20 = last_row['MA20']
    upper = last_row['Upper_Band']
    lower = last_row['Lower_Band']
    band_width = last_row['Band_Width']
    volume = last_row['Volume']
    
    bb_status = "밴드 중심"
    if close >= upper * 0.99: bb_status = "밴드 상단 터치 (과매수?)"
    elif close <= lower * 1.01: bb_status = "밴드 하단 터치 (과매도?)"
    
    volatility = "수렴(응축)" if band_width < df['Band_Width'].mean() else "발산(확산)"

    return f"""
[분석 요청: {ticker}]
- 분석 시점: {datetime.datetime.now().strftime('%Y-%m-%d')}

[실시간 기술적 데이터 (Fact)]
1. 가격: {close:,.0f} (20일선 {ma20:,.0f} 대비 {'위' if close > ma20 else '아래'})
2. RSI(14): {rsi:.2f} (70이상 과열, 30이하 침체)
3. 볼린저 밴드: 현재 **{bb_status}**, 변동성은 **{volatility}** 상태.
4. 거래량: {volume:,.0f}

위 데이터를 바탕으로 '5단계 하이엔드 분석' 및 승률 높은 포지션을 제안해줘.

[★특별 요청 사항: 기록용 메모 작성]
답변의 맨 마지막 줄에, 내가 구글 시트에 바로 '복사+붙여넣기' 할 수 있도록 **[한 줄 기록용 메모]**를 작성해줘.
형식: "[추천포지션] 핵심 근거 요약"
예시 1: "[분할 매수] RSI 28 침체권 진입 및 볼린저 밴드 하단 지지"
예시 2: "[관망] RSI 55 중립 구간이며 거래량 부족으로 추세 미확정"
"""

def save_to_google_sheet(url, data):
    try:
        # Streamlit Cloud의 Secrets 기능을 사용
        if "gcp_service_account" not in st.secrets:
            return False, "설정 오류: Secrets에 gcp_service_account가 없습니다."

        scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
        # secrets를 dict 형태로 변환하여 인증
        creds_dict = dict(st.secrets["gcp_service_account"])
        credentials = Credentials.from_service_account_info(creds_dict, scopes=scope)
        client = gspread.authorize(credentials)
        
        sheet = client.open_by_url(url).sheet1
        sheet.append_row(data)
        return True, "✅ 구글 시트에 성공적으로 저장했습니다!"
    except Exception as e:
        return False, f"❌ 저장 실패: {str(e)}"

# ---------------------------------------------------------
# 5. 메인 로직
# ---------------------------------------------------------
def main():
    if not target_ticker:
        st.warning("종목 코드를 입력해주세요.")
        return

    raw_df = load_data(target_ticker, period_choice)
    if raw_df is None:
        st.error(f"❌ '{target_ticker}' 데이터를 찾을 수 없습니다.")
        return

    df = calculate_indicators(raw_df)
    
    # 최신 데이터
    last_row = df.iloc[-1]
    last_close = last_row['Close']
    last_rsi = last_row['RSI']
    
    # 상단 메트릭
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("현재가", f"{last_close:,.0f}")
    col2.metric("RSI(14)", f"{last_rsi:.1f}", delta="과열" if last_rsi >= 70 else "침체" if last_rsi <= 30 else "중립", delta_color="inverse")
    col3.metric("볼린저 상단", f"{last_row['Upper_Band']:,.0f}")
    col4.metric("볼린저 하단", f"{last_row['Lower_Band']:,.0f}")

    # Plotly 차트
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='OHLC'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Upper_Band'], line=dict(color='rgba(255,0,0,0.4)'), name='Upper'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Lower_Band'], line=dict(color='rgba(0,0,255,0.4)'), name='Lower'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='orange', dash='dot'), name='MA20'), row=1, col=1)
    
    colors = ['red' if r.Open > r.Close else 'green' for i, r in df.iterrows()]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name='Volume'), row=2, col=1)
    
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple'), name='RSI'), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="blue", row=3, col=1)
    
    fig.update_layout(height=800, xaxis_rangeslider_visible=False, hovermode="x unified", margin=dict(t=30, b=10, l=10, r=10))
    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------------------------------
    # [기능 1] 구글 시트 저장 (사용자 코멘트 포함)
    # -----------------------------------------------------
    st.divider()
    st.subheader("💾 분석 결과 저장하기")
    
    with st.container():
        c1, c2 = st.columns([3, 1])
        with c1:
            user_note = st.text_input("한 줄 메모 (예: RSI 다이버전스 확인, 매수 진입)", key="note")
        with c2:
            st.write("") # 여백용
            st.write("")
            if st.button("구글 시트에 기록", type="primary"):
                if not sheet_url:
                    st.error("사이드바에 시트 URL을 먼저 입력하세요.")
                else:
                    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    rsi_state = "과매수" if last_rsi >= 70 else "과매도" if last_rsi <= 30 else "중립"
                    
                    # 시트에 저장될 데이터 순서
                    row_data = [timestamp, target_ticker, last_close, last_rsi, rsi_state, user_note]
                    
                    with st.spinner("저장 중..."):
                        success, msg = save_to_google_sheet(sheet_url, row_data)
                        if success:
                            st.success(msg)
                            st.balloons()
                        else:
                            st.error(msg)

    # -----------------------------------------------------
    # [기능 2] Gems 프롬프트 생성 (기존 유지)
    # -----------------------------------------------------
    st.divider()
    st.subheader("💎 Gems 분석 요청 데이터")
    st.info("아래 내용을 복사해서 Gems 채팅창에 붙여넣으세요.")
    prompt_text = generate_gems_prompt(target_ticker, df)
    st.text_area("데이터 복사", value=prompt_text, height=200)

if __name__ == "__main__":
    main()
