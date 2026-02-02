import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime

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
    .big-font {
        font-size:18px !important;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 월가 퀀트 스타일 주식 분석 대시보드 (Pro)")

# ---------------------------------------------------------
# 2. 사이드바 및 설정
# ---------------------------------------------------------
with st.sidebar:
    st.header("🔍 분석 파라미터")
    target_ticker = st.text_input("종목 코드", value="005930.KS").upper()
    period_choice = st.selectbox("조회 기간", ["6mo", "1y", "3y", "5y", "max"], index=1)
    st.caption("예: 삼성전자(005930.KS), 애플(AAPL), 비트코인(BTC-USD)")
    
    st.divider()
    st.markdown("### 💡 Gems 활용 팁")
    st.markdown("1. 차트가 나오면 스크린샷을 찍습니다.\n2. 하단에 생성된 **'분석 요청 데이터'**를 복사합니다.\n3. **월가 퀀트 마스터 Gems**에 [이미지 + 텍스트]를 같이 넣으세요.")

# ---------------------------------------------------------
# 3. 데이터 로드 및 지표 계산 (핵심 로직)
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
    
    # 1. 이동평균 및 볼린저 밴드
    data['MA20'] = data['Close'].rolling(window=20).mean()
    std_dev = data['Close'].rolling(window=20).std()
    data['Upper_Band'] = data['MA20'] + (std_dev * 2)
    data['Lower_Band'] = data['MA20'] - (std_dev * 2)
    data['Band_Width'] = (data['Upper_Band'] - data['Lower_Band']) / data['MA20'] # 밴드폭(변동성 지표)

    # 2. RSI (Wilder's Smoothing 적용 - 정밀 계산)
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # 지수이동평균(EMA)을 활용한 Wilder's Smoothing 근사
    avg_gain = gain.ewm(alpha=1/14, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14).mean()
    
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    return data

# ---------------------------------------------------------
# 4. Gems 전용 프롬프트 생성기 (데이터 추출)
# ---------------------------------------------------------
def generate_gems_prompt(ticker, df):
    # 최신 데이터 추출
    last_row = df.iloc[-1]
    
    close = last_row['Close']
    rsi = last_row['RSI']
    ma20 = last_row['MA20']
    upper = last_row['Upper_Band']
    lower = last_row['Lower_Band']
    volume = last_row['Volume']
    band_width = last_row['Band_Width']
    
    # 볼린저 밴드 위치 판단
    bb_status = "밴드 내 중심 부근"
    if close >= upper * 0.99: bb_status = "밴드 상단 터치 (과매수 위험?)"
    elif close <= lower * 1.01: bb_status = "밴드 하단 터치 (과매도 기회?)"
    
    # 밴드 폭 판단 (변동성)
    volatility = "수렴(응축)" if band_width < df['Band_Width'].mean() else "발산(확산)"

    # Gems에 보낼 '순수 데이터' 위주의 프롬프트
    prompt = f"""
[분석 요청: {ticker}]
- 분석 시점: {datetime.datetime.now().strftime('%Y-%m-%d')}

[실시간 기술적 데이터 (Fact)]
1. 가격 데이터: 현재가 {close:,.0f} (20일 이평선 {ma20:,.0f} 대비 {'위' if close > ma20 else '아래'})
2. RSI(14): {rsi:.2f} (70이상 과열, 30이하 침체)
3. 볼린저 밴드: 현재 주가는 **{bb_status}**에 위치하며, 변동성은 **{volatility}** 중임.
4. 거래량: 금일 거래량 {volume:,.0f}

이 데이터와 동봉된 차트 이미지를 바탕으로 '5단계 하이엔드 분석'을 수행해줘.
"""
    return prompt

# ---------------------------------------------------------
# 5. 메인 분석 및 시각화 로직
# ---------------------------------------------------------
def main():
    if not target_ticker:
        st.warning("종목 코드를 입력해주세요.")
        return

    # 1. 데이터 로드
    raw_df = load_data(target_ticker, period_choice)
    if raw_df is None:
        st.error(f"❌ '{target_ticker}' 데이터를 찾을 수 없습니다.")
        return

    # 2. 지표 계산
    df = calculate_indicators(raw_df)
    
    # 3. 주요 메트릭 표시
    last_close = df['Close'].iloc[-1]
    prev_close = df['Close'].iloc[-2]
    change = last_close - prev_close
    pct_change = (change / prev_close) * 100
    last_rsi = df['RSI'].iloc[-1]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("현재가", f"{last_close:,.0f}", f"{pct_change:.2f}%")
    col2.metric("RSI(14)", f"{last_rsi:.1f}", delta="과열" if last_rsi >= 70 else "침체" if last_rsi <= 30 else "중립", delta_color="inverse")
    col3.metric("볼린저 상단", f"{df['Upper_Band'].iloc[-1]:,.0f}")
    col4.metric("볼린저 하단", f"{df['Lower_Band'].iloc[-1]:,.0f}")

    # 4. Plotly 인터랙티브 차트
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, 
                        subplot_titles=(f'{target_ticker} Price & Bollinger', 'Volume', 'RSI'),
                        row_heights=[0.6, 0.2, 0.2])

    # [Candle]
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='OHLC'), row=1, col=1)
    
    # [Bollinger Bands]
    fig.add_trace(go.Scatter(x=df.index, y=df['Upper_Band'], line=dict(color='rgba(255, 0, 0, 0.4)', width=1), name='Upper'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Lower_Band'], line=dict(color='rgba(0, 0, 255, 0.4)', width=1), name='Lower'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='orange', width=1, dash='dot'), name='MA20'), row=1, col=1)

    # [Volume]
    colors = ['red' if r.Open > r.Close else 'green' for i, r in df.iterrows()]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name='Volume'), row=2, col=1)

    # [RSI]
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple', width=2), name='RSI'), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="blue", row=3, col=1)

    fig.update_layout(height=800, xaxis_rangeslider_visible=False, hovermode="x unified", margin=dict(t=30, b=10, l=10, r=10))
    st.plotly_chart(fig, use_container_width=True)

    # 5. Gems 전용 데이터 생성 섹션
    st.divider()
    st.subheader("💎 Gems 분석 요청 데이터")
    st.info("이 내용을 복사해서 Gems 채팅창에 붙여넣으세요. (차트 캡처본과 함께 넣으면 완벽합니다)")
    
    # Gems에는 '지침'이 이미 있으므로, 데이터만 깔끔하게 전달하는 형태로 수정
    prompt_text = generate_gems_prompt(target_ticker, df)
    st.text_area("분석 데이터 복사", value=prompt_text, height=200)

if __name__ == "__main__":
    main()
