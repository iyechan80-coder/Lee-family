import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import gspread
from google.oauth2.service_account import Credentials
import time

# 1. 초기 설정 (버전 v5.8: AI 제거, 백테스팅 탑재, 시트 저장 개선)
st.set_page_config(page_title="Wonju AI Quant Lab v5.8", layout="wide", page_icon="💎")

# 2. 데이터 캐싱 및 초기화
@st.cache_data(show_spinner=False, ttl=3600)
def get_stock_info(symbol):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            tick = yf.Ticker(symbol)
            info = tick.info
            if info and 'symbol' in info:
                return info
        except Exception:
            time.sleep(1)
            continue
    return None

# 3. 펀더멘털 지표 시각화
def display_fundamental_metrics(info):
    if not info:
        st.warning("⚠️ 기업 재무 정보를 불러올 수 없습니다. (차트 및 백테스팅은 가능)")
        return

    currency = info.get('currency', 'KRW')
    market_cap = info.get('marketCap', 0)
    
    if currency == 'KRW':
        cap_display = f"{market_cap / 1_000_000_000_000:.2f}조 원"
    elif currency == 'USD':
        cap_display = f"${market_cap / 1_000_000_000:.2f} B"
    else:
        cap_display = f"{market_cap:,.0f} {currency}"

    st.markdown(f"### 🏢 {info.get('shortName', 'Unknown')} 펀더멘털 분석")
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("시가총액", cap_display)
    with col2: st.metric("PER", f"{info.get('trailingPE', 0):.2f}배" if info.get('trailingPE') else "N/A")
    with col3: st.metric("PBR", f"{info.get('priceToBook', 0):.2f}배" if info.get('priceToBook') else "N/A")
    with col4: st.metric("배당수익률", f"{info.get('dividendYield', 0)*100:.2f}%" if info.get('dividendYield') else "N/A")
    st.divider()

# 4. 구글 시트 저장 (상단 삽입 방식으로 변경)
def save_to_google_sheet(url, data):
    try:
        if "gcp_service_account" not in st.secrets:
            st.error("❌ 설정 오류: 'secrets.toml' 인증 정보 누락")
            return False

        scope = ["https://www.googleapis.com/auth/spreadsheets"]
        creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scope)
        client = gspread.authorize(creds)
        
        # URL로 시트 열기
        spreadsheet = client.open_by_url(url)
        sheet = spreadsheet.sheet1 # 첫 번째 워크시트
        
        # [Fix] append 대신 insert_row 사용 (2번째 줄에 삽입하여 최상단 노출)
        sheet.insert_row(data, index=2)
        
        st.toast(f"✅ '{spreadsheet.title}' 시트 상단에 저장되었습니다!", icon="💾")
        return True
    except Exception as e:
        st.error(f"❌ 저장 실패: {str(e)}")
        return False

# 5. 뉴스 가져오기
def get_robust_news(ticker):
    max_retries = 2
    for attempt in range(max_retries):
        try:
            stock = yf.Ticker(ticker)
            if attempt > 0: time.sleep(1)
            news_data = stock.news
            if isinstance(news_data, list) and len(news_data) > 0:
                news_list = []
                for n in news_data[:5]:
                    if isinstance(n, dict):
                        title = n.get('title', '제목 정보 없음')
                        publisher = n.get('publisher', '출처 미상')
                        news_list.append(f"- {title} ({publisher})")
                if news_list: return "\n".join(news_list)
            return "[데이터 없음] 현재 야후 파이낸스 뉴스 부재"
        except Exception:
            if attempt == max_retries - 1: return "[오류] 뉴스 서버 연결 불안정"
            continue
    return "[데이터 없음]"

# 6. 백테스팅 엔진 (Phase 1)
def run_backtest(df, buy_rsi, sell_rsi):
    df = df.copy()
    position = 0 # 0: 현금, 1: 주식
    trades = []
    
    for i in range(len(df)):
        rsi = df['RSI'].iloc[i]
        price = df['Close'].iloc[i]
        
        if position == 0 and rsi <= buy_rsi: # 매수
            position = 1
            buy_price = price
            df.at[df.index[i], 'Signal'] = 'Buy'
        elif position == 1 and rsi >= sell_rsi: # 매도
            position = 0
            profit = (price - buy_price) / buy_price * 100
            trades.append(profit)
            df.at[df.index[i], 'Signal'] = 'Sell'
            
    total_return = np.sum(trades) if trades else 0.0
    win_rate = (len([t for t in trades if t > 0]) / len(trades) * 100) if trades else 0.0
    return df, trades, total_return, win_rate

# 7. 기술적 데이터 계산
@st.cache_data(ttl=3600)
def get_advanced_data(ticker, period):
    try:
        df = yf.Ticker(ticker).history(period=period)
        if df.empty: return None
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['RSI'] = 100 - (100 / (1 + gain/loss))
        # 볼린저 밴드
        df['MA20'] = df['Close'].rolling(window=20).mean()
        std = df['Close'].rolling(window=20).std()
        df['Upper'] = df['MA20'] + (std * 2)
        df['Lower'] = df['MA20'] - (std * 2)
        return df
    except Exception:
        return None

# --- 메인 실행 ---
with st.sidebar:
    st.header("🔍 설정")
    target_ticker = st.text_input("종목 코드", value="005930.KS").upper()
    period_choice = st.selectbox("분석 기간", ["1y", "2y", "5y"], index=1)
    
    st.divider()
    st.subheader("🛠️ 전략 검증 (Backtest)")
    rsi_buy_level = st.slider("매수 RSI 기준", 10, 40, 30, help="이 수치보다 낮으면 매수합니다.")
    rsi_sell_level = st.slider("매도 RSI 기준", 60, 90, 70, help="이 수치보다 높으면 매도합니다.")
    
    st.divider()
    sheet_url = st.text_input("구글 시트 URL", placeholder="https://docs.google.com/...")

df = get_advanced_data(target_ticker, period_choice)

if df is not None:
    last = df.iloc[-1]
    info_data = get_stock_info(target_ticker) or {}
    
    # 현재가 표시
    current_price = last['Close']
    pct_change = ((current_price - df.iloc[-2]['Close']) / df.iloc[-2]['Close'] * 100) if len(df) >= 2 else 0
    
    st.title(f"📈 {target_ticker} Pro Dashboard")
    st.metric(label="현재 주가", value=f"{current_price:,.0f}", delta=f"{pct_change:.2f}%")
    
    # 1. 백테스팅 결과 (여기에 뜹니다!)
    df_res, history, total_ret, win_rate = run_backtest(df, rsi_buy_level, rsi_sell_level)
    
    st.markdown("#### 🚀 전략 검증 결과 (과거 시뮬레이션)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("누적 수익률", f"{total_ret:.2f}%")
    m2.metric("승률", f"{win_rate:.1f}%")
    m3.metric("매매 횟수", f"{len(history)}회")
    bh_ret = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0] * 100
    m4.metric("존버(Buy&Hold) 수익률", f"{bh_ret:.2f}%", help="그냥 사서 가만히 있었을 때 수익률")
    
    st.divider()
    display_fundamental_metrics(info_data)

    # 2. 차트
    st.subheader("📊 기술적 분석 차트")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
    
    # 캔들
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="주가"), row=1, col=1)
    
    # 매매 마커 (백테스팅 시각화)
    buys = df_res[df_res['Signal'] == 'Buy']
    sells = df_res[df_res['Signal'] == 'Sell']
    fig.add_trace(go.Scatter(x=buys.index, y=buys['Low']*0.97, mode='markers', marker=dict(symbol='triangle-up', size=12, color='lime'), name="매수"), row=1, col=1)
    fig.add_trace(go.Scatter(x=sells.index, y=sells['High']*1.03, mode='markers', marker=dict(symbol='triangle-down', size=12, color='red'), name="매도"), row=1, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name="RSI", line=dict(color='orange')), row=2, col=1)
    fig.add_hline(y=rsi_buy_level, line_dash="dot", line_color="green", row=2, col=1)
    fig.add_hline(y=rsi_sell_level, line_dash="dot", line_color="red", row=2, col=1)
    
    fig.update_layout(height=700, template="plotly_dark", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    # 3. Gems 연동 및 저장
    st.divider()
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.subheader("🚀 Deep Research 데이터 팩")
        with st.expander("데이터 복사하기", expanded=True):
            news_txt = get_robust_news(target_ticker)
            if "데이터 없음" in news_txt or "오류" in news_txt:
                news_guide = "⚠️ 뉴스 수집 불가. 구글 검색으로 보완 필수."
            else:
                news_guide = ""
                
            pack = f"""[원주 퀀트 데이터팩: {target_ticker}]\n- 현재가: {current_price:,.0f}\n- RSI: {last['RSI']:.1f}\n- 백테스트 수익률: {total_ret:.2f}% (승률 {win_rate:.1f}%)\n- 뉴스:\n{news_txt}\n{news_guide}\n\n위 데이터를 바탕으로 구글 검색을 통해 심층 분석해줘. 손절가 필수."""
            st.code(pack, language="markdown")
            
    with c2:
        st.subheader("💾 기록 저장")
        if st.button("구글 시트에 저장"):
            # 저장 데이터: 시간, 종목, 가격, RSI, 백테스트 수익률
            data_row = [
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                target_ticker,
                float(last['Close']),
                float(last['RSI']),
                f"{total_ret:.2f}%"
            ]
            save_to_google_sheet(sheet_url, data_row)

    st.divider()
    st.caption("💎 원주 퀀트 연구소 v5.8 - Lite & Pro")
